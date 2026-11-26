//! FFN and linear functions share constants and padded `[channels, tokens]` surfaces across token counts.
//!
//! Feeds: `x, tx, wg, wu, sg, su, zg, wd, sd`. The output's last row carries the intermediate scale.
//! The gate uses `g/2 * (1 + tanh(g/2))` to avoid inaccurate ANE sigmoid results.
//!
//! Linear feeds: `x, w, s`; token scales are applied when unpacking.
//!
//! The FFN program is adapted from Splash `runtime/ops/AneFfn.cpp` (commit 031a25a, Apache License
//! 2.0). Not affiliated with Splash.

use super::{
    INPUT_CHANNELS,
    signs::{intermediate_signs, signed_hadamard_blocks},
};
use crate::{
    CompilationDescriptor, Convolution2dDescriptor, DataType, Error, Executable, Graph, GraphError,
    NSQualityOfService, PadMode, Tensor,
};

const INT8_SCALE: f32 = 1.0 / 128.0;
const INT8_PEAK: f32 = 127.0;
const MIN_TOKEN_PEAK: f32 = 1.0 / 512.0;
const MATMUL_SEGMENT: usize = 2048;
const INTERMEDIATE_ROTATION_BLOCK: usize = 512;

pub const TOKEN_COUNTS: [u32; 4] = [256, 512, 768, 1024];
pub const MAX_TOKENS: usize = TOKEN_COUNTS[TOKEN_COUNTS.len() - 1] as usize;
pub const UP_ROTATION_BLOCK: usize = 32;
/// The widest linear layer, in rows.
pub const MAX_LINEAR_OUTPUT: usize = 16384;
pub const MAX_INTERMEDIATE: usize = 3 * INPUT_CHANNELS;

pub fn compile_ffn(hidden: usize, intermediate: usize) -> Result<Vec<Executable>, Error> {
    let graph = Graph::new();
    let constants = Constants::new(&graph, intermediate)?;
    let outputs = TOKEN_COUNTS
        .iter()
        .map(|rows| ffn_function(&graph, hidden, intermediate, &constants, *rows as usize))
        .collect::<Result<Vec<_>, _>>()?;
    compile(&graph, &outputs)
}

/// Compiles the linear width into one executable per row count.
pub fn compile_linear(output: usize) -> Result<Vec<Executable>, Error> {
    let graph = Graph::new();
    let outputs = TOKEN_COUNTS
        .iter()
        .map(|rows| linear_function(&graph, output, *rows as usize))
        .collect::<Result<Vec<_>, _>>()?;
    compile(&graph, &outputs)
}

fn buffer(
    graph: &Graph,
    shape: [usize; 2],
    data_type: DataType,
    width: usize,
) -> Result<Tensor, GraphError> {
    graph.placeholder_buffer(shape, data_type, data_type.aligned_row_stride(width))
}

fn segments(graph: &Graph, input: &Tensor) -> Result<Vec<Tensor>, GraphError> {
    let (channels, rows) = (input.shape()[0], input.shape()[1]);
    (0..channels.div_ceil(MATMUL_SEGMENT))
        .map(|k| {
            let begin = k * MATMUL_SEGMENT;
            graph.slice(
                input,
                [begin, 0],
                [MATMUL_SEGMENT.min(channels - begin), rows],
            )
        })
        .collect()
}

fn project(
    graph: &Graph,
    weights: &Tensor,
    out: usize,
    inputs: &[Tensor],
) -> Result<Tensor, GraphError> {
    let product = |k: usize, input: &Tensor| -> Result<Tensor, GraphError> {
        let slice = graph.slice(weights, [0, k * MATMUL_SEGMENT], [out, input.shape()[0]])?;
        let dequantized = graph.dequantize(&slice, &[INT8_SCALE], None, None)?;
        graph.matrix_multiplication(&dequantized, input, false, false)
    };
    let mut total = product(0, &inputs[0])?;
    for (k, input) in inputs.iter().enumerate().skip(1) {
        total = graph.addition(&total, &product(k, input)?)?;
    }
    Ok(total)
}

/// `out[c] = sum_i weights[c][i] * input[block(c) + i]`, a grouped 1x1 convolution.
fn block_rotation(graph: &Graph, input: &Tensor, weights: &Tensor) -> Result<Tensor, GraphError> {
    let (channels, rows) = (input.shape()[0], input.shape()[1]);
    let planar = graph.reshape(input, [1, channels, 1, rows])?;
    let descriptor = Convolution2dDescriptor {
        groups: channels / weights.shape()[1],
        strides: [1, 1],
        dilations: [1, 1],
        padding: [0; 4],
        pad_mode: PadMode::Valid,
    };
    let rotated = graph.convolution_2d(&planar, weights, None, &descriptor)?;
    graph.reshape(&rotated, [channels, rows])
}

/// What every function of one program shares: the model stores each constant once.
struct Constants {
    up_rotation: Tensor,
    rotation: Tensor,
    one: Tensor,
    floor: Tensor,
    peak: Tensor,
    unit_ratio: Tensor,
}

impl Constants {
    fn new(graph: &Graph, intermediate: usize) -> Result<Self, GraphError> {
        let up_rotation = signed_hadamard_blocks(&vec![1.0; intermediate], UP_ROTATION_BLOCK);
        let rotation = signed_hadamard_blocks(
            &intermediate_signs(intermediate),
            INTERMEDIATE_ROTATION_BLOCK,
        );
        Ok(Self {
            up_rotation: graph.constant(&up_rotation, [intermediate, UP_ROTATION_BLOCK, 1, 1])?,
            rotation: graph
                .constant(&rotation, [intermediate, INTERMEDIATE_ROTATION_BLOCK, 1, 1])?,
            one: graph.constant(&[1.0], [1, 1])?,
            floor: graph.constant(&[MIN_TOKEN_PEAK], [1, 1])?,
            peak: graph.constant(&[INT8_PEAK], [1, 1])?,
            unit_ratio: graph.constant(&[1.0 / INT8_SCALE / INT8_PEAK], [1, 1])?,
        })
    }
}

fn ffn_function(
    graph: &Graph,
    hidden: usize,
    intermediate: usize,
    constants: &Constants,
    rows: usize,
) -> Result<Tensor, GraphError> {
    let (int8, fp16) = (DataType::Int8, DataType::Float16);
    let x = buffer(graph, [hidden, rows], int8, MAX_TOKENS)?;
    let tx = buffer(graph, [1, rows], fp16, MAX_TOKENS)?;
    let wg = buffer(graph, [intermediate, hidden], int8, hidden)?;
    let wu = buffer(graph, [intermediate, hidden], int8, hidden)?;
    let sg = buffer(graph, [intermediate, 1], fp16, 1)?;
    let su = buffer(graph, [intermediate, 1], fp16, 1)?;
    let zg = buffer(graph, [intermediate, 1], fp16, 1)?;
    let wd = buffer(graph, [hidden, intermediate], int8, intermediate)?;
    let sd = buffer(graph, [hidden, 1], fp16, 1)?;

    let x_dequantized = graph.dequantize(&x, &[INT8_SCALE], None, None)?;
    let x_segments = segments(graph, &x_dequantized)?;
    let gate = graph.multiplication(&project(graph, &wg, intermediate, &x_segments)?, &sg)?;
    let value = graph.multiplication(&project(graph, &wu, intermediate, &x_segments)?, &su)?;
    let gate = block_rotation(graph, &gate, &constants.up_rotation)?;
    let value = block_rotation(graph, &value, &constants.up_rotation)?;

    let scaled = graph.multiplication(&gate, &tx)?;
    let half_gate = graph.multiplication(&scaled, &zg)?;
    let tanh = graph.tanh(&half_gate)?;
    let silu = graph.multiplication(&half_gate, &graph.addition(&tanh, &constants.one)?)?;
    let state = graph.multiplication(&silu, &value)?;
    let rotated = block_rotation(graph, &state, &constants.rotation)?;

    // Per-token requantization to int8 before the down projection.
    let magnitude = graph.absolute(&rotated)?;
    let token_peak = graph.maximum(
        &graph.reduction_maximum(&magnitude, &[0])?,
        &constants.floor,
    )?;
    let inverse = graph.division(&constants.peak, &token_peak)?;
    let quantized = graph.quantize(
        &graph.multiplication(&rotated, &inverse)?,
        &[1.0],
        None,
        None,
        int8,
    )?;
    let requantized = graph.dequantize(&quantized, &[INT8_SCALE], None, None)?;
    let intermediate_scale = graph.multiplication(&token_peak, &constants.unit_ratio)?;
    let down_input = segments(graph, &requantized)?;
    let down = graph.multiplication(&project(graph, &wd, hidden, &down_input)?, &sd)?;
    let result = graph.concat(&[&down, &intermediate_scale], 0, false)?;
    graph.buffer_output(&result, fp16.aligned_row_stride(MAX_TOKENS))
}

/// Builds a linear function for `rows` tokens.
fn linear_function(graph: &Graph, output: usize, rows: usize) -> Result<Tensor, GraphError> {
    let (int8, fp16) = (DataType::Int8, DataType::Float16);
    let x = buffer(graph, [INPUT_CHANNELS, rows], int8, MAX_TOKENS)?;
    let w = buffer(
        graph,
        [MAX_LINEAR_OUTPUT, INPUT_CHANNELS],
        int8,
        INPUT_CHANNELS,
    )?;
    let s = buffer(graph, [MAX_LINEAR_OUTPUT, 1], fp16, 1)?;
    let x_dequantized = graph.dequantize(&x, &[INT8_SCALE], None, None)?;
    let x_segments = segments(graph, &x_dequantized)?;
    let product = project(graph, &w, output, &x_segments)?;
    // Only the leading `output` rows of the scales are read.
    let scales = graph.slice(&s, [0, 0], [output, 1])?;
    let result = graph.multiplication(&product, &scales)?;
    graph.buffer_output(&result, fp16.aligned_row_stride(MAX_TOKENS))
}

fn compile(graph: &Graph, outputs: &[Tensor]) -> Result<Vec<Executable>, Error> {
    let targets: Vec<_> = outputs
        .iter()
        .map(|output| (std::slice::from_ref(output), &[][..]))
        .collect();
    let descriptor = CompilationDescriptor {
        output_types: None,
        quality_of_service: NSQualityOfService::UserInteractive,
    };
    graph.compile_shared(&targets, Some(&descriptor))
}
