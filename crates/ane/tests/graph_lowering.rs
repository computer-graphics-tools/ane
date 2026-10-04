use ane::{
    BlockwiseQuantization, Convolution2dDescriptor, ConvolutionTranspose2dDescriptor, DataType,
    GeluMode, Graph, GraphError, PadFillMode, PadMode, Palettization, Pooling2dDescriptor,
    ResizeSamplingMode, SamplingDescriptor, Tensor, TensorData, WeightDataType,
};

#[path = "graph_lowering/case.rs"]
mod case;
use case::Case;

fn h<const RANK: usize>(g: &Graph, shape: [usize; RANK]) -> Result<Tensor, GraphError> {
    g.placeholder(shape, DataType::Float16)
}
fn x4(g: &Graph) -> Result<Tensor, GraphError> {
    h(g, [1, 2, 4, 4])
}
fn typed<const RANK: usize>(
    g: &Graph,
    shape: [usize; RANK],
    dtype: DataType,
) -> Result<Tensor, GraphError> {
    g.placeholder(shape, dtype)
}
fn c<const RANK: usize>(g: &Graph, shape: [usize; RANK]) -> Result<Tensor, GraphError> {
    let count = shape.iter().product::<usize>();
    let data: Vec<f32> = (0..count).map(|i| 0.25 + i as f32 * 0.125).collect();
    g.constant(&data, shape)
}
fn int8_blocks() -> BlockwiseQuantization<'static, 2> {
    BlockwiseQuantization::new(WeightDataType::Int8, &[0.1, 0.2, 0.3, 0.4], [4, 1])
}

macro_rules! unary {
    ($($name:ident),* $(,)?) => {
        &[$( (stringify!($name), |g: &Graph| { let x = x4(g)?; Ok(vec![g.$name(&x)?]) }) ),*]
    };
}
macro_rules! binary {
    ($($name:ident),* $(,)?) => {
        &[$( (stringify!($name), |g: &Graph| { let x = x4(g)?; let y = x4(g)?; Ok(vec![g.$name(&x, &y)?]) }) ),*]
    };
}
macro_rules! scalar {
    ($($name:ident($($value:expr),+)),* $(,)?) => {
        &[$( (concat!(stringify!($name), "(", stringify!($($value),+), ")"), |g: &Graph| { let x = x4(g)?; Ok(vec![g.$name(&x, $($value),+)?]) }) ),*]
    };
}
macro_rules! reduce {
    ($($name:ident),* $(,)?) => {
        &[$( (stringify!($name), |g: &Graph| { let x = x4(g)?; Ok(vec![g.$name(&x, &[-1, -2])?]) }) ),*]
    };
}

const UNARY: &[(&str, Case)] = unary!(
    absolute,
    atan,
    ceil,
    cos,
    erf,
    exponent,
    exponent_base2,
    floor,
    round,
    sign,
    sin,
    square_root,
    square,
    relu,
    relu6,
    sigmoid,
    silu,
    softplus,
    softsign,
    tanh,
);
const BINARY: &[(&str, Case)] = binary!(
    addition,
    subtraction,
    multiplication,
    division,
    power,
    maximum,
    minimum,
    equal,
    not_equal,
    less_than,
    less_than_or_equal_to,
    greater_than,
    greater_than_or_equal_to,
);
const SCALAR: &[(&str, Case)] = scalar!(
    logarithm(1e-4),
    reciprocal(1e-4),
    reciprocal_square_root(1e-4),
    l2_normalize(1e-5),
    soft_max(-1),
    clamp(-1.0, 1.0),
    leaky_relu(0.1),
    elu(1.0),
    thresholded_relu(0.5),
    linear_activation(2.0, 1.0),
    hard_sigmoid(0.2, 0.5),
    scaled_tanh(1.0, 2.0),
    clamped_relu(0.1, 6.0),
    gelu(GeluMode::Exact),
    gelu(GeluMode::TanhApproximation),
    gelu(GeluMode::SigmoidApproximation),
    local_response_norm(2, 1e-4, 0.75, 1.0),
    reduction_arg_maximum(-1),
    reduction_arg_minimum(1),
    resize_nearest([8, 8]),
    resize_bilinear([8, 8], ResizeSamplingMode::Default),
    resize_bilinear([8, 8], ResizeSamplingMode::StrictAlignCorners),
    resize_bilinear([8, 8], ResizeSamplingMode::AlignCorners),
    resize_bilinear([8, 8], ResizeSamplingMode::OffsetCorners),
    resize_bilinear([8, 8], ResizeSamplingMode::UnalignCorners),
    space_to_depth(2),
    space_to_batch([2, 2], [0, 0, 0, 0]),
    pad([1, 1, 1, 1], PadFillMode::Constant, 0.5),
    pad([1, 1, 1, 1], PadFillMode::Reflect, 0.0),
    pad([1, 1, 1, 1], PadFillMode::Replicate, 0.0),
    reverse(&[1, 3]),
    tile(&[1, 2, 1, 1]),
    transpose([0, 2, 1, 3]),
    reshape([2, 16]),
    slice([0, 0, 1, 1], [1, 2, 2, 2]),
    strided_slice(&[0, 0, 0, 1], &[1, 2, 4, 4], &[1, 1, 2, 2]),
    max_pooling_2d(&Pooling2dDescriptor::new([2, 2], [2, 2])),
    avg_pooling_2d(&Pooling2dDescriptor::new([3, 3], [1, 1])),
    l2_norm_pooling_2d(&Pooling2dDescriptor::new([2, 2], [2, 2])),
);
const REDUCE: &[(&str, Case)] = reduce!(
    reduction_sum,
    reduction_mean,
    reduction_minimum,
    reduction_maximum,
    reduction_l1_norm,
    reduction_l2_norm,
    reduction_log_sum,
    reduction_log_sum_exp,
    reduction_sum_square,
);

const SPECIAL: &[(&str, Case)] = &[
    ("placeholder_f32", |g| {
        let x = g.placeholder([2, 3], DataType::Float32)?;
        Ok(vec![g.relu(&x)?])
    }),
    ("integer_parameter", |g| {
        Ok(vec![g.placeholder([1, 1, 1, 1], DataType::Int32)?])
    }),
    ("constant", |g| {
        let x = h(g, [2, 3])?;
        let k = c(g, [2, 3])?;
        Ok(vec![g.addition(&x, &k)?])
    }),
    ("constant_f16_bytes", |g| {
        let x = h(g, [2, 3])?;
        let k = g.constant_with_bytes(&[0x3c; 12], [2, 3], DataType::Float16)?;
        Ok(vec![g.addition(&x, &k)?])
    }),
    ("logical_and", |g| {
        let x = typed(g, [2, 3], DataType::Bool)?;
        let y = typed(g, [2, 3], DataType::Bool)?;
        Ok(vec![g.logical_and(&x, &y)?])
    }),
    ("select", |g| {
        let b = typed(g, [1, 2, 4, 4], DataType::Bool)?;
        let x = x4(g)?;
        let y = x4(g)?;
        Ok(vec![g.select(&b, &x, &y)?])
    }),
    ("prelu", |g| {
        let x = x4(g)?;
        let a = c(g, [2])?;
        Ok(vec![g.prelu(&x, &a)?])
    }),
    ("softplus_parametric", |g| {
        let x = x4(g)?;
        let a = c(g, [2])?;
        let b = c(g, [2])?;
        Ok(vec![g.softplus_parametric(&x, &a, &b)?])
    }),
    ("cast_uint8", |g| {
        let x = x4(g)?;
        Ok(vec![g.cast(&x, DataType::UInt8)?])
    }),
    ("cast_int32", |g| {
        let x = x4(g)?;
        Ok(vec![g.cast(&x, DataType::Int32)?])
    }),
    ("layer_norm", |g| {
        let x = x4(g)?;
        let s = c(g, [4])?;
        let b = c(g, [4])?;
        Ok(vec![g.layer_norm(&x, &[-1], Some(&s), Some(&b), 1e-5)?])
    }),
    ("layer_norm_plain", |g| {
        let x = x4(g)?;
        Ok(vec![g.layer_norm(&x, &[-1, -2], None, None, 1e-5)?])
    }),
    ("instance_norm", |g| {
        let x = x4(g)?;
        let s = c(g, [2])?;
        let b = c(g, [2])?;
        Ok(vec![g.instance_norm(&x, Some(&s), Some(&b), 1e-5)?])
    }),
    ("batch_norm", |g| {
        let x = x4(g)?;
        let m = c(g, [2])?;
        let v = c(g, [2])?;
        let s = c(g, [2])?;
        let b = c(g, [2])?;
        Ok(vec![g.batch_norm(&x, &m, &v, Some(&s), Some(&b), 1e-5)?])
    }),
    ("matmul", |g| {
        let a = h(g, [2, 3])?;
        let b = h(g, [3, 4])?;
        Ok(vec![g.matrix_multiplication(&a, &b, false, false)?])
    }),
    ("matmul_transposed", |g| {
        let a = h(g, [3, 2])?;
        let b = h(g, [4, 3])?;
        Ok(vec![g.matrix_multiplication(&a, &b, true, true)?])
    }),
    ("linear", |g| {
        let x = h(g, [2, 4])?;
        let w = c(g, [3, 4])?;
        let b = c(g, [3])?;
        Ok(vec![g.linear(&x, &w, Some(&b))?])
    }),
    ("attention", |g| {
        let q = h(g, [1, 2, 4, 8])?;
        let k = h(g, [1, 2, 4, 8])?;
        let v = h(g, [1, 2, 4, 8])?;
        Ok(vec![g.scaled_dot_product_attention(&q, &k, &v, None)?])
    }),
    ("attention_mask", |g| {
        let q = h(g, [1, 2, 4, 8])?;
        let k = h(g, [1, 2, 4, 8])?;
        let v = h(g, [1, 2, 4, 8])?;
        let m = h(g, [1, 1, 4, 4])?;
        Ok(vec![g.scaled_dot_product_attention(
            &q,
            &k,
            &v,
            Some(&m),
        )?])
    }),
    ("max_pool_same", |g| {
        let x = x4(g)?;
        let mut descriptor = Pooling2dDescriptor::new([2, 2], [2, 2]);
        descriptor.pad_mode = PadMode::Same;
        Ok(vec![g.max_pooling_2d(&x, &descriptor)?])
    }),
    ("concat", |g| {
        let a = h(g, [2, 3])?;
        let b = h(g, [2, 5])?;
        Ok(vec![g.concat(&[&a, &b], 1, false)?])
    }),
    ("concat_interleave", |g| {
        let a = h(g, [2, 3])?;
        let b = h(g, [2, 3])?;
        Ok(vec![g.concat(&[&a, &b], 1, true)?])
    }),
    ("split", |g| {
        let x = h(g, [2, 6])?;
        g.split(&x, &[2, 4], 1)
    }),
    ("expand_dims", |g| {
        let x = h(g, [2, 3])?;
        Ok(vec![g.expand_dims(&x, &[0])?])
    }),
    ("squeeze", |g| {
        let x = h(g, [1, 2, 3])?;
        Ok(vec![g.squeeze(&x, &[0])?])
    }),
    ("slice_update", |g| {
        let x = x4(g)?;
        let u = h(g, [1, 1, 4, 4])?;
        Ok(vec![g.slice_update(&x, &u, &[0, 1, 0, 0])?])
    }),
    ("dynamic_slice", |g| {
        let x = h(g, [1, 1, 8, 4])?;
        let p = g.placeholder([1, 1, 1, 1], DataType::Int32)?;
        Ok(vec![g.slice_dynamic(&x, &p, 2, 2)?])
    }),
    ("depth_to_space", |g| {
        let x = h(g, [1, 4, 2, 2])?;
        Ok(vec![g.depth_to_space(&x, 2)?])
    }),
    ("pixel_shuffle", |g| {
        let x = h(g, [1, 4, 2, 2])?;
        Ok(vec![g.pixel_shuffle(&x, 2)?])
    }),
    ("batch_to_space", |g| {
        let x = h(g, [4, 1, 2, 2])?;
        Ok(vec![g.batch_to_space(&x, [2, 2], [0, 0, 0, 0])?])
    }),
    ("gather", |g| {
        let x = h(g, [4, 8])?;
        let i = typed(g, [3], DataType::UInt16)?;
        Ok(vec![g.gather(&x, &i, -1)?])
    }),
    ("gather_along_axis", |g| {
        let x = h(g, [4, 8])?;
        let i = typed(g, [4, 2], DataType::UInt16)?;
        Ok(vec![g.gather_along_axis(&x, &i, 1)?])
    }),
    ("topk", |g| {
        let x = h(g, [4, 8])?;
        let (v, i) = g.top_k(&x, 2, -1)?;
        Ok(vec![v, i])
    }),
    ("topk_ascending", |g| {
        let x = h(g, [4, 8])?;
        let (v, i) = g.bottom_k(&x, 3, 0)?;
        Ok(vec![v, i])
    }),
    ("quantize", |g| {
        let x = h(g, [2, 3])?;
        Ok(vec![g.quantize(&x, &[0.1], None, None, DataType::Int8)?])
    }),
    ("quantize_axis", |g| {
        let x = h(g, [2, 3, 4])?;
        Ok(vec![g.quantize(
            &x,
            &[0.1, 0.2],
            Some(&[1, 2]),
            Some(0),
            DataType::UInt8,
        )?])
    }),
    ("dequantize", |g| {
        let q = typed(g, [2, 3], DataType::Int8)?;
        Ok(vec![g.dequantize(&q, &[0.1], Some(&[1]), None)?])
    }),
    ("constexpr_blockwise_shift_scale", |g| {
        let x = h(g, [2, 4])?;
        let w = g.blockwise_weights(&[1; 16], [4, 4], &int8_blocks())?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("constexpr_lut_to_dense", |g| {
        let x = h(g, [2, 4])?;
        let w = g.palettized_weights(
            &[0x1b; 4],
            [4, 4],
            &Palettization::new(2, &[0.0, 0.5, 1.0, 1.5]),
        )?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("constexpr_lut_to_dense_grouped", |g| {
        let x = h(g, [2, 4])?;
        let palette: Vec<f32> = (0..8).map(|i| i as f32 * 0.25).collect();
        let w = g.palettized_weights(
            &[0x1b; 4],
            [4, 4],
            &Palettization {
                group_shape: [2, 1],
                ..Palettization::new(2, &palette)
            },
        )?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("constexpr_sparse_to_dense", |g| {
        let x = h(g, [2, 4])?;
        let w = g.sparse_weights(&[0x55, 0x55], [4, 4], &[1.0; 8])?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("constexpr_sparse_blockwise_shift_scale", |g| {
        let x = h(g, [2, 4])?;
        let w = g.sparse_blockwise_weights(&[1; 8], &[0x55, 0x55], [4, 4], &int8_blocks())?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("constexpr_lut_to_sparse", |g| {
        let x = h(g, [2, 4])?;
        let w = g.sparse_palettized_weights(
            &[0x1b; 2],
            &[0x55, 0x55],
            2,
            [4, 4],
            &[0.0, 0.5, 1.0, 1.5],
        )?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("conv", |g| {
        let x = x4(g)?;
        let w = c(g, [3, 2, 3, 3])?;
        let b = c(g, [3])?;
        let descriptor = Convolution2dDescriptor {
            pad_mode: PadMode::Same,
            ..Default::default()
        };
        Ok(vec![g.convolution_2d(&x, &w, Some(&b), &descriptor)?])
    }),
    ("conv_runtime_weights", |g| {
        let x = x4(g)?;
        let w = h(g, [3, 2, 1, 1])?;
        Ok(vec![g.convolution_2d(
            &x,
            &w,
            None,
            &Convolution2dDescriptor::default(),
        )?])
    }),
    ("conv_transpose", |g| {
        let x = x4(g)?;
        let w = c(g, [2, 3, 3, 3])?;
        let descriptor = ConvolutionTranspose2dDescriptor {
            strides: [2, 2],
            ..Default::default()
        };
        Ok(vec![g.convolution_transpose_2d(
            &x,
            &w,
            None,
            &descriptor,
        )?])
    }),
    ("resample", |g| {
        let x = x4(g)?;
        let k = h(g, [1, 4, 4, 2])?;
        Ok(vec![g.sample_grid(
            &x,
            &k,
            &SamplingDescriptor::default(),
        )?])
    }),
    ("state", |g| {
        let x = h(g, [2, 3])?;
        let v = g.variable_with_data(&[0.5; 6], [2, 3])?;
        let r = g.read_variable(&v)?;
        Ok(vec![g.addition(&x, &r)?])
    }),
    ("state_from_surface", |g| {
        let x = h(g, [2, 3])?;
        let d = TensorData::with_type([2, 3], DataType::Float16)?;
        let v = g.variable_with_tensor_data(&d)?;
        let r = g.read_variable(&v)?;
        Ok(vec![g.addition(&x, &r)?])
    }),
    ("foreign_tensor", |g| {
        let x = x4(g)?;
        let other = Graph::new();
        let y = x4(&other)?;
        Ok(vec![g.addition(&x, &y)?])
    }),
];

const CASES: [&[(&str, Case)]; 5] = [UNARY, BINARY, SCALAR, REDUCE, SPECIAL];
const INVALID: [&str; 3] = ["foreign_tensor", "cast_int32", "conv_runtime_weights"];

#[test]
fn all_operations_emit_mil() {
    let mut failures = Vec::new();
    for (name, case) in CASES.concat() {
        let graph = Graph::new();
        let output = case(&graph);
        if INVALID.contains(&name) {
            assert!(output.is_err());
            continue;
        }
        let mil = output
            .and_then(|output| graph.program(&output, &[DataType::Float32]))
            .and_then(|program| Ok(program.mil()?));
        match mil {
            Ok(mil) => assert!(mil.text.contains("func main<ios18>("), "{name}"),
            Err(error) => failures.push(format!("{name}: {error}")),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn invalid_parameter_does_not_change_the_graph() {
    let graph = Graph::new();
    let input = h(&graph, [2, 64]).unwrap();
    for value in [f32::MAX, 65536.0, -65536.0] {
        assert!(matches!(
            graph.leaky_relu(&input, value),
            Err(GraphError::Ir(ane::IrError::InvalidValue("fp16")))
        ));
        assert!(matches!(
            graph.linear_activation(&input, value, 0.0),
            Err(GraphError::Ir(ane::IrError::InvalidValue("fp16")))
        ));
    }
    let output = graph.relu(&input).unwrap();
    let actual = graph
        .program(&[output], &[DataType::Float16])
        .unwrap()
        .mil()
        .unwrap();
    let clean = Graph::new();
    let input = h(&clean, [2, 64]).unwrap();
    let output = clean.relu(&input).unwrap();
    let expected = clean
        .program(&[output], &[DataType::Float16])
        .unwrap()
        .mil()
        .unwrap();
    assert_eq!(actual.text, expected.text);
    assert_eq!(actual.weights, expected.weights);
}

#[test]
fn matmul_validation_and_transpose_share_shape_rules() {
    let graph = Graph::new();
    let x = h(&graph, [3, 4]).unwrap();
    let wrong = h(&graph, [5, 2]).unwrap();
    assert!(matches!(
        graph.matrix_multiplication(&x, &wrong, false, false),
        Err(GraphError::Ir(ane::IrError::InvalidProgram(_)))
    ));
    let y = h(&graph, [2, 4]).unwrap();
    let result = graph.matrix_multiplication(&x, &y, false, true).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    graph.program(&[result], &[DataType::Float16]).unwrap();
}

#[test]
fn boolean_gather_data_is_rejected() {
    let graph = Graph::new();
    let data = typed(&graph, [4], DataType::Bool).unwrap();
    let indices = typed(&graph, [2], DataType::UInt16).unwrap();
    assert!(matches!(
        graph.gather(&data, &indices, 0),
        Err(GraphError::UnsupportedDataType(_))
    ));
}

#[test]
fn all_operations_compile_for_ane() {
    let expected = [
        ("integer_parameter", "Int32 is only a state position type"),
        (
            "state_from_surface",
            "the bound surface uses an unpadded layout",
        ),
    ];
    let mut failures = Vec::new();
    for (name, case) in CASES.concat() {
        if INVALID.contains(&name) {
            continue;
        }
        let graph = Graph::new();
        let compiled = case(&graph)
            .map_err(ane::Error::from)
            .and_then(|outputs| graph.compile(&outputs, &[], None));
        if let Err(error) = compiled {
            println!("{name}: {error}");
            failures.push(name);
        }
    }
    assert_eq!(failures, expected.map(|(name, _)| name));
}

#[test]
fn constants_must_fit_fp16_unless_explicitly_infinite() {
    let graph = Graph::new();
    assert!(matches!(
        graph.constant(&[70_000.0], [1]),
        Err(GraphError::Ir(ane::IrError::InvalidValue("fp16")))
    ));
    assert!(graph.constant(&[f32::NEG_INFINITY, 65_504.0], [2]).is_ok());
}

#[test]
fn programs_write_at_most_seven_variables() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let x = h(&graph, [1, 1, 1, 64])?;
    let position = graph.placeholder([1], DataType::Int32)?;
    let mut writes = Vec::new();
    for _ in 0..8 {
        let variable = graph.variable_with_data(&[0.; 64], [1, 1, 1, 64])?;
        writes.push(graph.assign_variable_rows(&variable, &x, &position, 0)?);
    }
    let writes: Vec<_> = writes.iter().collect();
    assert!(matches!(
        graph.compile(&[x], &writes, None),
        Err(ane::Error::Graph(GraphError::UnsupportedComposition(_)))
    ));
    Ok(())
}
