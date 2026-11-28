//! GPU kernels that pack activations, stage RHT int4 weights as int8 rows, and unpack results.
//! The host encodes each [`Launch`] on its Metal 4 encoder and tracks its declared resource hazards.
//!
//! [`W4`] is affine group-32 int4 data: codes store `[rows, columns / 2]` bytes in the declared
//! [`W4Layout`] (even column in the low nibble), scales are bf16 `[columns / 32, scale_stride]`, and
//! zero points are `[columns / 32, scale_stride / 2]` (even row in the low nibble). Values dequantize
//! as `scale * (code - zero_point)`.

use std::{
    collections::HashMap,
    ffi::c_void,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ComputeCommandEncoder, MTLCompileOptions, MTLComputePipelineState,
    MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLanguageVersion, MTLLibrary, MTLSize,
};

use super::{INPUT_CHANNELS, gpu::Buffer};
use crate::TensorData;

const SOURCE: &str = concat!(
    include_str!("../w8a8.metal"),
    "\n",
    include_str!("stage_compact.metal")
);
const THREADS: usize = 256;
/// Values in one Hadamard chunk.
const CHUNK: u32 = INPUT_CHANNELS as u32;
const BLOCK: u32 = 512;

/// A GPU address and accessible byte length.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub address: u64,
    pub len: usize,
}

/// Packed int4 code order in the source weight buffer.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum W4Layout {
    #[default]
    RowMajor,
    InterleavedRows,
    MxuTiled,
}

/// Layout of compact U8 and U2 source codes after activation preparation.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WeightCodeLayout {
    #[default]
    RowMajor,
    MxuTiled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OutputBasis {
    Identity,
    Hadamard32,
}

/// An ANE tensor, its Metal alias, and its row stride in elements.
pub struct Surface {
    /// Must drop before the memory it aliases.
    pub _alias: Buffer,
    pub data: TensorData,
    pub region: Region,
    pub row_stride: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct W4 {
    pub codes: Region,
    pub scales: Region,
    pub zero_points: Region,
    /// Elements between consecutive groups of `scales`.
    pub scale_stride: u32,
    pub layout: W4Layout,
}

/// A compact source matrix that the queue adapter decodes into reusable ANE rows.
#[derive(Clone, Copy, Debug)]
pub enum WeightSource {
    W4(W4),
    SymmetricU8 {
        codes: Region,
        scales: Region,
        scale_stride: u32,
        group_size: u32,
        layout: WeightCodeLayout,
        signed_codes: bool,
    },
    MlxU2ScaleBias {
        codes: Region,
        scales: Region,
        biases: Region,
        scale_stride: u32,
        group_size: u32,
        layout: WeightCodeLayout,
        signed_codes: bool,
    },
}

/// One compact source matrix and the transforms that give it its model basis.
#[derive(Clone, Copy)]
pub struct WeightLayer<'a> {
    pub source: WeightSource,
    pub rows: u32,
    pub columns: u32,
    pub input_rotation: Rotation<'a>,
    pub output_rotation: Option<Rotation<'a>>,
}

#[derive(Clone, Copy)]
pub struct Rotation<'a> {
    pub block_size: u32,
    pub signs: &'a [i32],
}

/// RHT int4 weights and their sign vectors (`f32` unless noted).
#[derive(Clone, Copy, Debug)]
pub struct LayerRegions {
    /// Fused value/gate rows for an FFN, or the linear weights.
    pub up: WeightSource,
    /// Signs applied while staging the up rows.
    pub pre_signs: Region,
    /// Signs applied when unpacking.
    pub output_signs: Region,
    pub output_basis: OutputBasis,
    /// Source input rotation width for the up projection.
    pub input_block: u32,
    /// Up rows: FFN intermediate channels or linear outputs.
    pub channels: u32,
    /// FFN hidden width or linear input width.
    pub width: u32,
    /// Present only for an FFN.
    pub down: Option<DownRegions>,
}

/// FFN down-projection data.
#[derive(Clone, Copy, Debug)]
pub struct DownRegions {
    pub source: WeightSource,
    pub input_block: u32,
    pub pre_signs: Region,
    pub post_signs: Region,
    /// Gate output signs (`f32`).
    pub gate_signs: Region,
    /// Down input signs times value output signs, packed as bits.
    pub pre_bits: Region,
    /// Intermediate rotation signs, packed as bits.
    pub intermediate_bits: Region,
}

/// One compute dispatch for the host adapter to encode.
pub struct Launch {
    /// Trace label.
    pub name: &'static str,
    pipeline: Pipeline,
    /// Argument regions and write flags, in shader order. The adapter must honor these hazards.
    pub buffers: Vec<(Region, bool)>,
    /// Values read after the buffer arguments; empty if the shader has no constants.
    pub constants: Vec<u32>,
    groups: [usize; 3],
}

impl Launch {
    /// Builds a launch from its read/write regions and grid.
    fn new(
        name: &'static str,
        pipeline: &Pipeline,
        reads: &[Region],
        writes: &[Region],
        constants: &[u32],
        groups: [usize; 3],
    ) -> Self {
        let buffers = reads.iter().map(|region| (*region, false));
        let buffers = buffers.chain(writes.iter().map(|region| (*region, true)));
        Self {
            name,
            pipeline: pipeline.clone(),
            buffers: buffers.collect(),
            constants: constants.to_vec(),
            groups,
        }
    }

    /// Encodes the dispatch, with constants at `constants` when needed.
    ///
    /// # Safety
    ///
    /// `encoder`, `table`, the pipeline, constants, and regions use the same device; the table is
    /// bound to the encoder. If constants are present, their address is aligned and large enough.
    /// Regions cover shader accesses, resources stay live and resident through completion, and the
    /// adapter honors [`Self::buffers`] in order.
    pub unsafe fn encode(&self, encoder: *mut c_void, table: *mut c_void, constants: u64) {
        // SAFETY: forwarded to the caller.
        let (encoder, table) = unsafe {
            (
                &*encoder.cast::<ProtocolObject<dyn MTL4ComputeCommandEncoder>>(),
                &*table.cast::<ProtocolObject<dyn MTL4ArgumentTable>>(),
            )
        };
        for (index, (region, _)) in self.buffers.iter().enumerate() {
            unsafe { table.setAddress_atIndex(region.address, index) };
        }
        unsafe { table.setAddress_atIndex(constants, self.buffers.len()) };
        encoder.setComputePipelineState(&self.pipeline);
        let [width, height, depth] = self.groups;
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width,
                height,
                depth,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
    }
}

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct StageKey {
    source_format: u32,
    source_layout: u32,
    signed_codes: bool,
    source_block: u32,
    target_block: u32,
    has_post_signs: bool,
    blocks_per_simdgroup: u32,
}

fn stage_key(
    source: WeightSource,
    columns: u32,
    source_block: u32,
    target_block: u32,
    has_post_signs: bool,
) -> Result<Option<StageKey>, String> {
    let (source_format, layout, signed_codes) = match source {
        WeightSource::W4(_) => return Ok(None),
        WeightSource::SymmetricU8 {
            layout,
            signed_codes,
            ..
        } => (0, layout, signed_codes),
        WeightSource::MlxU2ScaleBias {
            layout,
            signed_codes,
            ..
        } => (1, layout, signed_codes),
    };
    if !matches!(source_block, 32 | 128) {
        return Err(format!("unsupported ANE stage source block {source_block}"));
    }
    let blocks_per_simdgroup = match target_block {
        CHUNK if columns == CHUNK && !has_post_signs => 1,
        CHUNK => return Err(format!("unsupported ANE up-stage shape {columns}")),
        BLOCK if columns > 0 && columns.is_multiple_of(BLOCK) => columns.div_ceil(CHUNK),
        BLOCK => return Err(format!("unsupported ANE down-stage shape {columns}")),
        _ => return Err(format!("unsupported ANE target block {target_block}")),
    };
    if !(1..=5).contains(&blocks_per_simdgroup) {
        return Err(format!("unsupported ANE down-stage shape {columns}"));
    }
    Ok(Some(StageKey {
        source_format,
        source_layout: layout as u32,
        signed_codes,
        source_block,
        target_block,
        has_post_signs,
        blocks_per_simdgroup,
    }))
}

fn stage_source_regions(
    source: WeightSource,
) -> Result<(Region, Region, Region, u32, u32), String> {
    match source {
        WeightSource::W4(_) => Err("W4 weights use the existing ANE staging kernels".into()),
        WeightSource::SymmetricU8 {
            codes,
            scales,
            scale_stride,
            group_size,
            ..
        } => Ok((codes, scales, scales, scale_stride, group_size)),
        WeightSource::MlxU2ScaleBias {
            codes,
            scales,
            biases,
            scale_stride,
            group_size,
            ..
        } => Ok((codes, scales, biases, scale_stride, group_size)),
    }
}

fn set_stage_u32(values: &MTLFunctionConstantValues, value: u32, index: usize) {
    // SAFETY: the value pointer is valid for the duration of this synchronous setter call.
    unsafe {
        values.setConstantValue_type_atIndex(
            NonNull::from(&value).cast(),
            MTLDataType::UInt,
            index,
        );
    }
}

fn set_stage_bool(values: &MTLFunctionConstantValues, value: bool, index: usize) {
    // SAFETY: the value pointer is valid for the duration of this synchronous setter call.
    unsafe {
        values.setConstantValue_type_atIndex(
            NonNull::from(&value).cast(),
            MTLDataType::Bool,
            index,
        );
    }
}

pub struct PackDimensions {
    pub tokens: u32,
    pub width: u32,
}

pub struct UpStageRows {
    pub width: u32,
    pub first_row: u32,
    pub rows: u32,
}

pub struct CompactStageRows {
    pub source_row_offset: u32,
    pub rows: u32,
    pub columns: u32,
    pub source_block: u32,
    pub pre_signs: Region,
    pub target_block: u32,
    pub post_signs: Option<Region>,
}

/// Metal pipelines shared by programs on one device.
#[derive(Clone)]
pub struct Kernels {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    compact_stage: Arc<Mutex<HashMap<StageKey, Pipeline>>>,
    pack: Pipeline,
    pack_blocks: [Pipeline; 2],
    transpose: Pipeline,
    transpose_blocks: Pipeline,
    unpack: Pipeline,
    unpack_identity: Pipeline,
    stage_gate_signs: Pipeline,
    stage_up: Pipeline,
    stage_up_blocks: [Pipeline; 2],
    /// One per chunk count (1 to 3).
    stage_down: [Pipeline; 3],
    /// One per number of 512-column blocks assigned to each simdgroup (1 to 5).
    stage_down_blocks: [Pipeline; 5],
}

impl Kernels {
    /// Compiles the Metal 4 source for `device`.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, String> {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version4_0);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
            .map_err(|error| error.to_string())?;
        let pipeline = |name: &str| -> Result<Pipeline, String> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or(name.to_string())?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| error.to_string())
        };
        Ok(Self {
            device: device.retain(),
            library: library.clone(),
            compact_stage: Arc::new(Mutex::new(HashMap::new())),
            pack: pipeline("pack_activations")?,
            pack_blocks: [
                pipeline("pack_activations_blocks_1")?,
                pipeline("pack_activations_blocks_2")?,
            ],
            transpose: pipeline("transpose_codes")?,
            transpose_blocks: pipeline("transpose_codes_blocks")?,
            unpack: pipeline("unpack_activations")?,
            unpack_identity: pipeline("unpack_activations_identity")?,
            stage_gate_signs: pipeline("stage_gate_signs")?,
            stage_up: pipeline("stage_up_rows")?,
            stage_up_blocks: [
                pipeline("stage_up_rows_blocks_1")?,
                pipeline("stage_up_rows_blocks_2")?,
            ],
            stage_down: [
                pipeline("stage_down_rows_1")?,
                pipeline("stage_down_rows_2")?,
                pipeline("stage_down_rows_3")?,
            ],
            stage_down_blocks: [
                pipeline("stage_down_rows_blocks_1")?,
                pipeline("stage_down_rows_blocks_2")?,
                pipeline("stage_down_rows_blocks_3")?,
                pipeline("stage_down_rows_blocks_4")?,
                pipeline("stage_down_rows_blocks_5")?,
            ],
        })
    }

    /// Prepares the one compact source/transform variant needed by a model layer.
    pub fn prepare_weight_stage(
        &self,
        source: WeightSource,
        columns: u32,
        source_block: u32,
        target_block: u32,
        has_post_signs: bool,
    ) -> Result<(), String> {
        let Some(key) = stage_key(source, columns, source_block, target_block, has_post_signs)?
        else {
            return Ok(());
        };
        let mut cache = self
            .compact_stage
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if cache.contains_key(&key) {
            return Ok(());
        }

        let values = MTLFunctionConstantValues::new();
        set_stage_u32(&values, key.source_format, 0);
        set_stage_u32(&values, key.source_layout, 1);
        set_stage_bool(&values, key.signed_codes, 2);
        set_stage_u32(&values, key.source_block, 3);
        set_stage_bool(&values, key.has_post_signs, 4);
        let name = if key.target_block == CHUNK {
            "stage_compact_up_rows".to_string()
        } else {
            format!(
                "stage_compact_down_rows_blocks_{}",
                key.blocks_per_simdgroup
            )
        };
        let function = self
            .library
            .newFunctionWithName_constantValues_error(&NSString::from_str(&name), &values)
            .map_err(|error| error.to_string())?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| error.to_string())?;
        cache.insert(key, pipeline);
        Ok(())
    }

    /// Builds a dispatch using a variant prepared during model binding.
    pub fn stage_weight(
        &self,
        source: WeightSource,
        stage: CompactStageRows,
        weights: &Surface,
        scales: &Surface,
    ) -> Result<Launch, String> {
        let CompactStageRows {
            source_row_offset,
            rows,
            columns,
            source_block,
            pre_signs,
            target_block,
            post_signs,
        } = stage;
        let key = stage_key(
            source,
            columns,
            source_block,
            target_block,
            post_signs.is_some(),
        )?
        .ok_or_else(|| "W4 weights use the existing ANE staging kernels".to_string())?;
        let pipeline = self
            .compact_stage
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&key)
            .cloned()
            .ok_or_else(|| {
                "compact weight staging variant was not prepared at bind time".to_string()
            })?;
        let (codes, source_scales, biases, scale_stride, group_size) =
            stage_source_regions(source)?;
        let post_signs = post_signs.unwrap_or(pre_signs);
        let constants = [
            source_row_offset,
            rows,
            columns,
            scale_stride,
            group_size,
            weights.row_stride,
            scales.row_stride,
        ];
        Ok(Launch::new(
            "AneStageCompact",
            &pipeline,
            &[codes, source_scales, biases, pre_signs, post_signs],
            &[weights.region, scales.region],
            &constants,
            [rows as usize, 1, 1],
        ))
    }

    /// Packs token-major bf16 input into signed Hadamard int8 codes and token scales.
    pub fn pack(
        &self,
        input: Region,
        signs: Region,
        codes: Region,
        x: &Surface,
        tx: &Surface,
        dimensions: PackDimensions,
    ) -> [Launch; 2] {
        let PackDimensions { tokens, width } = dimensions;
        let tiles = (tokens as usize).div_ceil(32);
        if width == CHUNK {
            [
                Launch::new(
                    "AnePack",
                    &self.pack,
                    &[input, signs],
                    &[codes, tx.region],
                    &[],
                    [tokens as usize, 1, 1],
                ),
                Launch::new(
                    "AneTranspose",
                    &self.transpose,
                    &[codes],
                    &[x.region],
                    &[tokens, x.row_stride],
                    [CHUNK as usize / 32, tiles, 1],
                ),
            ]
        } else {
            let blocks = width / BLOCK;
            let blocks_per_simdgroup = blocks.div_ceil((THREADS / 32) as u32);
            let pipeline_index = (blocks_per_simdgroup - 1) as usize;
            [
                Launch::new(
                    "AnePack",
                    &self.pack_blocks[pipeline_index],
                    &[input, signs],
                    &[codes, tx.region],
                    &[width, blocks],
                    [tokens as usize, 1, 1],
                ),
                Launch::new(
                    "AneTranspose",
                    &self.transpose_blocks,
                    &[codes],
                    &[x.region],
                    &[tokens, x.row_stride, width],
                    [width as usize / 32, tiles, 1],
                ),
            ]
        }
    }

    /// Unpacks results to token-major bf16, applying token/intermediate scales and output signs.
    /// FFN outputs include an intermediate scale row; linear outputs do not.
    pub fn unpack(
        &self,
        layer: &LayerRegions,
        y: &Surface,
        tx: &Surface,
        output: Region,
        tokens: u32,
    ) -> Launch {
        let constants = [
            layer.width,
            tokens,
            y.row_stride,
            layer.down.is_some() as u32,
        ];
        let groups = [layer.width as usize / 32, (tokens as usize).div_ceil(32), 1];
        match layer.output_basis {
            OutputBasis::Hadamard32 => Launch::new(
                "AneUnpack",
                &self.unpack,
                &[y.region, tx.region, layer.output_signs],
                &[output],
                &constants,
                groups,
            ),
            OutputBasis::Identity => Launch::new(
                "AneUnpackIdentity",
                &self.unpack_identity,
                &[y.region, tx.region],
                &[output],
                &constants,
                groups,
            ),
        }
    }

    /// Writes half-scaled gate signs to the FFN surface.
    pub fn stage_gate_signs(&self, signs: Region, zg: &Surface, rows: u32) -> Launch {
        Launch::new(
            "AneStageGateSigns",
            &self.stage_gate_signs,
            &[signs],
            &[zg.region],
            &[rows, zg.row_stride],
            [(rows as usize).div_ceil(THREADS), 1, 1],
        )
    }

    /// Stages rows of the up projection, applying signs and per-row quantization.
    pub fn stage_up(
        &self,
        w4: W4,
        pre_signs: Region,
        weights: &Surface,
        scales: &Surface,
        rows: UpStageRows,
    ) -> Launch {
        let UpStageRows {
            width,
            first_row,
            rows,
        } = rows;
        let blocks = width / BLOCK;
        let code_row_words = width / 8;
        let constants = [
            first_row,
            w4.scale_stride,
            weights.row_stride,
            scales.row_stride,
            code_row_words,
            w4.layout as u32,
        ];
        if width == CHUNK {
            Launch::new(
                "AneStageUp",
                &self.stage_up,
                &[w4.codes, w4.scales, w4.zero_points, pre_signs],
                &[weights.region, scales.region],
                &constants,
                [rows as usize, 1, 1],
            )
        } else {
            let blocks_per_simdgroup = blocks.div_ceil((THREADS / 32) as u32);
            let pipeline_index = (blocks_per_simdgroup - 1) as usize;
            let block_constants = [
                first_row,
                w4.scale_stride,
                weights.row_stride,
                scales.row_stride,
                code_row_words,
                w4.layout as u32,
                blocks,
            ];
            Launch::new(
                "AneStageUp",
                &self.stage_up_blocks[pipeline_index],
                &[w4.codes, w4.scales, w4.zero_points, pre_signs],
                &[weights.region, scales.region],
                &block_constants,
                [rows as usize, 1, 1],
            )
        }
    }

    /// Stages down-projection rows, with per-row quantization.
    pub fn stage_down(
        &self,
        down: &DownRegions,
        weights: &Surface,
        scales: &Surface,
        hidden: u32,
        columns: u32,
    ) -> Launch {
        let WeightSource::W4(w4) = down.source else {
            unreachable!("the W4 down staging path only accepts W4 sources")
        };
        let constants = [
            w4.scale_stride,
            weights.row_stride,
            scales.row_stride,
            columns / 8,
            w4.layout as u32,
        ];
        if columns.is_multiple_of(CHUNK) && columns / CHUNK <= self.stage_down.len() as u32 {
            Launch::new(
                "AneStageDown",
                &self.stage_down[(columns / CHUNK - 1) as usize],
                &[
                    w4.codes,
                    w4.scales,
                    w4.zero_points,
                    down.pre_bits,
                    down.intermediate_bits,
                ],
                &[weights.region, scales.region],
                &constants,
                [hidden as usize, 1, 1],
            )
        } else {
            let blocks = columns / BLOCK;
            let blocks_per_simdgroup = blocks.div_ceil((THREADS / 32) as u32);
            let pipeline_index = (blocks_per_simdgroup - 1) as usize;
            let block_constants = [
                w4.scale_stride,
                weights.row_stride,
                scales.row_stride,
                columns / 8,
                w4.layout as u32,
                blocks,
            ];
            Launch::new(
                "AneStageDown",
                &self.stage_down_blocks[pipeline_index],
                &[
                    w4.codes,
                    w4.scales,
                    w4.zero_points,
                    down.pre_bits,
                    down.intermediate_bits,
                ],
                &[weights.region, scales.region],
                &block_constants,
                [hidden as usize, 1, 1],
            )
        }
    }
}
