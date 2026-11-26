//! GPU kernels that pack activations, stage RHT int4 weights as int8 rows, and unpack results.
//! The host encodes each [`Launch`] on its Metal 4 encoder and tracks its declared resource hazards.
//!
//! [`W4`] is affine group-32 int4 data: codes store `[rows, columns / 2]` bytes in the declared
//! [`W4Layout`] (even column in the low nibble), scales are bf16 `[columns / 32, scale_stride]`, and
//! zero points are `[columns / 32, scale_stride / 2]` (even row in the low nibble). Values dequantize
//! as `scale * (code - zero_point)`.

use std::ffi::c_void;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ComputeCommandEncoder, MTLCompileOptions, MTLComputePipelineState,
    MTLDevice, MTLLanguageVersion, MTLLibrary, MTLSize,
};

use super::{INPUT_CHANNELS, gpu::Buffer};
use crate::TensorData;

const SOURCE: &str = include_str!("../w8a8.metal");
const THREADS: usize = 256;
/// Values in one Hadamard chunk.
const CHUNK: u32 = INPUT_CHANNELS as u32;

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

/// RHT int4 weights and their sign vectors (`f32` unless noted).
#[derive(Clone, Copy, Debug)]
pub struct LayerRegions {
    /// Fused value/gate rows for an FFN, or the linear weights.
    pub up: W4,
    /// Signs applied while staging the up rows.
    pub pre_signs: Region,
    /// Signs applied when unpacking.
    pub output_signs: Region,
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
    pub w4: W4,
    /// Gate output signs (`f32`).
    pub gate_signs: Region,
    /// Down input signs times value output signs, packed as bits.
    pub pre_bits: Region,
    /// Intermediate rotation signs, packed as bits.
    pub intermediate_bits: Region,
}

/// One compute dispatch for the host adapter to encode.
pub struct Launch<'a> {
    /// Trace label.
    pub name: &'static str,
    pipeline: &'a ProtocolObject<dyn MTLComputePipelineState>,
    /// Argument regions and write flags, in shader order. The adapter must honor these hazards.
    pub buffers: Vec<(Region, bool)>,
    /// Values read after the buffer arguments; empty if the shader has no constants.
    pub constants: Vec<u32>,
    groups: [usize; 3],
}

impl<'a> Launch<'a> {
    /// Builds a launch from its read/write regions and grid.
    fn new(
        name: &'static str,
        pipeline: &'a Pipeline,
        reads: &[Region],
        writes: &[Region],
        constants: &[u32],
        groups: [usize; 3],
    ) -> Self {
        let buffers = reads.iter().map(|region| (*region, false));
        let buffers = buffers.chain(writes.iter().map(|region| (*region, true)));
        Self {
            name,
            pipeline,
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
        encoder.setComputePipelineState(self.pipeline);
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

/// Metal pipelines shared by programs on one device.
#[derive(Clone)]
pub struct Kernels {
    pack: Pipeline,
    transpose: Pipeline,
    unpack: Pipeline,
    stage_gate_signs: Pipeline,
    stage_up: Pipeline,
    /// One per chunk count (1 to 3).
    stage_down: [Pipeline; 3],
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
            pack: pipeline("pack_activations")?,
            transpose: pipeline("transpose_codes")?,
            unpack: pipeline("unpack_activations")?,
            stage_gate_signs: pipeline("stage_gate_signs")?,
            stage_up: pipeline("stage_up_rows")?,
            stage_down: [
                pipeline("stage_down_rows_1")?,
                pipeline("stage_down_rows_2")?,
                pipeline("stage_down_rows_3")?,
            ],
        })
    }

    /// Packs token-major bf16 input into signed Hadamard int8 codes and token scales.
    pub fn pack(
        &self,
        input: Region,
        signs: Region,
        codes: Region,
        x: &Surface,
        tx: &Surface,
        tokens: u32,
    ) -> [Launch<'_>; 2] {
        let tiles = (tokens as usize).div_ceil(32);
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
    ) -> Launch<'_> {
        let constants = [
            layer.width,
            tokens,
            y.row_stride,
            layer.down.is_some() as u32,
        ];
        let groups = [layer.width as usize / 32, (tokens as usize).div_ceil(32), 1];
        Launch::new(
            "AneUnpack",
            &self.unpack,
            &[y.region, tx.region, layer.output_signs],
            &[output],
            &constants,
            groups,
        )
    }

    /// Writes half-scaled gate signs to the FFN surface.
    pub fn stage_gate_signs(&self, signs: Region, zg: &Surface, rows: u32) -> Launch<'_> {
        Launch::new(
            "AneStageGateSigns",
            &self.stage_gate_signs,
            &[signs],
            &[zg.region],
            &[rows, zg.row_stride],
            [(rows as usize).div_ceil(THREADS), 1, 1],
        )
    }

    /// Stages rows of the 4096-column up projection, applying signs and per-row quantization.
    pub fn stage_up(
        &self,
        w4: W4,
        pre_signs: Region,
        weights: &Surface,
        scales: &Surface,
        first_row: u32,
        rows: u32,
    ) -> Launch<'_> {
        let constants = [
            first_row,
            w4.scale_stride,
            weights.row_stride,
            scales.row_stride,
            CHUNK / 8,
            w4.layout as u32,
        ];
        Launch::new(
            "AneStageUp",
            &self.stage_up,
            &[w4.codes, w4.scales, w4.zero_points, pre_signs],
            &[weights.region, scales.region],
            &constants,
            [rows as usize, 1, 1],
        )
    }

    /// Stages down-projection rows from one to three 4096-column chunks, with per-row quantization.
    pub fn stage_down(
        &self,
        down: &DownRegions,
        weights: &Surface,
        scales: &Surface,
        hidden: u32,
        columns: u32,
    ) -> Launch<'_> {
        let w4 = down.w4;
        let constants = [
            w4.scale_stride,
            weights.row_stride,
            scales.row_stride,
            columns / 8,
            w4.layout as u32,
        ];
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
    }
}
