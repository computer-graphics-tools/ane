mod ane_in_memory_model;
mod ane_in_memory_model_descriptor;
mod ane_io_surface_object;
mod ane_request;
mod client;
use client::compile_model;
mod completion;
mod data_type;
pub use data_type::DataType;
mod error;
mod executable;
pub mod graph;
pub mod io_surface;
pub mod ops;
mod request;
mod submission;
mod tensor_data;

pub use error::Error;
pub use executable::{Executable, PreparedRequest};
pub use graph::{
    Convolution2dDescriptor, ConvolutionTranspose2dDescriptor, Graph, MIN_SPATIAL_WIDTH, State,
    Tensor,
};
pub use io_surface::IOSurfaceExt;
pub use objc2_foundation::NSQualityOfService;
pub use objc2_io_surface::IOSurface;
pub use ops::{
    ActivationMode, ActivationOp, ConcatOp, ConstantOp, ConvOp, DeconvOp, ElementwiseOp,
    ElementwiseOpType, FlattenOp, InnerProductOp, InstanceNormOp, MatmulOp, MilProgram, Op,
    PadFillMode, PadMode, PaddingOp, PoolType, PoolingOp, ReductionMode, ReductionOp, ReshapeOp,
    ScalarOp, ScalarOpType, SliceBySizeOp, SoftmaxOp, TransposeOp,
};
pub use submission::Submission;
pub use tensor_data::{LockedSlice, LockedSliceMut, TensorData};

fn dimensions(shape: &[usize]) -> [usize; 4] {
    shape
        .try_into()
        .expect("shape must contain four dimensions: batch, channels, height, width")
}

pub fn f32_to_fp16_bytes(values: &[f32]) -> Box<[u8]> {
    let mut bytes = vec![0u8; values.len() * 2];
    for (index, &value) in values.iter().enumerate() {
        let f16 = ops::weights::f32_to_f16(value);
        bytes[index * 2] = (f16 & 0xFF) as u8;
        bytes[index * 2 + 1] = (f16 >> 8) as u8;
    }
    bytes.into_boxed_slice()
}
