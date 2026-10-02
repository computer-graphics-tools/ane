#![doc = include_str!("../../../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]

#[macro_use]
mod raw_message;

#[cfg(not(target_endian = "little"))]
compile_error!("ANE tensor storage requires a little-endian target");

mod apple_neural_engine;
mod compilation_cache;
mod compilation_descriptor;
mod compilation_report;
mod completion;
mod completion_state;
mod core;
mod device;
mod error;
mod executable;
mod execution_descriptor;
mod graph;
mod io_surface;
mod ir;
mod loaded_program;
mod native_outputs;
mod ops;
mod request;
mod request_cache;
mod shared_event;
mod submission;
mod tensor_data;
mod unavailable_interface;
mod variable_data;

pub use apple_neural_engine::AneError;
pub use compilation_cache::CompilationCache;
pub use compilation_descriptor::CompilationDescriptor;
pub use compilation_report::CompilationReport;
pub use core::{DataType, PadFillMode, PadMode, PoolType, WeightDataType};
pub use device::{DeviceError, DeviceInfo};
pub use error::Error;
pub use executable::Executable;
pub use execution_descriptor::ExecutionDescriptor;
pub use graph::{Graph, GraphError, Operation, Tensor};
pub use io_surface::{IOSurfaceError, IOSurfaceExt};
pub use ir::{IrError, Program};
pub use objc2_foundation::NSQualityOfService;
pub use objc2_io_surface::IOSurface;
pub use ops::{
    BlockwiseQuantization, Convolution2dDescriptor, ConvolutionTranspose2dDescriptor,
    CoordinateMode, Pooling2dDescriptor, SamplingDescriptor, SamplingMode,
};
pub use shared_event::{SharedEvent, SharedEventError};
pub use submission::Submission;
pub use tensor_data::{
    LockedSlice, LockedSliceMut, TensorData, TensorDataError, TensorElement, TensorSpec,
};
pub use unavailable_interface::UnavailableInterface;

use core::{logical_shape, padded_shape};
use error::require;
use loaded_program::LoadedProgram;
use native_outputs::NativeOutputs;
use request_cache::RequestCache;
use variable_data::VariableData;
