#![doc = include_str!("../../../README.md")]
#![deny(unsafe_op_in_unsafe_fn)]

#[macro_use]
mod raw_message;

#[cfg(not(target_endian = "little"))]
compile_error!("ANE tensor storage requires a little-endian target");

mod apple_neural_engine;
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
mod ops;
mod outcome;
mod procedure;
mod request;
mod request_cache;
mod shared_event;
mod state_data;
mod submission;
mod tensor_data;
mod unavailable_interface;

pub use apple_neural_engine::AneError;
pub use compilation_descriptor::CompilationDescriptor;
pub use compilation_report::CompilationReport;
pub use core::{DataType, GeluMode, PadFillMode, PadMode, WeightDataType};
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
    CoordinateMode, Palette, Palettization, Pooling2dDescriptor, ResizeSamplingMode,
    SamplingDescriptor, SamplingMode,
};
pub use shared_event::{SharedEvent, SharedEventError};
pub use submission::Submission;
pub use tensor_data::{
    LockedSlice, LockedSliceMut, TensorData, TensorDataError, TensorElement, TensorSpec,
};
pub use unavailable_interface::UnavailableInterface;

use completion::Completion;
use completion_state::CompletionState;
use core::{logical_shape, padded_shape};
use error::require;
use loaded_program::LoadedProgram;
use outcome::Outcome;
use procedure::Procedure;
use request::Request;
use request_cache::RequestCache;
use state_data::StateData;
use unavailable_interface::ensure_interfaces;
