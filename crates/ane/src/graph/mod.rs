mod descriptor;
mod tensor;

pub use descriptor::{Convolution2dDescriptor, ConvolutionTranspose2dDescriptor};
pub use tensor::{Graph, MIN_SPATIAL_WIDTH, State, Tensor};
