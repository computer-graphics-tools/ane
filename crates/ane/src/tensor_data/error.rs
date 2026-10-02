use crate::io_surface::IOSurfaceError;
use crate::{DataType, TensorSpec};

#[derive(Clone, Debug, thiserror::Error)]
pub enum TensorDataError {
    #[error("shape {0:?} needs at most four positive Int32 dimensions within the address space")]
    InvalidShape(Box<[usize]>),
    #[error(transparent)]
    IOSurface(#[from] IOSurfaceError),
    #[error("{actual:?} values do not match {expected:?} storage")]
    DataTypeMismatch {
        expected: DataType,
        actual: DataType,
    },
    #[error("expected {expected} elements, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("expected {expected} bytes, got {actual}")]
    ByteLengthMismatch { expected: usize, actual: usize },
    #[error("{0:?} storage is not a floating-point type")]
    NotFloatingPoint(DataType),
    #[error("zero-copy access requires contiguous Float32 storage")]
    NotContiguousFloat32,
    #[error("strides {0:?} overlap or do not match the element size")]
    InvalidStrides([usize; 4]),
    #[error("{available}-byte allocation does not cover {required} bytes")]
    AllocationTooSmall { required: usize, available: usize },
    #[error("surface is not aligned for {0:?} elements")]
    Misaligned(DataType),
    #[error("element {index} is out of range for {count} elements")]
    IndexOutOfRange { index: usize, count: usize },
    #[error("logical shape {actual:?} differs from compiled shape {expected:?}")]
    ShapeMismatch {
        expected: Box<[usize]>,
        actual: Box<[usize]>,
    },
    #[error(
        "{} requires {:?} {:?}, strides {:?}, at least {} bytes; supplied {:?} {:?}, strides {:?}, {allocation} bytes",
        .expected.name(), .expected.shape(), .expected.data_type(), .expected.strides(), .expected.allocation_size(),
        .actual.shape(), .actual.data_type(), .actual.strides()
    )]
    LayoutMismatch {
        expected: Box<TensorSpec>,
        actual: Box<TensorSpec>,
        allocation: usize,
    },
    #[error("tensor bytes are not a valid bit pattern for the element type")]
    InvalidBitPattern,
}
