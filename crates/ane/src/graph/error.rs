use crate::{IrError, TensorDataError, padded_shape};

#[derive(Clone, Debug, thiserror::Error)]
pub enum GraphError {
    #[error(transparent)]
    TensorData(#[from] TensorDataError),
    #[error(transparent)]
    Ir(#[from] IrError),
    #[error("shape {0:?} needs at most four positive Int32 dimensions within the address space")]
    InvalidShape(Box<[usize]>),
    #[error("tensor size overflows the address space")]
    Overflow,
    #[error("tensor belongs to another graph or has invalid metadata")]
    ForeignTensor,
    #[error("operation requires a floating-point tensor")]
    NotFloatingPoint,
    #[error("axis {axis} is outside a rank-{rank} tensor")]
    InvalidAxis { axis: i64, rank: usize },
    #[error("invalid axes: {0}")]
    InvalidAxes(&'static str),
    #[error("unsupported data type: {0}")]
    UnsupportedDataType(&'static str),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(&'static str),
    #[error("out of bounds: {0}")]
    OutOfBounds(&'static str),
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),
    #[error("non-constant weights: {0}")]
    NonConstantWeights(&'static str),
    #[error("invalid compilation targets: {0}")]
    InvalidTargets(&'static str),
    #[error("unsupported ANE compiler composition: {0}")]
    UnsupportedComposition(&'static str),
}

pub fn ensure(condition: bool, error: GraphError) -> Result<(), GraphError> {
    if condition { Ok(()) } else { Err(error) }
}

pub fn checked_shape(shape: &[usize]) -> Result<[usize; 4], GraphError> {
    padded_shape(shape).ok_or_else(|| GraphError::InvalidShape(shape.into()))
}
