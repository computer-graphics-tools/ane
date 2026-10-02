use crate::WeightDataType;
use crate::ir::CompilerException;

#[derive(Clone, Debug, thiserror::Error)]
pub enum IrError {
    #[error("Apple graph compiler rejected the graph: {0}")]
    Native(#[from] CompilerException),
    #[error("Apple graph compiler failed without an exception")]
    CompilerFailed,
    #[error("cannot lower graph operation: {0}")]
    Lowering(&'static str),
    #[error("invalid {operation} argument {parameter}: {reason}")]
    Argument {
        operation: &'static str,
        parameter: &'static str,
        reason: &'static str,
    },
    #[error("invalid graph program: {0}")]
    InvalidProgram(&'static str),
    #[error("graph value cannot be represented as {0}")]
    InvalidValue(&'static str),
    #[error("{elements} {data_type:?} weights overflow the address space")]
    WeightSizeOverflow {
        elements: usize,
        data_type: WeightDataType,
    },
    #[error("{elements} {data_type:?} weights need {expected} bytes, got {actual}")]
    WeightLength {
        elements: usize,
        data_type: WeightDataType,
        expected: usize,
        actual: usize,
    },
}
