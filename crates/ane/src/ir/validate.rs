use crate::DataType;
use crate::graph::{Tensor, TensorHandle};
use crate::ir::{IrError, Op};
use std::collections::HashSet;

pub fn validate_program(
    ops: &[Op],
    inputs: &[(Tensor, DataType)],
    outputs: &[Tensor],
    output_types: &[DataType],
) -> Result<(), IrError> {
    if outputs.is_empty() || !(output_types.len() == 1 || output_types.len() == outputs.len()) {
        return Err(IrError::InvalidProgram(
            "output storage types do not match targets",
        ));
    }
    let identity = outputs[0].graph_identity();
    let mut defined = HashSet::new();
    for &(tensor, _) in inputs {
        if tensor.graph_identity() != identity || !defined.insert(tensor) {
            return Err(IrError::InvalidProgram("duplicate or foreign input"));
        }
    }
    for op in ops {
        if let Op::Builtin(op) = op {
            op.validate()?;
        }
        if let Op::Constant(op) = op
            && op.top.physical_shape().iter().product::<usize>() != op.data.element_count()
        {
            return Err(IrError::InvalidProgram(
                "constant shape differs from its data",
            ));
        }
        for tensor in op.bottoms() {
            if !defined.contains(&tensor) {
                return Err(IrError::InvalidProgram(
                    "operand used before its definition",
                ));
            }
        }
        for tensor in op.tops() {
            if tensor.graph_identity() != identity || !defined.insert(tensor) {
                return Err(IrError::InvalidProgram("duplicate or foreign result"));
            }
        }
    }
    let mut targets = HashSet::new();
    for tensor in outputs {
        if !defined.contains(tensor) || !targets.insert(tensor) {
            return Err(IrError::InvalidProgram("undefined or duplicate output"));
        }
    }
    Ok(())
}
