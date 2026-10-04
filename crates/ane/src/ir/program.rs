use crate::graph::TensorHandle;
use crate::ir::{IrError, MilProgram, Op, validate_program};
use crate::{DataType, Tensor};

#[derive(Clone, PartialEq)]
pub struct Program {
    name: String,
    ops: Box<[Op]>,
    feed_tensors: Box<[Tensor]>,
    target_tensors: Box<[Tensor]>,
    inputs: Box<[(String, [usize; 4], DataType)]>,
    outputs: Box<[(String, [usize; 4], DataType)]>,
}

impl Program {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }
    pub fn feed_tensors(&self) -> &[Tensor] {
        &self.feed_tensors
    }
    pub fn target_tensors(&self) -> &[Tensor] {
        &self.target_tensors
    }
    pub fn inputs(&self) -> &[(String, [usize; 4], DataType)] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[(String, [usize; 4], DataType)] {
        &self.outputs
    }
    pub fn new(
        name: &str,
        ops: &[Op],
        inputs: &[(Tensor, DataType)],
        outputs: &[Tensor],
        output_types: &[DataType],
    ) -> Result<Self, IrError> {
        let mut characters = name.chars();
        if !characters
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            || !characters.all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(IrError::InvalidProgram(
                "MIL function names are ASCII identifiers",
            ));
        }
        validate_program(ops, inputs, outputs, output_types)?;
        Ok(Self {
            name: name.into(),
            ops: ops.into(),
            feed_tensors: inputs.iter().map(|(t, _)| *t).collect(),
            target_tensors: outputs.into(),
            inputs: inputs
                .iter()
                .map(|(t, d)| (t.symbol(), t.physical_shape(), *d))
                .collect(),
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    (
                        t.symbol(),
                        t.physical_shape(),
                        output_types[i.min(output_types.len() - 1)],
                    )
                })
                .collect(),
        })
    }
    pub fn constant_bytes(&self) -> usize {
        self.ops
            .iter()
            .map(|op| match op {
                Op::Constant(op) => op.data.bytes().len(),
                Op::Builtin(op) => op.blobs.iter().map(|b| b.data.bytes().len()).sum(),
                _ => 0,
            })
            .sum()
    }
    pub fn operations(&self) -> Vec<String> {
        self.ops
            .iter()
            .map(|op| {
                match op {
                    Op::Constant(_) => "constant",
                    Op::StateRead(_) => "read_state",
                    Op::StateWrite(_) => "write_state",
                    Op::StateUpdate(_) => "write_state_rows",
                    Op::Builtin(op) => op.operation.as_str(),
                }
                .to_string()
            })
            .collect()
    }
    pub fn mil(&self) -> Result<MilProgram, IrError> {
        MilProgram::new(std::slice::from_ref(self))
    }
}
