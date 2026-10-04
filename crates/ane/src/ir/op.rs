use std::collections::HashMap;

use crate::graph::{Tensor, TensorHandle};
use crate::ir::{BuiltinOp, ConstantOp, StateReadOp, StateUpdateOp, StateWriteOp};

#[derive(Clone, PartialEq)]
pub enum Op {
    Constant(ConstantOp),
    StateUpdate(StateUpdateOp),
    StateRead(StateReadOp),
    StateWrite(StateWriteOp),
    Builtin(BuiltinOp),
}

impl Op {
    pub fn tops(&self) -> Vec<Tensor> {
        match self {
            Self::Constant(operation) => vec![operation.top],
            Self::StateUpdate(operation) => vec![operation.top],
            Self::StateRead(operation) => vec![operation.top],
            Self::StateWrite(operation) => vec![operation.top],
            Self::Builtin(operation) => operation.outputs.to_vec(),
        }
    }

    pub fn substitute(&mut self, replacements: &HashMap<Tensor, Tensor>) {
        let swap = |tensor: &mut Tensor| {
            if let Some(&replacement) = replacements.get(tensor) {
                *tensor = replacement;
            }
        };
        match self {
            Self::Constant(_) => {}
            Self::StateUpdate(l) => [&mut l.bottom, &mut l.state, &mut l.position]
                .into_iter()
                .for_each(swap),
            Self::StateRead(l) => swap(&mut l.state),
            Self::StateWrite(l) => [&mut l.state, &mut l.bottom].into_iter().for_each(swap),
            Self::Builtin(l) => l.inputs.iter_mut().for_each(|(_, tensor)| swap(tensor)),
        }
    }

    pub fn same_value(&self, other: &Self) -> bool {
        let shaped = |a: Tensor, b: Tensor| {
            a.physical_shape() == b.physical_shape()
                && a.rank() == b.rank()
                && a.data_type() == b.data_type()
        };
        match (self, other) {
            (Self::Constant(a), Self::Constant(b)) => shaped(a.top, b.top) && a.data == b.data,
            (Self::Builtin(a), Self::Builtin(b)) => {
                a.inputs.is_empty()
                    && b.inputs.is_empty()
                    && a.outputs.len() == 1
                    && b.outputs.len() == 1
                    && shaped(a.outputs[0], b.outputs[0])
                    && a.operation == b.operation
                    && a.attributes == b.attributes
                    && a.blobs == b.blobs
            }
            _ => false,
        }
    }

    pub fn bottoms(&self) -> Vec<Tensor> {
        match self {
            Self::Constant(_) => vec![],
            Self::StateUpdate(l) => vec![l.bottom, l.state, l.position],
            Self::StateRead(l) => vec![l.state],
            Self::StateWrite(l) => vec![l.state, l.bottom],
            Self::Builtin(l) => l.inputs.iter().map(|&(_, tensor)| tensor).collect(),
        }
    }
}
