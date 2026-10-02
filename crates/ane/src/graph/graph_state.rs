use crate::graph::{GraphError, checked_shape, ensure};
use crate::ir::{BuiltinOp, ConstantInput, Op, Operator, Parameter, Value, WeightBlob};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::graph::{Tensor, TensorHandle};
use crate::{DataType, VariableData};

pub struct GraphState {
    pub inputs: Vec<(Tensor, DataType)>,
    pub constants: HashMap<usize, (WeightBlob, [usize; 4])>,
    pub ops: Vec<(Op, Tensor)>,
    pub tensors: Vec<Tensor>,
    pub identity: u64,
    pub state_versions: HashMap<usize, Tensor>,
    pub variables: HashMap<Tensor, VariableData>,
}

impl Default for GraphState {
    fn default() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            inputs: Vec::new(),
            constants: HashMap::new(),
            ops: Vec::new(),
            tensors: Vec::new(),
            identity: NEXT
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("graph identity space exhausted"),
            state_versions: HashMap::new(),
            variables: HashMap::new(),
        }
    }
}

impl GraphState {
    pub fn builtin_many(
        &mut self,
        operation: Operator,
        inputs: &[(Parameter, Tensor)],
        attributes: &[(Parameter, Value)],
        shapes: &[(DataType, &[usize])],
        blobs: Box<[ConstantInput]>,
    ) -> Result<Vec<Tensor>, GraphError> {
        for (_, tensor) in inputs {
            self.check_tensor(*tensor)?;
        }
        ensure(
            !shapes.is_empty(),
            GraphError::InvalidArgument("operation requires an output"),
        )?;
        let specs = shapes
            .iter()
            .map(|(dtype, shape)| Ok((checked_shape(shape)?, shape.len(), *dtype)))
            .collect::<Result<Vec<_>, GraphError>>()?;
        let outputs: Vec<_> = specs
            .into_iter()
            .enumerate()
            .map(|(index, (shape, rank, dtype))| {
                Tensor::new(
                    self.tensors.len() + index,
                    self.identity,
                    shape,
                    rank,
                    dtype,
                )
            })
            .collect();
        let op = BuiltinOp {
            operation,
            logical: false,
            blobs,
            inputs: inputs.into(),
            attributes: attributes.into(),
            outputs: outputs.as_slice().into(),
        };
        op.validate()?;
        self.tensors.extend_from_slice(&outputs);
        self.ops.push((Op::Builtin(op), outputs[0]));
        Ok(outputs)
    }

    pub fn logical_builtin(
        &mut self,
        operation: Operator,
        inputs: &[(Parameter, Tensor)],
        attributes: &[(Parameter, Value)],
        shapes: &[(DataType, &[usize])],
    ) -> Result<Vec<Tensor>, GraphError> {
        let outputs = self.builtin_many(operation, inputs, attributes, shapes, Box::new([]))?;
        if let Some((Op::Builtin(op), _)) = self.ops.last_mut() {
            op.logical = true;
        }
        Ok(outputs)
    }

    pub fn numeric(&self, tensor: Tensor) -> Result<(), GraphError> {
        self.check_tensor(tensor)?;
        ensure(
            tensor.data_type() == DataType::Float16,
            GraphError::NotFloatingPoint,
        )
    }

    pub fn axis(&self, tensor: Tensor, axis: i64) -> Result<usize, GraphError> {
        self.check_tensor(tensor)?;
        let rank = tensor.rank() as i64;
        ensure(
            rank > 0 && (-rank..rank).contains(&axis),
            GraphError::InvalidAxis {
                axis,
                rank: tensor.rank(),
            },
        )?;
        Ok((4 - rank + axis.rem_euclid(rank)) as usize)
    }

    pub fn alloc_rank(&mut self, shape: [usize; 4], rank: usize) -> Tensor {
        self.alloc_typed(shape, rank, DataType::Float16)
    }

    pub fn alloc_typed(&mut self, shape: [usize; 4], rank: usize, dtype: DataType) -> Tensor {
        let tensor = Tensor::new(self.tensors.len(), self.identity, shape, rank, dtype);
        self.tensors.push(tensor);
        tensor
    }

    pub fn check_tensor(&self, tensor: Tensor) -> Result<(), GraphError> {
        ensure(
            tensor.graph_identity() == self.identity
                && self.tensors.get(tensor.id()) == Some(&tensor),
            GraphError::ForeignTensor,
        )
    }
}
