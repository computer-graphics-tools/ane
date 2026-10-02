use crate::graph::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Operation {
    tensor: Tensor,
}

pub fn state_operation(tensor: Tensor) -> Operation {
    Operation { tensor }
}

pub fn operation_tensor(operation: &Operation) -> Tensor {
    operation.tensor
}
