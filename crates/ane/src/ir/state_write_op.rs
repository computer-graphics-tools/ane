use crate::graph::Tensor;

#[derive(Clone, PartialEq)]
pub struct StateWriteOp {
    pub top: Tensor,
    pub state: Tensor,
    pub bottom: Tensor,
}
