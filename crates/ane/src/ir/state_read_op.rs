use crate::graph::Tensor;

#[derive(Clone, PartialEq)]
pub struct StateReadOp {
    pub top: Tensor,
    pub state: Tensor,
}
