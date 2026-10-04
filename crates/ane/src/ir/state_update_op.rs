use crate::graph::Tensor;

#[derive(Clone, PartialEq)]
pub struct StateUpdateOp {
    pub top: Tensor,
    pub bottom: Tensor,
    pub state: Tensor,
    pub position: Tensor,
    pub rows: usize,
    pub channel: usize,
    pub channels: usize,
}
