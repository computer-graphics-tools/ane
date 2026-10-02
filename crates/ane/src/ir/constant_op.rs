use crate::graph::Tensor;
use crate::ir::WeightBlob;

#[derive(Clone, PartialEq)]
pub struct ConstantOp {
    pub top: Tensor,
    pub data: WeightBlob,
}
