use crate::DataType;
use crate::graph::Tensor;
use crate::ir::Op;

pub struct LoweredGraph {
    pub ops: Vec<Op>,
    pub inputs: Vec<(Tensor, DataType)>,
}
