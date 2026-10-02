use crate::DataType;
use crate::graph::TensorHandle;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    id: usize,
    graph: u64,
    shape: [usize; 4],
    rank: usize,
    dtype: DataType,
}

impl Tensor {
    pub fn symbol(&self) -> String {
        format!("t{}", self.id)
    }
    pub fn data_type(&self) -> DataType {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape[4 - self.rank..]
    }
}

impl TensorHandle for Tensor {
    fn new(id: usize, graph: u64, shape: [usize; 4], rank: usize, dtype: DataType) -> Self {
        Self {
            id,
            graph,
            shape,
            rank,
            dtype,
        }
    }
    fn id(&self) -> usize {
        self.id
    }
    fn graph_identity(&self) -> u64 {
        self.graph
    }
    fn physical_shape(&self) -> [usize; 4] {
        self.shape
    }
    fn rank(&self) -> usize {
        self.rank
    }
}
