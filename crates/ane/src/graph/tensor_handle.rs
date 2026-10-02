use crate::DataType;

pub trait TensorHandle {
    fn new(id: usize, graph: u64, shape: [usize; 4], rank: usize, dtype: DataType) -> Self;
    fn id(&self) -> usize;
    fn graph_identity(&self) -> u64;
    fn physical_shape(&self) -> [usize; 4];
    fn rank(&self) -> usize;
}
