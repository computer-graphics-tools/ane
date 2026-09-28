#[path = "graph.rs"]
mod graph;
pub use graph::{Graph, MIN_SPATIAL_WIDTH, State};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    id: usize,
    pub shape: [usize; 4],
}
