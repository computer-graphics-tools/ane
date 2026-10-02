use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::GraphError;
use crate::graph::Tensor;

impl Graph {
    pub fn top_k(
        &self,
        input: &Tensor,
        k: usize,
        axis: i64,
    ) -> Result<(Tensor, Tensor), GraphError> {
        self.ranked(*input, k, axis, false)
    }

    pub fn bottom_k(
        &self,
        input: &Tensor,
        k: usize,
        axis: i64,
    ) -> Result<(Tensor, Tensor), GraphError> {
        self.ranked(*input, k, axis, true)
    }
}
