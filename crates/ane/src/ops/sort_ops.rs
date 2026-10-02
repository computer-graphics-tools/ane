use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::GraphError;
use crate::graph::Tensor;
use crate::graph::TensorHandle;

impl Graph {
    pub fn sort(&self, input: &Tensor, axis: i64, ascending: bool) -> Result<Tensor, GraphError> {
        let length = input.physical_shape()[self.axis(*input, axis)?];
        Ok(self.ranked(*input, length, axis, ascending)?.0)
    }

    pub fn argsort(
        &self,
        input: &Tensor,
        axis: i64,
        ascending: bool,
    ) -> Result<Tensor, GraphError> {
        let length = input.physical_shape()[self.axis(*input, axis)?];
        Ok(self.ranked(*input, length, axis, ascending)?.1)
    }
}
