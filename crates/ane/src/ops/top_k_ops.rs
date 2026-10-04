use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    /// The `k` largest values along `axis` in descending order and their UInt16 indices; the axis has at
    /// most 2048 elements. MIL `topk`.
    pub fn top_k(&self, x: &Tensor, k: usize, axis: i64) -> Result<(Tensor, Tensor), GraphError> {
        self.topk(*x, k, axis, false)
    }

    /// The `k` smallest values along `axis` in ascending order and their UInt16 indices; the axis has at
    /// most 2048 elements. MIL `topk` with `ascending`.
    pub fn bottom_k(
        &self,
        x: &Tensor,
        k: usize,
        axis: i64,
    ) -> Result<(Tensor, Tensor), GraphError> {
        self.topk(*x, k, axis, true)
    }

    fn topk(
        &self,
        x: Tensor,
        k: usize,
        axis: i64,
        ascending: bool,
    ) -> Result<(Tensor, Tensor), GraphError> {
        self.numeric(x)?;
        let axis = self.axis(x, axis)?;
        ensure(
            k > 0 && k <= x.physical_shape()[axis] && x.physical_shape()[axis] <= 2048,
            GraphError::OutOfBounds(
                "topk requires 0 < k <= axis length; ANE indices are exact only for axes up to 2048",
            ),
        )?;
        let mut shape = x.physical_shape();
        shape[axis] = k;
        let logical = &shape[4 - x.rank()..];
        let outputs = self.builtin_many(
            Operator::Topk,
            &[(Parameter::X, x)],
            &[
                (Parameter::K, Value::Int32(k)),
                (Parameter::Axis, Value::Int32(axis)),
                (Parameter::Ascending, Value::Bool(ascending)),
                (Parameter::Sort, Value::Bool(true)),
                (Parameter::ReturnIndices, Value::Bool(true)),
                (Parameter::OutputIndicesDtype, Value::String("uint16")),
            ],
            &[(DataType::Float16, logical), (DataType::UInt16, logical)],
        )?;
        Ok((outputs[0], outputs[1]))
    }
}
