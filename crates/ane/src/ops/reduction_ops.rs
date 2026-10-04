use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    fn reduce(
        &self,
        input: Tensor,
        operation: Operator,
        axes: &[i64],
    ) -> Result<Tensor, GraphError> {
        self.numeric(input)?;
        ensure(
            !axes.is_empty(),
            GraphError::InvalidAxes("reduction requires at least one axis"),
        )?;
        let mut physical = Vec::with_capacity(axes.len());
        for &axis in axes {
            let axis = self.axis(input, axis)?;
            ensure(
                !physical.contains(&axis),
                GraphError::InvalidAxes("duplicate reduction axis"),
            )?;
            physical.push(axis);
        }
        let mut shape = input.physical_shape();
        for &axis in &physical {
            shape[axis] = 1;
        }
        self.builtin(
            operation,
            &[(Parameter::X, input)],
            &[
                (Parameter::Axes, Value::int32_list(&physical)),
                (Parameter::KeepDims, Value::Bool(true)),
            ],
            &shape[4 - input.rank()..],
            DataType::Float16,
        )
    }

    /// Sum over `axes`, kept as size 1. MIL `reduce_sum`.
    pub fn reduction_sum(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceSum, axes)
    }

    /// Mean over `axes`, kept as size 1. MIL `reduce_mean`.
    pub fn reduction_mean(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceMean, axes)
    }

    /// Minimum over `axes`, kept as size 1. MIL `reduce_min`.
    pub fn reduction_minimum(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceMin, axes)
    }

    /// Maximum over `axes`, kept as size 1. MIL `reduce_max`.
    pub fn reduction_maximum(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceMax, axes)
    }

    /// `Σ|x|` over `axes`, kept as size 1. MIL `reduce_l1_norm`.
    pub fn reduction_l1_norm(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceL1Norm, axes)
    }

    /// `sqrt(Σx²)` over `axes`, kept as size 1. MIL `reduce_l2_norm`.
    pub fn reduction_l2_norm(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceL2Norm, axes)
    }

    /// `ln(Σx)` over `axes`, kept as size 1. MIL `reduce_log_sum`.
    pub fn reduction_log_sum(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceLogSum, axes)
    }

    /// `ln(Σe^x)` over `axes`, kept as size 1. MIL `reduce_log_sum_exp`.
    pub fn reduction_log_sum_exp(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceLogSumExp, axes)
    }

    /// `Σx²` over `axes`, kept as size 1. MIL `reduce_sum_square`.
    pub fn reduction_sum_square(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceSumSquare, axes)
    }

    /// UInt16 index of the maximum along `axis`, kept as size 1; the axis has at most 2048 elements.
    /// MIL `reduce_argmax`.
    pub fn reduction_arg_maximum(&self, x: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        self.reduce_arg(*x, axis, Operator::ReduceArgmax)
    }

    /// UInt16 index of the minimum along `axis`, kept as size 1; the axis has at most 2048 elements.
    /// MIL `reduce_argmin`.
    pub fn reduction_arg_minimum(&self, x: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        self.reduce_arg(*x, axis, Operator::ReduceArgmin)
    }

    fn reduce_arg(&self, x: Tensor, axis: i64, operation: Operator) -> Result<Tensor, GraphError> {
        self.numeric(x)?;
        let axis = self.axis(x, axis)?;
        ensure(
            x.physical_shape()[axis] <= 2048,
            GraphError::OutOfBounds("ANE index results are exact only for axes up to 2048"),
        )?;
        let mut shape = x.physical_shape();
        shape[axis] = 1;
        self.builtin(
            operation,
            &[(Parameter::X, x)],
            &[
                (Parameter::Axis, Value::Int32(axis)),
                (Parameter::KeepDims, Value::Bool(true)),
                (Parameter::OutputDtype, Value::String("uint16")),
            ],
            &shape[4 - x.rank()..],
            DataType::UInt16,
        )
    }
}
