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

    pub fn reduction_sum(&self, input: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        self.reduce(*input, Operator::ReduceSum, &[axis])
    }

    pub fn reduction_mean(&self, input: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        self.reduce(*input, Operator::ReduceMean, &[axis])
    }

    pub fn reduction_minimum(&self, input: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        self.reduce(*input, Operator::ReduceMin, &[axis])
    }

    pub fn reduction_maximum(&self, input: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        self.reduce(*input, Operator::ReduceMax, &[axis])
    }

    pub fn sum(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceSum, axes)
    }

    pub fn mean(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        self.reduce(*x, Operator::ReduceMean, axes)
    }

    pub fn variance(&self, x: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        let mean = self.mean(x, axes)?;
        let centered = self.subtraction(x, &mean)?;
        let squared = self.square(&centered)?;
        self.mean(&squared, axes)
    }

    pub fn reduction_sum_square(&self, x: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        let squared = self.square(x)?;
        self.reduction_sum(&squared, axis)
    }

    pub fn reduction_l1_norm(&self, x: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        let absolute = self.absolute(x)?;
        self.reduction_sum(&absolute, axis)
    }

    pub fn reduction_l2_norm(&self, x: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        let squared = self.reduction_sum_square(x, axis)?;
        self.square_root(&squared)
    }

    pub fn reduction_log_sum(&self, x: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        let sum = self.reduction_sum(x, axis)?;
        self.logarithm(&sum)
    }

    pub fn reduction_log_sum_exp(&self, x: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        let max = self.reduction_maximum(x, axis)?;
        let centered = self.subtraction(x, &max)?;
        let exp = self.exponent(&centered)?;
        let sum = self.reduction_sum(&exp, axis)?;
        let log = self.logarithm(&sum)?;
        self.addition(&log, &max)
    }

    pub fn reduction_arg_maximum(&self, input: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        Ok(self.top_k(input, 1, axis)?.1)
    }

    pub fn reduction_arg_minimum(&self, input: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        Ok(self.bottom_k(input, 1, axis)?.1)
    }

    pub fn reduction_product(&self, input: &Tensor, axis: usize) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        self.axis(*input, axis as i64)?;
        let mut current = *input;
        while current.shape()[axis] > 1 {
            let length = current.shape()[axis];
            let pairs = length / 2;
            let mut chunks = Vec::new();
            for i in 0..pairs {
                let mut begin = vec![0; input.rank()];
                let mut shape = current.shape().to_vec();
                shape[axis] = 1;
                begin[axis] = 2 * i;
                let a = self.slice(&current, &begin, &shape)?;
                begin[axis] += 1;
                let b = self.slice(&current, &begin, &shape)?;
                chunks.push(self.multiplication(&a, &b)?);
            }
            if !length.is_multiple_of(2) {
                let mut begin = vec![0; input.rank()];
                let mut shape = current.shape().to_vec();
                shape[axis] = 1;
                begin[axis] = length - 1;
                chunks.push(self.slice(&current, &begin, &shape)?);
            }
            current = self.concat(&chunks.iter().collect::<Vec<_>>(), axis)?;
        }
        Ok(current)
    }
}
