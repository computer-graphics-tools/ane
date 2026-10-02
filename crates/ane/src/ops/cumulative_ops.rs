use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::GraphError;
use crate::graph::Tensor;
use crate::graph::TensorHandle;

impl Graph {
    fn scan(
        &self,
        input: Tensor,
        axis: usize,
        options: (bool, bool),
        combine: fn(&Graph, &Tensor, &Tensor) -> Result<Tensor, GraphError>,
        identity: f32,
    ) -> Result<Tensor, GraphError> {
        let (exclusive, reverse) = options;
        self.numeric(input)?;
        self.axis(input, axis as i64)?;
        let length = input.shape()[axis];
        let mut result = if reverse {
            self.reverse(&input, &[axis as i64])?
        } else {
            input
        };
        let mut offset = 1;
        while offset < length {
            let mut begin = vec![0; input.rank()];
            let mut shape = input.shape().to_vec();
            shape[axis] = offset;
            let prefix = self.slice(&result, &begin, &shape)?;
            shape[axis] = length - offset;
            let previous = self.slice(&result, &begin, &shape)?;
            begin[axis] = offset;
            let current = self.slice(&result, &begin, &shape)?;
            let combined = combine(self, &current, &previous)?;
            result = self.concat(&[&prefix, &combined], axis)?;
            offset = offset.saturating_mul(2);
        }
        if exclusive {
            let mut shape = input.shape().to_vec();
            shape[axis] = 1;
            let first = self.constant_scalar(identity, &shape)?;
            result = if length == 1 {
                first
            } else {
                shape[axis] = length - 1;
                let tail = self.slice(&result, vec![0; input.rank()], &shape)?;
                self.concat(&[&first, &tail], axis)?
            };
        }
        if reverse {
            self.reverse(&result, &[axis as i64])
        } else {
            Ok(result)
        }
    }

    pub fn cumulative_sum(
        &self,
        input: &Tensor,
        axis: usize,
        exclusive: bool,
        reverse: bool,
    ) -> Result<Tensor, GraphError> {
        self.scan(*input, axis, (exclusive, reverse), Self::addition, 0.0)
    }

    pub fn cumulative_product(
        &self,
        input: &Tensor,
        axis: usize,
        exclusive: bool,
        reverse: bool,
    ) -> Result<Tensor, GraphError> {
        self.scan(
            *input,
            axis,
            (exclusive, reverse),
            Self::multiplication,
            1.0,
        )
    }

    pub fn cumulative_min(
        &self,
        input: &Tensor,
        axis: usize,
        exclusive: bool,
        reverse: bool,
    ) -> Result<Tensor, GraphError> {
        self.scan(
            *input,
            axis,
            (exclusive, reverse),
            Self::minimum,
            f32::INFINITY,
        )
    }

    pub fn cumulative_max(
        &self,
        input: &Tensor,
        axis: usize,
        exclusive: bool,
        reverse: bool,
    ) -> Result<Tensor, GraphError> {
        self.scan(
            *input,
            axis,
            (exclusive, reverse),
            Self::maximum,
            f32::NEG_INFINITY,
        )
    }
}
