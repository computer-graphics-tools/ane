use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, checked_shape, ensure};
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    fn indices(&self, indices: Tensor) -> Result<(), GraphError> {
        self.check_tensor(indices)?;
        ensure(
            indices.data_type() == DataType::UInt16,
            GraphError::UnsupportedDataType("ANE indices require UInt16 tensors"),
        )
    }

    fn gather_operands(&self, source: Tensor, indices: Tensor) -> Result<(), GraphError> {
        self.check_tensor(source)?;
        ensure(
            source.data_type() != DataType::Bool,
            GraphError::UnsupportedDataType(
                "ANE cannot gather Boolean data; cast it to UInt8 first",
            ),
        )?;
        self.indices(indices)
    }

    pub fn gather(
        &self,
        input: &Tensor,
        indices: &Tensor,
        axis: i64,
    ) -> Result<Tensor, GraphError> {
        let indices = *indices;
        self.gather_operands(*input, indices)?;
        let axis = self.axis(*input, axis)? - (4 - input.rank());
        let mut shape = input.shape()[..axis].to_vec();
        shape.extend(indices.shape());
        shape.extend(&input.shape()[axis + 1..]);
        checked_shape(&shape)?;
        let count = indices.shape().iter().product();
        let indices = self.reshape_to(indices, &[count])?;
        let mut flat_shape = input.shape().to_vec();
        flat_shape[axis] = count;
        let output = self.logical_builtin(
            Operator::Gather,
            &[(Parameter::X, *input), (Parameter::Indices, indices)],
            &[
                (Parameter::Axis, Value::Int32(axis + 4 - input.rank())),
                (Parameter::BatchDims, Value::Int32(0)),
                (Parameter::ValidateIndices, Value::Bool(false)),
            ],
            &[(input.data_type(), &flat_shape)],
        )?[0];
        self.reshape_to(output, &shape)
    }

    pub fn one_hot(&self, indices: &Tensor, depth: usize) -> Result<Tensor, GraphError> {
        self.indices(*indices)?;
        ensure(
            indices.rank() < 4 && (1..=2048).contains(&depth),
            GraphError::InvalidArgument(
                "one-hot requires indices below rank 4 and a depth of at most 2048",
            ),
        )?;
        let values = self.cast(indices, DataType::Float16)?;
        let mut shape = indices.shape().to_vec();
        shape.push(1);
        let values = self.reshape_to(values, &shape)?;
        let classes: Vec<f32> = (0..depth).map(|class| class as f32).collect();
        let classes = self.constant(&classes, [depth])?;
        let hot = self.equal(&values, &classes)?;
        self.cast(&hot, DataType::Float16)
    }

    pub fn gather_along_axis(
        &self,
        input: &Tensor,
        indices: &Tensor,
        axis: i64,
    ) -> Result<Tensor, GraphError> {
        self.gather_operands(*input, *indices)?;
        let physical = self.axis(*input, axis)?;
        ensure(
            input.rank() == indices.rank()
                && (0..4).all(|a| {
                    a == physical || input.physical_shape()[a] == indices.physical_shape()[a]
                }),
            GraphError::ShapeMismatch("gather-along-axis shapes differ"),
        )?;
        let axis = physical;
        self.builtin(
            Operator::GatherAlongAxis,
            &[(Parameter::X, *input), (Parameter::Indices, *indices)],
            &[
                (Parameter::Axis, Value::Int32(axis)),
                (Parameter::ValidateIndices, Value::Bool(false)),
            ],
            indices.shape(),
            input.data_type(),
        )
    }

    pub fn gather_nd(&self, input: &Tensor, indices: &Tensor) -> Result<Tensor, GraphError> {
        self.gather_operands(*input, *indices)?;
        ensure(
            indices.rank() > 0 && indices.physical_shape()[3] <= input.rank(),
            GraphError::OutOfBounds("gather-ND index depth exceeds rank"),
        )?;
        let depth = indices.physical_shape()[3];
        let mut shape = indices.shape()[..indices.rank() - 1].to_vec();
        shape.extend(&input.shape()[depth..]);
        checked_shape(&shape)?;
        Ok(self.logical_builtin(
            Operator::GatherNd,
            &[(Parameter::X, *input), (Parameter::Indices, *indices)],
            &[
                (Parameter::BatchDims, Value::Int32(0)),
                (Parameter::ValidateIndices, Value::Bool(false)),
            ],
            &[(input.data_type(), &shape)],
        )?[0])
    }
}
