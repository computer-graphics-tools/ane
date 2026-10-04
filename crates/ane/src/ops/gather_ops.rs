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

    /// Slices of `input` at UInt16 `indices` along `axis`. The ANE runs it on the innermost axis only.
    /// MIL `gather`.
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
        self.builtin(
            Operator::Gather,
            &[(Parameter::X, *input), (Parameter::Indices, indices)],
            &[
                (Parameter::Axis, Value::Int32(axis + 4 - input.rank())),
                (Parameter::BatchDims, Value::Int32(0)),
                (Parameter::ValidateIndices, Value::Bool(false)),
            ],
            &shape,
            input.data_type(),
        )
    }

    /// Elements of `input` at UInt16 `indices` along `axis`; `indices` matches the input shape except
    /// along `axis`. MIL `gather_along_axis`.
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
}
