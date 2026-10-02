use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, checked_shape, ensure};
use crate::graph::{Operation, state_operation};
use crate::ir::{Op, StateReadOp, StateUpdateOp, StateWriteOp, WeightBlob};
use crate::{TensorData, VariableData, WeightDataType, logical_shape};

impl Graph {
    pub fn placeholder<const RANK: usize>(
        &self,
        shape: [usize; RANK],
        data_type: DataType,
    ) -> Result<Tensor, GraphError> {
        self.input_placeholder(logical_shape(&shape), data_type)
    }

    pub fn variable_with_data<const RANK: usize>(
        &self,
        data: &[f32],
        shape: [usize; RANK],
    ) -> Result<Tensor, GraphError> {
        ensure(
            data.len() == checked_shape(&shape)?.iter().product::<usize>(),
            GraphError::ShapeMismatch("variable data length differs from shape"),
        )?;
        let variable = self.variable_placeholder(shape)?;
        self.state()
            .variables
            .insert(variable, VariableData::Values(data.into()));
        Ok(variable)
    }

    pub fn variable_with_tensor_data(&self, data: &TensorData) -> Result<Tensor, GraphError> {
        ensure(
            matches!(
                data.data_type(),
                DataType::Float16 | DataType::Int8 | DataType::UInt8
            ),
            GraphError::UnsupportedDataType("ANE variables require Float16, Int8 or UInt8 storage"),
        )?;
        let variable = self.input_placeholder(data.shape(), data.data_type())?;
        self.state()
            .variables
            .insert(variable, VariableData::Surface(data.clone()));
        Ok(variable)
    }

    pub fn variable_placeholder<const RANK: usize>(
        &self,
        shape: [usize; RANK],
    ) -> Result<Tensor, GraphError> {
        self.placeholder(shape, DataType::Float16)
    }

    pub fn read_variable(&self, variable: &Tensor) -> Result<Tensor, GraphError> {
        self.ensure_variable(*variable)?;
        if let Some(value) = self.state().state_versions.get(&variable.id()) {
            return Ok(*value);
        }
        let result = self.state().alloc_typed(
            variable.physical_shape(),
            variable.rank(),
            variable.data_type(),
        );
        self.state().ops.push((
            Op::StateRead(StateReadOp {
                top: result,
                state: *variable,
            }),
            result,
        ));
        Ok(result)
    }

    pub fn assign_variable(
        &self,
        variable: &Tensor,
        value: &Tensor,
    ) -> Result<Operation, GraphError> {
        self.ensure_variable(*variable)?;
        self.check_tensor(*value)?;
        ensure(
            value.data_type() == variable.data_type(),
            GraphError::UnsupportedDataType("state assignment storage type differs"),
        )?;
        ensure(
            value.physical_shape() == variable.physical_shape(),
            GraphError::ShapeMismatch("state assignment shape differs"),
        )?;
        let result = self.state().alloc_typed(
            variable.physical_shape(),
            variable.rank(),
            variable.data_type(),
        );
        let mut state = self.state();
        let previous = state.state_versions.get(&variable.id()).copied();
        state.ops.push((
            Op::StateWrite(StateWriteOp {
                top: result,
                state: *variable,
                previous,
                bottom: *value,
            }),
            result,
        ));
        state.state_versions.insert(variable.id(), result);
        Ok(state_operation(result))
    }

    pub fn assign_variable_rows(
        &self,
        variable: &Tensor,
        update: &Tensor,
        position: &Tensor,
        channel: usize,
    ) -> Result<Operation, GraphError> {
        self.ensure_variable(*variable)?;
        self.check_tensor(*update)?;
        ensure(
            update.data_type() == variable.data_type(),
            GraphError::UnsupportedDataType("state update storage type differs"),
        )?;
        self.check_tensor(*position)?;
        let shape = variable.physical_shape();
        ensure(
            self.state().inputs.iter().any(|(t, d)| {
                t.id() == position.id()
                    && *d == DataType::Int32
                    && t.physical_shape().iter().product::<usize>() == 1
            }),
            GraphError::UnsupportedDataType("state position must be an integer parameter"),
        )?;
        let update_shape = update.physical_shape();
        ensure(
            (update_shape[0], update_shape[3]) == (shape[0], shape[3]),
            GraphError::ShapeMismatch("state update dimensions differ"),
        )?;
        ensure(
            update_shape[2] > 0 && update_shape[2] <= shape[2],
            GraphError::OutOfBounds("state update exceeds height"),
        )?;
        ensure(
            channel
                .checked_add(update_shape[1])
                .is_some_and(|end| end <= shape[1]),
            GraphError::OutOfBounds("state update exceeds channels"),
        )?;
        let result = self
            .state()
            .alloc_typed(shape, variable.rank(), variable.data_type());
        let mut state = self.state();
        let previous = state.state_versions.get(&variable.id()).copied();
        state.ops.push((
            Op::StateUpdate(StateUpdateOp {
                top: result,
                bottom: *update,
                state: *variable,
                previous,
                position: *position,
                rows: update_shape[2],
                channel,
                channels: update_shape[1],
            }),
            result,
        ));
        state.state_versions.insert(variable.id(), result);
        Ok(state_operation(result))
    }

    fn ensure_variable(&self, variable: Tensor) -> Result<(), GraphError> {
        self.check_tensor(variable)?;
        ensure(
            self.state().inputs.iter().any(|(tensor, dtype)| {
                *tensor == variable
                    && matches!(dtype, DataType::Float16 | DataType::Int8 | DataType::UInt8)
            }),
            GraphError::InvalidArgument("tensor is not a variable"),
        )
    }

    pub fn constant<const RANK: usize>(
        &self,
        data: &[f32],
        shape: [usize; RANK],
    ) -> Result<Tensor, GraphError> {
        self.constant_values(data, logical_shape(&shape))
    }

    pub fn constant_with_bytes<const RANK: usize>(
        &self,
        data: &[u8],
        shape: [usize; RANK],
        dtype: DataType,
    ) -> Result<Tensor, GraphError> {
        let weight_type = match dtype {
            DataType::Float16 => WeightDataType::Float16,
            DataType::Int8 => WeightDataType::Int8,
            DataType::UInt8 => WeightDataType::UInt8,
            _ => {
                return Err(GraphError::UnsupportedDataType(
                    "byte constants require Float16, Int8 or UInt8",
                ));
            }
        };
        let physical = checked_shape(logical_shape(&shape))?;
        let blob = WeightBlob::from_bytes(data, physical.iter().product(), weight_type)?;
        let mut state = self.state();
        let tensor = state.alloc_typed(physical, RANK, dtype);
        state.constants.insert(tensor.id(), (blob, physical));
        Ok(tensor)
    }

    pub fn constant_with_scalar<const RANK: usize>(
        &self,
        scalar: f32,
        shape: [usize; RANK],
    ) -> Result<Tensor, GraphError> {
        self.constant_scalar(scalar, logical_shape(&shape))
    }

    pub fn boolean_constant(&self, value: bool) -> Result<Tensor, GraphError> {
        let value = self.constant_scalar(if value { 1.0 } else { 0.0 }, &[])?;
        self.cast(&value, DataType::Bool)
    }

    pub fn fill_like(&self, input: &Tensor, value: f32) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        self.constant_scalar(value, input.shape())
    }

    pub fn range(&self, start: f32, step: f32, count: usize) -> Result<Tensor, GraphError> {
        ensure(
            start.is_finite() && step.is_finite() && step != 0.0 && count > 0,
            GraphError::InvalidArgument("invalid constant range"),
        )?;
        checked_shape(&[count])?;
        let values: Vec<_> = (0..count).map(|i| start + step * i as f32).collect();
        ensure(values.iter().all(|v| v.is_finite()), GraphError::Overflow)?;
        self.constant(&values, [count])
    }

    pub fn coordinate_along_axis<const RANK: usize>(
        &self,
        shape: [usize; RANK],
        axis: usize,
    ) -> Result<Tensor, GraphError> {
        checked_shape(logical_shape(&shape))?;
        ensure(
            axis < RANK,
            GraphError::InvalidAxes("coordinate axis exceeds rank"),
        )?;
        let range = self.range(0.0, 1.0, shape[axis])?;
        let mut expanded = [1; RANK];
        expanded[axis] = shape[axis];
        let range = self.reshape(&range, expanded)?;
        self.broadcast_to(&range, shape)
    }
}
