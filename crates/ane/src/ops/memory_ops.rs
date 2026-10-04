use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, checked_shape, ensure};
use crate::graph::{Operation, state_operation};
use crate::ir::{Op, StateReadOp, StateUpdateOp, StateWriteOp, WeightBlob};
use crate::{StateData, TensorData, WeightDataType, logical_shape};

impl Graph {
    /// A runtime input with up to four axes. Float32 and Float16 inputs compute in Float16; Int32 is
    /// only accepted as a scalar position.
    pub fn placeholder<const RANK: usize>(
        &self,
        shape: [usize; RANK],
        data_type: DataType,
    ) -> Result<Tensor, GraphError> {
        self.input_placeholder(logical_shape(&shape), data_type)
    }

    /// A Float16 variable initialised from `data`. Its storage persists across runs and is shared by
    /// executables compiled together.
    pub fn variable_with_data<const RANK: usize>(
        &self,
        data: &[f32],
        shape: [usize; RANK],
    ) -> Result<Tensor, GraphError> {
        ensure(
            data.len() == checked_shape(&shape)?.iter().product::<usize>(),
            GraphError::ShapeMismatch("input data length differs from shape"),
        )?;
        let input = self.variable_placeholder(shape)?;
        self.state()
            .states
            .insert(input, StateData::Values(data.into()));
        Ok(input)
    }

    /// A variable stored in the caller's IOSurface-backed `data` (Float16, Int8 or UInt8), which the
    /// CPU can read and rewrite between runs.
    pub fn variable_with_tensor_data(&self, data: &TensorData) -> Result<Tensor, GraphError> {
        ensure(
            matches!(
                data.data_type(),
                DataType::Float16 | DataType::Int8 | DataType::UInt8
            ),
            GraphError::UnsupportedDataType("ANE states require Float16, Int8 or UInt8 storage"),
        )?;
        let input = self.input_placeholder(data.shape(), data.data_type())?;
        self.state()
            .states
            .insert(input, StateData::Surface(data.clone()));
        Ok(input)
    }

    /// A Float16 variable whose storage the caller passes as an input on every run.
    pub fn variable_placeholder<const RANK: usize>(
        &self,
        shape: [usize; RANK],
    ) -> Result<Tensor, GraphError> {
        self.placeholder(shape, DataType::Float16)
    }

    /// The variable's value at this point of the graph. It reflects an earlier assign only in
    /// executables that perform that assign. MIL `read_state`.
    pub fn read_variable(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.ensure_state(*input)?;
        let result =
            self.state()
                .alloc_typed(input.physical_shape(), input.rank(), input.data_type());
        self.state().ops.push((
            Op::StateRead(StateReadOp {
                top: result,
                state: *input,
            }),
            result,
        ));
        Ok(result)
    }

    /// Writes `value` into the variable when the returned operation is one of the executable's target
    /// operations. MIL `write_state`.
    pub fn assign_variable(&self, input: &Tensor, value: &Tensor) -> Result<Operation, GraphError> {
        self.ensure_state(*input)?;
        self.check_tensor(*value)?;
        ensure(
            value.data_type() == input.data_type(),
            GraphError::UnsupportedDataType("state assignment storage type differs"),
        )?;
        ensure(
            value.physical_shape() == input.physical_shape(),
            GraphError::ShapeMismatch("state assignment shape differs"),
        )?;
        let result =
            self.state()
                .alloc_typed(input.physical_shape(), input.rank(), input.data_type());
        let mut state = self.state();
        state.ops.push((
            Op::StateWrite(StateWriteOp {
                top: result,
                state: *input,
                bottom: *value,
            }),
            result,
        ));
        Ok(state_operation(result))
    }

    /// Writes `update` into the variable from the runtime scalar Int32 row `position` and channel
    /// `channel`, when the returned operation is a target operation. MIL `slice_update` and
    /// `write_state`; the begin and end vectors are assembled in the program because the ANE takes
    /// dynamic offsets only as scalar parameters.
    pub fn assign_variable_rows(
        &self,
        input: &Tensor,
        update: &Tensor,
        position: &Tensor,
        channel: usize,
    ) -> Result<Operation, GraphError> {
        self.ensure_state(*input)?;
        self.check_tensor(*update)?;
        ensure(
            update.data_type() == input.data_type(),
            GraphError::UnsupportedDataType("state update storage type differs"),
        )?;
        self.check_tensor(*position)?;
        let shape = input.physical_shape();
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
            .alloc_typed(shape, input.rank(), input.data_type());
        let mut state = self.state();
        state.ops.push((
            Op::StateUpdate(StateUpdateOp {
                top: result,
                bottom: *update,
                state: *input,
                position: *position,
                rows: update_shape[2],
                channel,
                channels: update_shape[1],
            }),
            result,
        ));
        Ok(state_operation(result))
    }

    fn ensure_state(&self, input: Tensor) -> Result<(), GraphError> {
        self.check_tensor(input)?;
        ensure(
            self.state().inputs.iter().any(|(tensor, dtype)| {
                *tensor == input
                    && matches!(dtype, DataType::Float16 | DataType::Int8 | DataType::UInt8)
            }),
            GraphError::InvalidArgument("tensor is not a state"),
        )
    }

    /// A Float16 constant. Identical constants are stored once per model.
    pub fn constant<const RANK: usize>(
        &self,
        data: &[f32],
        shape: [usize; RANK],
    ) -> Result<Tensor, GraphError> {
        self.constant_values(data, logical_shape(&shape))
    }

    /// A constant from raw Float16, Int8 or UInt8 bytes.
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
        state.constants.insert(tensor.id(), blob);
        Ok(tensor)
    }
}
