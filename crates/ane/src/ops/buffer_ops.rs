use crate::graph::{Graph, GraphBuilder, GraphError, Tensor, TensorHandle, checked_shape, ensure};
use crate::ir::{Operator, Parameter, Value};
use crate::{DataType, logical_shape, row_strides};

fn buffer_strides(
    shape: &[usize; 4],
    data_type: DataType,
    row_stride: usize,
) -> Result<[usize; 4], GraphError> {
    ensure(
        matches!(
            data_type,
            DataType::Float16 | DataType::Int8 | DataType::UInt8
        ),
        GraphError::UnsupportedDataType("surface buffers require Float16, Int8 or UInt8 storage"),
    )?;
    ensure(
        row_stride >= shape[3],
        GraphError::InvalidArgument("buffer row stride is shorter than a row"),
    )?;
    let row_bytes = row_stride
        .checked_mul(data_type.byte_width())
        .ok_or(GraphError::Overflow)?;
    ensure(
        row_bytes.is_multiple_of(64),
        GraphError::InvalidArgument(
            "buffer rows must be 64-byte aligned (see DataType::aligned_row_stride)",
        ),
    )?;
    let strides = row_strides(shape, row_stride).ok_or(GraphError::Overflow)?;
    let extent = strides[0].checked_mul(shape[0]);
    ensure(
        extent.is_some_and(|extent| extent < i32::MAX as usize),
        GraphError::Overflow,
    )?;
    Ok(strides)
}

impl Graph {
    /// An IOSurface input; `row_stride` is in elements and must give 64-byte-aligned rows.
    /// Supports Float16, Int8 and UInt8.
    pub fn placeholder_buffer<const RANK: usize>(
        &self,
        shape: [usize; RANK],
        data_type: DataType,
        row_stride: usize,
    ) -> Result<Tensor, GraphError> {
        let physical = checked_shape(logical_shape(&shape))?;
        let strides = buffer_strides(&physical, data_type, row_stride)?;
        let tensor = self.input_placeholder(logical_shape(&shape), data_type)?;
        self.state().input_buffer_strides.insert(tensor, strides);
        Ok(tensor)
    }

    /// Stores `tensor` in an IOSurface with the given row stride in elements.
    /// Rows must be 64-byte aligned. The result must be a compilation target with no consumers.
    pub fn buffer_output(&self, tensor: &Tensor, row_stride: usize) -> Result<Tensor, GraphError> {
        self.check_tensor(*tensor)?;
        let physical = tensor.physical_shape();
        let strides = buffer_strides(&physical, tensor.data_type(), row_stride)?;
        self.builtin(
            Operator::TensorToTensorBuffer,
            &[(Parameter::X, *tensor)],
            &[(Parameter::Strides, Value::int32_list(&strides))],
            tensor.shape(),
            tensor.data_type(),
        )
    }
}
