use crate::graph::{GraphError, checked_shape, ensure};
use crate::ir::{ConstantInput, Operator, Parameter, Value, WeightBlob};
use std::cell::RefMut;

use crate::DataType;
use crate::graph::{GraphState, Tensor, TensorHandle};

pub trait GraphBuilder {
    fn state(&self) -> RefMut<'_, GraphState>;

    fn builtin(
        &self,
        operation: Operator,
        inputs: &[(Parameter, Tensor)],
        attributes: &[(Parameter, Value)],
        shape: &[usize],
        dtype: DataType,
    ) -> Result<Tensor, GraphError> {
        Ok(self.builtin_many(operation, inputs, attributes, &[(dtype, shape)])?[0])
    }

    fn builtin_many(
        &self,
        operation: Operator,
        inputs: &[(Parameter, Tensor)],
        attributes: &[(Parameter, Value)],
        shapes: &[(DataType, &[usize])],
    ) -> Result<Vec<Tensor>, GraphError> {
        self.state()
            .builtin_many(operation, inputs, attributes, shapes, Box::new([]))
    }

    fn numeric(&self, tensor: Tensor) -> Result<(), GraphError> {
        self.state().numeric(tensor)
    }

    fn axis(&self, tensor: Tensor, axis: i64) -> Result<usize, GraphError> {
        self.state().axis(tensor, axis)
    }

    fn check_tensor(&self, tensor: Tensor) -> Result<(), GraphError> {
        self.state().check_tensor(tensor)
    }

    fn input_placeholder(
        &self,
        shape: &[usize],
        data_type: DataType,
    ) -> Result<Tensor, GraphError> {
        let physical = checked_shape(shape)?;
        let internal = if matches!(
            data_type,
            DataType::Bool
                | DataType::Int8
                | DataType::UInt8
                | DataType::Int16
                | DataType::UInt16
                | DataType::Int32
        ) {
            data_type
        } else {
            DataType::Float16
        };
        let mut state = self.state();
        let tensor = state.alloc_typed(physical, shape.len(), internal);
        state.inputs.push((tensor, data_type));
        Ok(tensor)
    }

    fn constant_values(&self, data: &[f32], shape: &[usize]) -> Result<Tensor, GraphError> {
        let physical = checked_shape(shape)?;
        ensure(
            data.len() == physical.iter().product::<usize>(),
            GraphError::ShapeMismatch("constant data length differs from shape"),
        )?;
        let mut state = self.state();
        let tensor = state.alloc_rank(physical, shape.len());
        state
            .constants
            .insert(tensor.id(), WeightBlob::from_f32(data)?);
        Ok(tensor)
    }

    fn reshape_to(&self, input: Tensor, shape: &[usize]) -> Result<Tensor, GraphError> {
        self.check_tensor(input)?;
        let physical = checked_shape(shape)?;
        ensure(
            physical.iter().product::<usize>() == input.physical_shape().iter().product::<usize>(),
            GraphError::ShapeMismatch("reshape element count differs"),
        )?;
        self.builtin(
            Operator::Reshape,
            &[(Parameter::X, input)],
            &[(Parameter::Shape, Value::int32_list(&physical))],
            shape,
            input.data_type(),
        )
    }

    fn unary(
        &self,
        input: Tensor,
        operation: Operator,
        attributes: &[(Parameter, Value)],
    ) -> Result<Tensor, GraphError> {
        self.numeric(input)?;
        self.builtin(
            operation,
            &[(Parameter::X, input)],
            attributes,
            input.shape(),
            DataType::Float16,
        )
    }

    fn constant_input(
        &self,
        tensor: Tensor,
        name: Parameter,
        shape: &[usize],
    ) -> Result<ConstantInput, GraphError> {
        self.check_tensor(tensor)?;
        let data = self.state().constants.get(&tensor.id()).cloned().ok_or(
            GraphError::NonConstantWeights("this MIL argument must be a constant tensor"),
        )?;
        ensure(
            data.element_count() == shape.iter().product::<usize>(),
            GraphError::ShapeMismatch("constant argument size differs from its MIL shape"),
        )?;
        Ok((name, shape.into(), data).into())
    }

    fn builtin_with_constants(
        &self,
        operation: Operator,
        inputs: &[(Parameter, Tensor)],
        attributes: &[(Parameter, Value)],
        constants: Vec<ConstantInput>,
        shape: &[usize],
        dtype: DataType,
    ) -> Result<Tensor, GraphError> {
        Ok(self.state().builtin_many(
            operation,
            inputs,
            attributes,
            &[(dtype, shape)],
            constants.into(),
        )?[0])
    }
}
