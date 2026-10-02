use crate::graph::{GraphError, checked_shape, ensure};
use crate::ir::{Op, Operator, Parameter, Value, WeightBlob};
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

    fn logical_builtin(
        &self,
        operation: Operator,
        inputs: &[(Parameter, Tensor)],
        attributes: &[(Parameter, Value)],
        shapes: &[(DataType, &[usize])],
    ) -> Result<Vec<Tensor>, GraphError> {
        self.state()
            .logical_builtin(operation, inputs, attributes, shapes)
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
            .insert(tensor.id(), (WeightBlob::from_f32(data)?, physical));
        Ok(tensor)
    }

    fn constant_scalar(&self, scalar: f32, shape: &[usize]) -> Result<Tensor, GraphError> {
        let count = checked_shape(shape)?.iter().product();
        self.constant_values(&vec![scalar; count], shape)
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

    fn unary(&self, input: Tensor, operation: Operator) -> Result<Tensor, GraphError> {
        self.numeric(input)?;
        let epsilon = [(Parameter::Epsilon, Value::Fp16(0.0))];
        let attributes = if matches!(
            operation,
            Operator::Log | Operator::Inverse | Operator::Rsqrt
        ) {
            &epsilon[..]
        } else {
            &[]
        };
        self.builtin(
            operation,
            &[(Parameter::X, input)],
            attributes,
            input.shape(),
            DataType::Float16,
        )
    }

    fn ranked(
        &self,
        input: Tensor,
        k: usize,
        axis: i64,
        ascending: bool,
    ) -> Result<(Tensor, Tensor), GraphError> {
        self.numeric(input)?;
        let axis = self.axis(input, axis)?;
        ensure(
            k > 0 && k <= input.physical_shape()[axis] && input.physical_shape()[axis] <= 2048,
            GraphError::OutOfBounds(
                "native top-k indices are exact only for axis lengths up to 2048; split larger axes into tiles",
            ),
        )?;
        let mut shape = input.physical_shape();
        shape[axis] = k;
        let logical = &shape[4 - input.rank()..];
        let outputs = self.builtin_many(
            Operator::Topk,
            &[(Parameter::X, input)],
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

    fn convolution_bias(
        &self,
        output: Tensor,
        bias: Option<Tensor>,
        channels: usize,
    ) -> Result<Tensor, GraphError> {
        if let Some(bias) = bias {
            self.check_tensor(bias)?;
            ensure(
                bias.physical_shape().iter().product::<usize>() == channels,
                GraphError::ShapeMismatch("convolution bias count differs"),
            )?;
            let blob = self
                .state()
                .constants
                .get(&bias.id())
                .ok_or(GraphError::NonConstantWeights(
                    "convolution bias must be constant",
                ))?
                .0
                .clone();
            if let Some((Op::Builtin(op), _)) = self.state().ops.last_mut() {
                op.blobs = vec![(Parameter::Bias, vec![channels].into(), blob).into()].into();
            }
        }
        Ok(output)
    }
}
