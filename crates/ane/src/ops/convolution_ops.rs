use crate::Convolution2dDescriptor;
use crate::DataType;
use crate::PadMode;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Op, Operator, Parameter, Value};

impl Graph {
    pub fn convolution_2d_1x1(
        &self,
        input: &Tensor,
        weights: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        self.numeric(*weights)?;
        ensure(
            weights.physical_shape()[2..] == [1, 1],
            GraphError::ShapeMismatch("1x1 convolution requires a 1x1 kernel"),
        )?;
        let constant = {
            let state = self.state();
            state.constants.contains_key(&weights.id())
                || state.ops.iter().any(|(op, t)| {
                    *t == *weights && matches!(op, Op::Builtin(op) if op.operation.is_constant())
                })
        };
        if !constant {
            let shape = Convolution2dDescriptor::default()
                .output_shape(input.physical_shape(), weights.physical_shape())?;
            let count = shape[0]
                .checked_mul(shape[2])
                .and_then(|n| n.checked_mul(shape[3]))
                .ok_or(GraphError::Overflow)?;
            let input = self.reshape_to(*input, &input.physical_shape())?;
            let input = self.transpose(&input, [0, 2, 3, 1])?;
            let input = self.reshape(&input, [count, weights.physical_shape()[1]])?;
            let weights = self.reshape(weights, [shape[1], weights.physical_shape()[1]])?;
            let output = self.matrix_multiplication(&input, &weights, false, true)?;
            let output = self.reshape(&output, [shape[0], shape[2], shape[3], shape[1]])?;
            let output = self.transpose(&output, [0, 3, 1, 2])?;
            let Some(bias) = bias else {
                return Ok(output);
            };
            self.check_tensor(*bias)?;
            ensure(
                bias.physical_shape().iter().product::<usize>() == shape[1],
                GraphError::ShapeMismatch("convolution bias count differs"),
            )?;
            let bias = self.reshape(bias, [1, shape[1], 1, 1])?;
            return self.addition(&output, &bias);
        }
        self.convolution_2d(input, weights, bias, &Convolution2dDescriptor::default())
    }

    pub fn convolution_2d(
        &self,
        input: &Tensor,
        weights: &Tensor,
        bias: Option<&Tensor>,
        descriptor: &Convolution2dDescriptor,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        self.numeric(*weights)?;
        let shape = descriptor.output_shape(input.physical_shape(), weights.physical_shape())?;
        let padding = if descriptor.pad_mode == PadMode::Same {
            "same_lower"
        } else if descriptor.padding == [0; 4] {
            "valid"
        } else {
            "custom"
        };
        let attrs = [
            (Parameter::Strides, Value::int32_list(&descriptor.strides)),
            (
                Parameter::Dilations,
                Value::int32_list(&descriptor.dilations),
            ),
            (Parameter::Groups, Value::Int32(descriptor.groups)),
            (Parameter::Pad, Value::int32_list(&descriptor.padding)),
            (Parameter::PadType, Value::String(padding)),
        ];
        let output = self.builtin(
            Operator::Conv,
            &[(Parameter::X, *input), (Parameter::Weight, *weights)],
            &attrs,
            &shape,
            DataType::Float16,
        )?;
        self.convolution_bias(output, bias.copied(), shape[1])
    }
}
