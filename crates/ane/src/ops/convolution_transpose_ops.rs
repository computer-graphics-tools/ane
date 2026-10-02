use crate::ConvolutionTranspose2dDescriptor;
use crate::DataType;
use crate::PadMode;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::GraphError;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    pub fn convolution_transpose_2d(
        &self,
        input: &Tensor,
        weights: &Tensor,
        bias: Option<&Tensor>,
        descriptor: &ConvolutionTranspose2dDescriptor,
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
            (Parameter::OutputShape, Value::int32_list(&shape)),
        ];
        let output = self.builtin(
            Operator::ConvTranspose,
            &[(Parameter::X, *input), (Parameter::Weight, *weights)],
            &attrs,
            &shape,
            DataType::Float16,
        )?;
        self.convolution_bias(output, bias.copied(), shape[1])
    }
}
