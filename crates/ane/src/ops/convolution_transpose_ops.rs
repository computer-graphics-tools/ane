use crate::ConvolutionTranspose2dDescriptor;
use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};
use crate::ops::pad_type;

impl Graph {
    /// Transposed 2-D convolution with constant weights `[Cin, Cout / groups, kH, kW]` and an optional
    /// constant bias `[Cout]`. Runtime weights are rejected. MIL `conv_transpose`.
    pub fn convolution_transpose_2d(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        descriptor: &ConvolutionTranspose2dDescriptor,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*x)?;
        self.numeric(*weight)?;
        ensure(
            self.state().is_constant(*weight),
            GraphError::NonConstantWeights(
                "ANE convolution weights must be compile-time constants",
            ),
        )?;
        let shape = descriptor.output_shape(x.physical_shape(), weight.physical_shape())?;
        let constants = bias
            .map(|bias| self.constant_input(*bias, Parameter::Bias, &[shape[1]]))
            .transpose()?
            .into_iter()
            .collect();
        self.builtin_with_constants(
            Operator::ConvTranspose,
            &[(Parameter::X, *x), (Parameter::Weight, *weight)],
            &[
                (Parameter::Strides, Value::int32_list(&descriptor.strides)),
                (
                    Parameter::Dilations,
                    Value::int32_list(&descriptor.dilations),
                ),
                (Parameter::Groups, Value::Int32(descriptor.groups)),
                (Parameter::Pad, Value::int32_list(&descriptor.padding)),
                (
                    Parameter::PadType,
                    Value::String(pad_type(descriptor.pad_mode, descriptor.padding)),
                ),
                (Parameter::OutputShape, Value::int32_list(&shape)),
            ],
            constants,
            &shape,
            DataType::Float16,
        )
    }
}
