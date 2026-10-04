use crate::DataType;
use crate::Pooling2dDescriptor;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::GraphError;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::ir::{Operator, Parameter, Value};
use crate::ops::pad_type;

impl Graph {
    /// 2-D average pooling over the last two axes. MIL `avg_pool`.
    pub fn avg_pooling_2d(
        &self,
        x: &Tensor,
        descriptor: &Pooling2dDescriptor,
    ) -> Result<Tensor, GraphError> {
        self.pool(
            *x,
            Operator::AvgPool,
            descriptor,
            &[(
                Parameter::ExcludePaddingFromAverage,
                Value::Bool(descriptor.exclude_padding),
            )],
        )
    }

    /// 2-D max pooling over the last two axes; the ANE rejects `ceil_mode` together with padding.
    /// MIL `max_pool`.
    pub fn max_pooling_2d(
        &self,
        x: &Tensor,
        descriptor: &Pooling2dDescriptor,
    ) -> Result<Tensor, GraphError> {
        self.pool(*x, Operator::MaxPool, descriptor, &[])
    }

    /// 2-D L2-norm pooling over the last two axes. MIL `l2_pool`.
    pub fn l2_norm_pooling_2d(
        &self,
        x: &Tensor,
        descriptor: &Pooling2dDescriptor,
    ) -> Result<Tensor, GraphError> {
        self.pool(*x, Operator::L2Pool, descriptor, &[])
    }

    fn pool(
        &self,
        x: Tensor,
        operation: Operator,
        descriptor: &Pooling2dDescriptor,
        extra: &[(Parameter, Value)],
    ) -> Result<Tensor, GraphError> {
        self.numeric(x)?;
        let shape = descriptor.output_shape(x.physical_shape())?;
        let mut attributes = vec![
            (
                Parameter::KernelSizes,
                Value::int32_list(&descriptor.kernel),
            ),
            (Parameter::Strides, Value::int32_list(&descriptor.strides)),
            (Parameter::Pad, Value::int32_list(&descriptor.padding)),
            (
                Parameter::PadType,
                Value::String(pad_type(descriptor.pad_mode, descriptor.padding)),
            ),
            (Parameter::CeilMode, Value::Bool(descriptor.ceil_mode)),
        ];
        attributes.extend_from_slice(extra);
        self.builtin(
            operation,
            &[(Parameter::X, x)],
            &attributes,
            &shape[4 - x.rank()..],
            DataType::Float16,
        )
    }
}
