use crate::DataType;
use crate::PadMode;
use crate::PoolType;
use crate::Pooling2dDescriptor;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::GraphError;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    fn pool(
        &self,
        input: Tensor,
        pool_type: PoolType,
        kernel: [usize; 2],
        strides: [usize; 2],
        pad_mode: PadMode,
    ) -> Result<Tensor, GraphError> {
        let mut descriptor = Pooling2dDescriptor::new(kernel, strides);
        descriptor.pad_mode = pad_mode;
        self.pooling_2d(&input, pool_type, &descriptor)
    }

    pub fn pooling_2d(
        &self,
        input: &Tensor,
        pool_type: PoolType,
        descriptor: &Pooling2dDescriptor,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        let shape = descriptor.output_shape(input.physical_shape())?;
        let mode = if descriptor.pad_mode == PadMode::Same {
            "same_lower"
        } else if descriptor.padding == [0; 4] {
            "valid"
        } else {
            "custom"
        };
        let op = match pool_type {
            PoolType::Average => Operator::AvgPool,
            PoolType::Max => Operator::MaxPool,
            PoolType::L2 => Operator::L2Pool,
        };
        let mut attrs = vec![
            (
                Parameter::KernelSizes,
                Value::int32_list(&descriptor.kernel),
            ),
            (Parameter::Strides, Value::int32_list(&descriptor.strides)),
            (Parameter::Pad, Value::int32_list(&descriptor.padding)),
            (Parameter::PadType, Value::String(mode)),
            (Parameter::CeilMode, Value::Bool(descriptor.ceil_mode)),
        ];
        if pool_type == PoolType::Average {
            attrs.push((
                Parameter::ExcludePaddingFromAverage,
                Value::Bool(descriptor.exclude_padding),
            ));
        }
        self.builtin(
            op,
            &[(Parameter::X, *input)],
            &attrs,
            &shape[4 - input.rank()..],
            DataType::Float16,
        )
    }

    pub fn global_avg_pool(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        let kh = input.physical_shape()[2];
        let kw = input.physical_shape()[3];
        self.pool(*input, PoolType::Average, [kh, kw], [1, 1], PadMode::Valid)
    }

    pub fn global_max_pool(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        self.pool(
            *input,
            PoolType::Max,
            [input.physical_shape()[2], input.physical_shape()[3]],
            [1, 1],
            PadMode::Valid,
        )
    }
}
