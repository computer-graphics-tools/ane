use crate::DataType;
use crate::PadFillMode;
use crate::SamplingDescriptor;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    fn sampling_attributes(
        descriptor: &SamplingDescriptor,
    ) -> Result<Vec<(Parameter, Value)>, GraphError> {
        ensure(
            descriptor.padding_value.is_finite(),
            GraphError::InvalidArgument("sampling padding must be finite"),
        )?;
        let padding = match descriptor.padding {
            PadFillMode::Constant => "constant",
            PadFillMode::Reflect => "reflection",
            PadFillMode::Symmetric => "symmetric",
            PadFillMode::Replicate => "border",
        };
        Ok(vec![
            (
                Parameter::SamplingMode,
                Value::String(descriptor.mode.as_str()),
            ),
            (Parameter::PaddingMode, Value::String(padding)),
            (
                Parameter::PaddingValue,
                Value::Fp16(descriptor.padding_value),
            ),
            (
                Parameter::CoordinatesMode,
                Value::String(descriptor.coordinates.as_str()),
            ),
            (
                Parameter::AlignCorners,
                Value::Bool(descriptor.align_corners),
            ),
        ])
    }

    /// Samples `input` at `[N, H, W, 2]` coordinates as the descriptor configures. MIL `resample`.
    pub fn sample_grid(
        &self,
        input: &Tensor,
        coordinates: &Tensor,
        descriptor: &SamplingDescriptor,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        self.numeric(*coordinates)?;
        ensure(
            input.rank() == 4
                && coordinates.rank() == 4
                && input.physical_shape()[0] == coordinates.physical_shape()[0]
                && coordinates.physical_shape()[3] == 2,
            GraphError::ShapeMismatch(
                "grid sampling requires NCHW input and [batch,height,width,2] coordinates",
            ),
        )?;
        let shape = [
            input.physical_shape()[0],
            input.physical_shape()[1],
            coordinates.physical_shape()[1],
            coordinates.physical_shape()[2],
        ];
        self.builtin(
            Operator::Resample,
            &[
                (Parameter::X, *input),
                (Parameter::Coordinates, *coordinates),
            ],
            &Self::sampling_attributes(descriptor)?,
            &shape,
            DataType::Float16,
        )
    }
}
