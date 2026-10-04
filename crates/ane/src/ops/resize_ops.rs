use crate::DataType;
use crate::ResizeSamplingMode;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    /// Nearest-neighbour resize of the last two axes to `size`. MIL `resize_nearest_neighbor`.
    pub fn resize_nearest(&self, x: &Tensor, size: [usize; 2]) -> Result<Tensor, GraphError> {
        self.resize(*x, size, Operator::ResizeNearestNeighbor, &[])
    }

    /// Bilinear resize of the last two axes to `size` with MIL's `sampling_mode`. MIL `resize_bilinear`.
    pub fn resize_bilinear(
        &self,
        x: &Tensor,
        size: [usize; 2],
        sampling_mode: ResizeSamplingMode,
    ) -> Result<Tensor, GraphError> {
        self.resize(
            *x,
            size,
            Operator::ResizeBilinear,
            &[(
                Parameter::SamplingMode,
                Value::String(sampling_mode.as_str()),
            )],
        )
    }

    fn resize(
        &self,
        x: Tensor,
        size: [usize; 2],
        operation: Operator,
        extra: &[(Parameter, Value)],
    ) -> Result<Tensor, GraphError> {
        self.numeric(x)?;
        ensure(
            x.rank() >= 2 && size.iter().all(|&n| n > 0),
            GraphError::ShapeMismatch("resize requires spatial dimensions and a positive size"),
        )?;
        let mut shape = x.physical_shape();
        shape[2] = size[0];
        shape[3] = size[1];
        let mut attributes = vec![
            (Parameter::TargetSizeHeight, Value::Int32(size[0])),
            (Parameter::TargetSizeWidth, Value::Int32(size[1])),
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
