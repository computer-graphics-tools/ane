use crate::CoordinateMode;
use crate::DataType;
use crate::SamplingDescriptor;
use crate::SamplingMode;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    pub fn resize(
        &self,
        input: &Tensor,
        size: [usize; 2],
        mode: SamplingMode,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        ensure(
            input.rank() >= 2,
            GraphError::ShapeMismatch("resize requires spatial dimensions"),
        )?;
        let mut shape = input.physical_shape();
        shape[2] = size[0];
        shape[3] = size[1];
        let mut attrs = vec![
            (Parameter::TargetSizeHeight, Value::Int32(size[0])),
            (Parameter::TargetSizeWidth, Value::Int32(size[1])),
        ];
        let op = match mode {
            SamplingMode::Nearest => Operator::ResizeNearestNeighbor,
            SamplingMode::Bilinear => {
                attrs.push((Parameter::SamplingMode, Value::String("UNALIGN_CORNERS")));
                Operator::ResizeBilinear
            }
        };
        self.builtin(
            op,
            &[(Parameter::X, *input)],
            &attrs,
            &shape[4 - input.rank()..],
            input.data_type(),
        )
    }

    pub fn upsample(
        &self,
        input: &Tensor,
        factors: [usize; 2],
        mode: SamplingMode,
    ) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        let size = [
            input.physical_shape()[2].checked_mul(factors[0]),
            input.physical_shape()[3].checked_mul(factors[1]),
        ];
        let size = [
            size[0].ok_or(GraphError::Overflow)?,
            size[1].ok_or(GraphError::Overflow)?,
        ];
        self.resize(input, size, mode)
    }

    pub fn crop_resize(
        &self,
        input: &Tensor,
        boxes_xyxy: &Tensor,
        box_indices: &Tensor,
        size: [usize; 2],
        normalized: bool,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        self.numeric(*boxes_xyxy)?;
        self.check_tensor(*box_indices)?;
        ensure(
            input.rank() == 4
                && boxes_xyxy.rank() == 2
                && boxes_xyxy.physical_shape()[3] == 4
                && box_indices.rank() == 1
                && box_indices.physical_shape()[3] == boxes_xyxy.physical_shape()[2]
                && box_indices.data_type() == DataType::UInt16,
            GraphError::ShapeMismatch(
                "crop-resize requires NCHW, [boxes,4] xyxy coordinates and [boxes] UInt16 indices",
            ),
        )?;
        ensure(
            size[0] > 0 && size[1] > 0,
            GraphError::InvalidArgument("crop-resize size must be positive"),
        )?;
        let boxes = boxes_xyxy.physical_shape()[2];
        let images = self.gather(input, box_indices, 0)?;
        let ramp = |count: usize| -> Vec<f32> {
            if count == 1 {
                vec![0.5]
            } else {
                (0..count).map(|i| i as f32 / (count - 1) as f32).collect()
            }
        };
        let corner = |index: usize| {
            let value = self.slice(boxes_xyxy, [0, index], [boxes, 1])?;
            self.reshape(&value, [boxes, 1, 1, 1])
        };
        let axis = |start: Tensor, end: Tensor, ramp: Tensor| {
            let extent = self.subtraction(&end, &start)?;
            let offset = self.multiplication(&ramp, &extent)?;
            let value = self.addition(&start, &offset)?;
            self.broadcast_to(&value, [boxes, size[0], size[1], 1])
        };
        let rows = self.constant(&ramp(size[0]), [1, size[0], 1, 1])?;
        let columns = self.constant(&ramp(size[1]), [1, 1, size[1], 1])?;
        let x = axis(corner(0)?, corner(2)?, columns)?;
        let y = axis(corner(1)?, corner(3)?, rows)?;
        let grid = self.concat(&[&x, &y], 3)?;
        self.sample_grid(
            &images,
            &grid,
            &SamplingDescriptor {
                coordinates: if normalized {
                    CoordinateMode::ZeroToOne
                } else {
                    CoordinateMode::Unnormalized
                },
                align_corners: true,
                ..SamplingDescriptor::default()
            },
        )
    }
}
