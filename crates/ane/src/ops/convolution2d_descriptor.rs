use crate::PadMode;
use crate::graph::{GraphError, checked_shape, ensure};

#[derive(Clone, Debug)]
pub struct Convolution2dDescriptor {
    pub groups: usize,
    pub strides: [usize; 2],
    pub dilations: [usize; 2],
    pub padding: [usize; 4],
    pub pad_mode: PadMode,
}

impl Convolution2dDescriptor {
    pub fn output_shape(
        &self,
        input: [usize; 4],
        weights: [usize; 4],
    ) -> Result<[usize; 4], GraphError> {
        checked_shape(&input)?;
        checked_shape(&weights)?;
        ensure(
            self.strides
                .iter()
                .chain(&self.dilations)
                .all(|&n| n > 0 && n <= i32::MAX as usize)
                && self.padding.iter().all(|&n| n <= i32::MAX as usize),
            GraphError::InvalidArgument("invalid convolution geometry"),
        )?;
        ensure(
            self.groups > 0
                && input[1].is_multiple_of(self.groups)
                && weights[0].is_multiple_of(self.groups),
            GraphError::ShapeMismatch("convolution channels must divide into groups"),
        )?;
        ensure(
            weights[1] == input[1] / self.groups,
            GraphError::ShapeMismatch(
                "convolution weights must be [output channels, input channels per group, height, width]",
            ),
        )?;
        let mut output = [input[0], weights[0], 1, 1];
        for axis in 0..2 {
            let stride = self.strides[axis];
            let dilation = self.dilations[axis];
            let effective = (weights[axis + 2] - 1)
                .checked_mul(dilation)
                .and_then(|n| n.checked_add(1))
                .ok_or(GraphError::Overflow)?;
            output[axis + 2] = if self.pad_mode == PadMode::Same {
                ensure(
                    self.padding == [0; 4],
                    GraphError::InvalidArgument(
                        "same padding cannot be combined with explicit padding",
                    ),
                )?;
                input[axis + 2].div_ceil(stride)
            } else {
                let padded = input[axis + 2]
                    .checked_add(self.padding[axis * 2])
                    .and_then(|n| n.checked_add(self.padding[axis * 2 + 1]))
                    .ok_or(GraphError::Overflow)?;
                ensure(
                    padded >= effective,
                    GraphError::OutOfBounds("convolution kernel exceeds padded input"),
                )?;
                (padded - effective) / stride + 1
            };
        }
        checked_shape(&output)
    }
}

impl Default for Convolution2dDescriptor {
    fn default() -> Self {
        Self {
            groups: 1,
            strides: [1; 2],
            dilations: [1; 2],
            padding: [0; 4],
            pad_mode: PadMode::Valid,
        }
    }
}
