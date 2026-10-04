use crate::PadMode;
use crate::graph::{GraphError, checked_shape, ensure};

/// Geometry of a transposed 2-D convolution.
#[derive(Clone, Debug)]
pub struct ConvolutionTranspose2dDescriptor {
    /// Channel groups.
    pub groups: usize,
    /// Vertical and horizontal stride.
    pub strides: [usize; 2],
    /// Vertical and horizontal kernel dilation.
    pub dilations: [usize; 2],
    /// `[top, bottom, left, right]` cropped from the full output.
    pub padding: [usize; 4],
    /// Extra rows and columns added to the bottom and right.
    pub output_padding: [usize; 2],
    /// `Same` produces `input · stride` outputs; `Valid` uses `padding`.
    pub pad_mode: PadMode,
}

impl ConvolutionTranspose2dDescriptor {
    /// Output shape for an `[N, Cin, H, W]` input and `[Cin, Cout / groups, kH, kW]` weights.
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
                && self
                    .padding
                    .iter()
                    .chain(&self.output_padding)
                    .all(|&n| n <= i32::MAX as usize),
            GraphError::InvalidArgument("invalid transposed convolution geometry"),
        )?;
        ensure(
            self.groups > 0 && input[1].is_multiple_of(self.groups) && weights[0] == input[1],
            GraphError::ShapeMismatch(
                "transposed convolution weights must be [input channels, output channels per group, height, width]",
            ),
        )?;
        let channels = weights[1]
            .checked_mul(self.groups)
            .ok_or(GraphError::Overflow)?;
        let mut output = [input[0], channels, 1, 1];
        for axis in 0..2 {
            let stride = self.strides[axis];
            let dilation = self.dilations[axis];
            ensure(
                stride > 0 && dilation > 0 && self.output_padding[axis] < stride,
                GraphError::InvalidArgument(
                    "invalid transposed convolution stride, dilation or output padding",
                ),
            )?;
            let size = if self.pad_mode == PadMode::Same {
                ensure(
                    self.padding == [0; 4],
                    GraphError::InvalidArgument(
                        "same padding cannot be combined with explicit padding",
                    ),
                )?;
                input[axis + 2].checked_mul(stride)
            } else {
                (input[axis + 2] - 1)
                    .checked_mul(stride)
                    .and_then(|n| {
                        (weights[axis + 2] - 1)
                            .checked_mul(dilation)
                            .and_then(|k| n.checked_add(k))
                    })
                    .and_then(|n| n.checked_add(1))
                    .and_then(|n| n.checked_sub(self.padding[2 * axis]))
                    .and_then(|n| n.checked_sub(self.padding[2 * axis + 1]))
            };
            output[axis + 2] = size
                .and_then(|n| n.checked_add(self.output_padding[axis]))
                .ok_or(GraphError::Overflow)?;
        }
        checked_shape(&output)
    }
}

impl Default for ConvolutionTranspose2dDescriptor {
    fn default() -> Self {
        Self {
            groups: 1,
            strides: [1; 2],
            dilations: [1; 2],
            padding: [0; 4],
            output_padding: [0; 2],
            pad_mode: PadMode::Valid,
        }
    }
}
