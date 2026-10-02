use crate::PadMode;
use crate::graph::{GraphError, checked_shape, ensure};

#[derive(Clone, Debug)]
pub struct Pooling2dDescriptor {
    pub kernel: [usize; 2],
    pub strides: [usize; 2],
    pub padding: [usize; 4],
    pub pad_mode: PadMode,
    pub ceil_mode: bool,
    pub exclude_padding: bool,
}
impl Pooling2dDescriptor {
    pub fn new(kernel: [usize; 2], strides: [usize; 2]) -> Self {
        Self {
            kernel,
            strides,
            padding: [0; 4],
            pad_mode: PadMode::Valid,
            ceil_mode: false,
            exclude_padding: false,
        }
    }
    pub fn output_shape(&self, input: [usize; 4]) -> Result<[usize; 4], GraphError> {
        checked_shape(&input)?;
        ensure(
            self.kernel
                .iter()
                .chain(&self.strides)
                .all(|&v| v > 0 && v <= i32::MAX as usize)
                && self.padding.iter().all(|&v| v <= i32::MAX as usize),
            GraphError::InvalidArgument("invalid pooling geometry"),
        )?;
        let mut output = input;
        for axis in 0..2 {
            output[axis + 2] = if self.pad_mode == PadMode::Same {
                ensure(
                    self.padding == [0; 4] && !self.ceil_mode,
                    GraphError::InvalidArgument(
                        "same pooling padding cannot combine with explicit padding or ceil mode",
                    ),
                )?;
                input[axis + 2].div_ceil(self.strides[axis])
            } else {
                let padded = input[axis + 2]
                    .checked_add(self.padding[axis * 2])
                    .and_then(|v| v.checked_add(self.padding[axis * 2 + 1]))
                    .ok_or(GraphError::Overflow)?;
                ensure(
                    padded >= self.kernel[axis],
                    GraphError::OutOfBounds("pool kernel exceeds padded input"),
                )?;
                let extent = padded - self.kernel[axis];
                let mut count = if self.ceil_mode {
                    extent.div_ceil(self.strides[axis]) + 1
                } else {
                    extent / self.strides[axis] + 1
                };
                if self.ceil_mode
                    && (count - 1) * self.strides[axis] >= input[axis + 2] + self.padding[axis * 2]
                {
                    count -= 1;
                }
                count
            };
        }
        checked_shape(&output)
    }
}
