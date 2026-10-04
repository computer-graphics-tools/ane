use crate::{CoordinateMode, PadFillMode, SamplingMode};

/// How `sample_grid` interpolates and treats out-of-range coordinates.
#[derive(Clone, Debug)]
pub struct SamplingDescriptor {
    /// Interpolation between pixels.
    pub mode: SamplingMode,
    /// `Constant` returns `padding_value` for every sample outside `[0, size - 1]`, as Core ML's
    /// `resample` and TensorFlow's `crop_and_resize` do; border samples are not blended.
    pub padding: PadFillMode,
    /// Value of samples outside the input in `Constant` mode.
    pub padding_value: f32,
    /// How coordinates map onto pixels.
    pub coordinates: CoordinateMode,
    /// Maps normalized coordinate extremes to corner pixel centres.
    pub align_corners: bool,
}
impl Default for SamplingDescriptor {
    fn default() -> Self {
        Self {
            mode: SamplingMode::Bilinear,
            padding: PadFillMode::Constant,
            padding_value: 0.0,
            coordinates: CoordinateMode::MinusOneToOne,
            align_corners: false,
        }
    }
}
