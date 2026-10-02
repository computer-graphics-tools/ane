use crate::{CoordinateMode, PadFillMode, SamplingMode};

#[derive(Clone, Debug)]
pub struct SamplingDescriptor {
    pub mode: SamplingMode,
    pub padding: PadFillMode,
    pub padding_value: f32,
    pub coordinates: CoordinateMode,
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
