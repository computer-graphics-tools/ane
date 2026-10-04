/// Coordinate system of `sample_grid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinateMode {
    /// Pixel coordinates.
    Unnormalized,
    /// `-1` and `1` are the input extremes.
    MinusOneToOne,
    /// `0` and `1` are the input extremes.
    ZeroToOne,
}
impl CoordinateMode {
    /// MIL name of the mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unnormalized => "unnormalized",
            Self::MinusOneToOne => "normalized_minus_one_to_one",
            Self::ZeroToOne => "normalized_zero_to_one",
        }
    }
}
