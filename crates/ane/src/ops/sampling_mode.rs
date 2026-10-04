/// Interpolation of `sample_grid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingMode {
    /// Nearest pixel.
    Nearest,
    /// Bilinear blend of the four nearest pixels.
    Bilinear,
}
impl SamplingMode {
    /// MIL name of the mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
        }
    }
}
