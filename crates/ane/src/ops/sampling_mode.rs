#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingMode {
    Nearest,
    Bilinear,
}
impl SamplingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
        }
    }
}
