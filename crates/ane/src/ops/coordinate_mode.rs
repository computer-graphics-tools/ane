#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinateMode {
    Unnormalized,
    MinusOneToOne,
    ZeroToOne,
}
impl CoordinateMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unnormalized => "unnormalized",
            Self::MinusOneToOne => "normalized_minus_one_to_one",
            Self::ZeroToOne => "normalized_zero_to_one",
        }
    }
}
