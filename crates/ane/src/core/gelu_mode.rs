#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeluMode {
    Exact,
    TanhApproximation,
    SigmoidApproximation,
}

impl GeluMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "EXACT",
            Self::TanhApproximation => "TANH_APPROXIMATION",
            Self::SigmoidApproximation => "SIGMOID_APPROXIMATION",
        }
    }
}
