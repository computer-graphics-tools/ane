#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarOpType {
    Mul,
    Add,
    RSub,
    Pow,
    Min,
    Max,
}
