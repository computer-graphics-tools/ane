#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementwiseOpType {
    Add = 0,
    Multiply = 1,
    Max = 3,
    Min = 4,
    Inverse = 10,
    Sqrt = 11,
    Rsqrt = 12,
    Pow = 13,
    Abs = 24,
    Threshold = 25,
    Log = 26,
    Exp = 27,
    Sub = 28,
    Div = 29,
    Floor = 30,
}
