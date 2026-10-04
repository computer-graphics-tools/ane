#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadFillMode {
    Constant,
    Reflect,
    Replicate,
    Symmetric,
}
