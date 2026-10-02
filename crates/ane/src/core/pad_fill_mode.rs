#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadFillMode {
    Constant = 0,
    Reflect = 1,
    Replicate = 2,
    Symmetric = 3,
}
