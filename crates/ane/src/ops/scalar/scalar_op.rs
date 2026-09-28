use super::ScalarOpType;

#[derive(Clone)]
pub struct ScalarOp {
    pub name: String,
    pub bottom: String,
    pub top: String,
    pub op: ScalarOpType,
    pub scalar: f32,
}
