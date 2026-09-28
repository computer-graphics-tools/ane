#[derive(Clone)]
pub struct StateUpdateOp {
    pub name: String,
    pub top: String,
    pub bottom: String,
    pub state: String,
    pub position: String,
    pub rows: usize,
    pub channel: usize,
    pub channels: usize,
}
