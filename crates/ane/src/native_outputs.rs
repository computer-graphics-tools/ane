use crate::TensorSpec;

#[derive(Clone, Default)]
pub struct NativeOutputs {
    pub states: Vec<(u32, usize)>,
    pub discarded: Vec<(u32, TensorSpec)>,
}
