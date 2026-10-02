use crate::TensorSpec;

#[derive(Debug)]
pub struct CompilationReport<'a> {
    pub inputs: &'a [TensorSpec],
    pub outputs: &'a [TensorSpec],
    pub operations: &'a [String],
    pub anec_bytes: usize,
    pub constant_bytes: usize,
}
