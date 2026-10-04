#[derive(Debug)]
pub struct CompilationReport<'a> {
    pub operations: &'a [String],
    pub mil_bytes: usize,
    pub constant_bytes: usize,
}
