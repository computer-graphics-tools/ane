use crate::ir::{IrError, MilEmitter, Program};

pub struct MilProgram {
    pub text: String,
    pub weights: Vec<u8>,
    pub outputs: Vec<Vec<String>>,
}

impl MilProgram {
    pub const WEIGHT_PATH: &str = "@model_path/weights/weight.bin";

    pub fn new(functions: &[Program]) -> Result<Self, IrError> {
        MilEmitter::emit(functions)
    }
}
