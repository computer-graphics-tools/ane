use crate::DataType;

pub struct MilProgram {
    pub text: String,
    pub weights: Box<[u8]>,
    pub inputs: Box<[(String, [usize; 4], DataType)]>,
    pub outputs: Box<[(String, [usize; 4], DataType)]>,
}
