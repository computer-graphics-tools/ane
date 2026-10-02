use crate::ir::{Parameter, WeightBlob};

#[derive(Clone, PartialEq)]
pub struct ConstantInput {
    pub name: Parameter,
    pub shape: Box<[usize]>,
    pub data: WeightBlob,
}

impl From<(Parameter, Box<[usize]>, WeightBlob)> for ConstantInput {
    fn from((name, shape, data): (Parameter, Box<[usize]>, WeightBlob)) -> Self {
        Self { name, shape, data }
    }
}
