use crate::{Error, TensorData, TensorSpec};

pub enum VariableData {
    Values(Box<[f32]>),
    Surface(TensorData),
}

impl VariableData {
    pub fn initialize(&self, spec: &TensorSpec) -> Result<TensorData, Error> {
        let data = match self {
            Self::Values(values) => {
                let data = spec.allocate()?;
                data.copy_from_f32(values)?;
                data
            }
            Self::Surface(data) => data.clone(),
        };
        spec.validate(&data)?;
        Ok(data)
    }
}
