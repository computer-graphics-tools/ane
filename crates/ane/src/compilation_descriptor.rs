use crate::{DataType, NSQualityOfService};

#[derive(Clone, Debug)]
pub struct CompilationDescriptor {
    /// Storage types of the target tensors: one for all targets or one per target. `None` stores
    /// Float16 results as Float32 and keeps every other type.
    pub output_types: Option<Vec<DataType>>,
    pub quality_of_service: NSQualityOfService,
}

impl Default for CompilationDescriptor {
    fn default() -> Self {
        Self {
            output_types: None,
            quality_of_service: NSQualityOfService::Default,
        }
    }
}
