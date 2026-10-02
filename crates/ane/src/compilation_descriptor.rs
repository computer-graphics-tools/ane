use crate::{DataType, NSQualityOfService};

#[derive(Clone, Debug)]
pub struct CompilationDescriptor {
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
