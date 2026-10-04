use crate::apple_neural_engine::{AneError, ModelAttributes};
use objc2::rc::Retained;
use objc2_foundation::NSDictionary;

raw_class!(ANEModel);

impl ANEModel {
    pub fn model_attributes(&self) -> Result<ModelAttributes, AneError> {
        let attributes = unsafe {
            Retained::retain_autoreleased(raw_message!(self,c"modelAttributes";*mut NSDictionary))
        }
        .ok_or(AneError::ObjectCreation("model metadata"))?;
        ModelAttributes::new(&attributes)
    }
}
