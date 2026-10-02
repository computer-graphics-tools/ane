use crate::apple_neural_engine::{AneError, ModelAttributes, ensure_model_interfaces};
use objc2::rc::Retained;
use objc2_foundation::{NSDictionary, NSString, NSURL};
use std::path::Path;

raw_class!(ANEModel, c"_ANEModel");

impl ANEModel {
    pub fn new(directory: &Path, key: &str) -> Result<Retained<Self>, AneError> {
        ensure_model_interfaces()?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(&directory.to_string_lossy()));
        unsafe {Retained::retain_autoreleased(raw_message!(Self::class(),c"modelAtURL:key:",&*url => &NSURL,&*NSString::from_str(key) => &NSString; *mut Self))}
            .ok_or(AneError::ObjectCreation("compiled model"))
    }
    pub fn model_attributes(&self) -> Result<ModelAttributes, AneError> {
        let attributes = unsafe {
            Retained::retain_autoreleased(raw_message!(self,c"modelAttributes";*mut NSDictionary))
        }
        .ok_or(AneError::ObjectCreation("model metadata"))?;
        ModelAttributes::new(&attributes)
    }
}
