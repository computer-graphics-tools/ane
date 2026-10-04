use objc2::rc::Retained;
use objc2::runtime::Bool;
use objc2_foundation::{NSDictionary, NSError, NSQualityOfService, NSString};

use crate::apple_neural_engine::{
    ANEInMemoryModelDescriptor, ANEModel, AneError, ensure_model_interfaces, qos_class, with_error,
};

raw_class!(ANEInMemoryModel, c"_ANEInMemoryModel");

impl ANEInMemoryModel {
    pub fn new(descriptor: &ANEInMemoryModelDescriptor) -> Result<Retained<Self>, AneError> {
        ensure_model_interfaces()?;
        unsafe {
            Retained::retain_autoreleased(
                raw_message!(Self::class(), c"inMemoryModelWithDescriptor:",
                descriptor => &ANEInMemoryModelDescriptor; *mut Self),
            )
        }
        .ok_or(AneError::ObjectCreation("in-memory model"))
    }
    pub fn identifier(&self) -> Result<String, AneError> {
        unsafe {
            Retained::retain_autoreleased(raw_message!(self, c"hexStringIdentifier"; *mut NSString))
        }
        .map(|identifier| identifier.to_string())
        .ok_or(AneError::ObjectCreation("model identifier"))
    }
    pub fn is_compiled(&self) -> bool {
        unsafe { raw_message!(self, c"compiledModelExists"; Bool) }.as_bool()
    }
    pub fn compile(&self, qos: NSQualityOfService) -> Result<(), AneError> {
        with_error(|error| unsafe {
            raw_message!(self, c"compileWithQoS:options:error:",
                qos_class(qos) => u32, &*NSDictionary::new() => &NSDictionary,
                error => *mut *mut NSError; Bool)
        })
    }
    pub fn load(&self, qos: NSQualityOfService) -> Result<(), AneError> {
        with_error(|error| unsafe {
            raw_message!(self, c"loadWithQoS:options:error:",
                qos_class(qos) => u32, &*NSDictionary::new() => &NSDictionary,
                error => *mut *mut NSError; Bool)
        })
    }
    pub fn unload(&self, qos: NSQualityOfService) -> Result<(), AneError> {
        with_error(|error| unsafe {
            raw_message!(self, c"unloadWithQoS:error:",
                qos_class(qos) => u32, error => *mut *mut NSError; Bool)
        })
    }
    pub fn purge_compiled_model(&self) {
        unsafe { raw_message!(self, c"purgeCompiledModel"; ()) }
    }
    pub fn model(&self) -> Result<Retained<ANEModel>, AneError> {
        unsafe { Retained::retain_autoreleased(raw_message!(self, c"model"; *mut ANEModel)) }
            .ok_or(AneError::ObjectCreation("loaded model"))
    }
}
