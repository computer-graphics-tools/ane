use crate::apple_neural_engine::{ANEModel, ANERequest, AneError, ensure_model_interfaces};
use objc2::{rc::Retained, runtime::Bool};
use objc2_foundation::{NSDictionary, NSError, NSQualityOfService};

raw_class!(ANEClient, c"_ANEClient");

impl ANEClient {
    pub fn shared() -> Result<Retained<Self>, AneError> {
        ensure_model_interfaces()?;
        unsafe {
            Retained::retain_autoreleased(raw_message!(Self::class(),c"sharedConnection";*mut Self))
        }
        .ok_or(AneError::ObjectCreation("ANE connection"))
    }
    pub fn load(
        &self,
        model: &ANEModel,
        options: &NSDictionary,
        qos: NSQualityOfService,
    ) -> Result<(), AneError> {
        with_error(|error| unsafe {
            raw_message!(self,c"loadModel:options:qos:error:",model => &ANEModel,options => &NSDictionary,qos_class(qos) => u32,error => *mut *mut NSError; Bool)
        })
    }
    pub fn evaluate(
        &self,
        model: &ANEModel,
        qos: NSQualityOfService,
        request: &ANERequest,
    ) -> Result<(), AneError> {
        with_error(|error| unsafe {
            raw_message!(self,c"evaluateWithModel:options:request:qos:error:",model => &ANEModel,&*NSDictionary::new() => &NSDictionary,request => &ANERequest,qos_class(qos) => u32,error => *mut *mut NSError; Bool)
        })
    }
    pub fn unload(&self, model: &ANEModel, qos: NSQualityOfService) -> Result<(), AneError> {
        with_error(|error| unsafe {
            raw_message!(self,c"unloadModel:options:qos:error:",model => &ANEModel,&*NSDictionary::new() => &NSDictionary,qos_class(qos) => u32,error => *mut *mut NSError; Bool)
        })
    }
    pub fn map(&self, model: &ANEModel, request: &ANERequest) -> Result<(), AneError> {
        with_error(|error| unsafe {
            raw_message!(self,c"mapIOSurfacesWithModel:request:cacheInference:error:",model => &ANEModel,request => &ANERequest,Bool::YES => Bool,error => *mut *mut NSError; Bool)
        })
    }
    pub fn unmap(&self, model: &ANEModel, request: &ANERequest) {
        unsafe {
            raw_message!(self,c"unmapIOSurfacesWithModel:request:",model => &ANEModel,request => &ANERequest; ())
        }
    }
}
fn with_error(call: impl FnOnce(*mut *mut NSError) -> Bool) -> Result<(), AneError> {
    let mut error = std::ptr::null_mut();
    if call(&mut error).as_bool() {
        Ok(())
    } else {
        Err(unsafe { Retained::retain(error) }
            .map(AneError::from)
            .unwrap_or(AneError::EvaluationFailed))
    }
}
fn qos_class(qos: NSQualityOfService) -> u32 {
    if qos == NSQualityOfService::Default {
        libc::qos_class_t::QOS_CLASS_DEFAULT as u32
    } else {
        qos.0 as u32
    }
}
