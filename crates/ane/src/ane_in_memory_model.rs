use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, NSObject};
use objc2::{ClassType, extern_class, extern_conformance, msg_send};
use objc2_foundation::{NSDictionary, NSError, NSObjectProtocol, NSQualityOfService, NSString};

use crate::ane_in_memory_model_descriptor::ANEInMemoryModelDescriptor;
use crate::ane_request::ANERequest;

extern_class!(
    #[unsafe(super(NSObject))]
    #[name = "_ANEInMemoryModel"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct ANEInMemoryModel;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for ANEInMemoryModel {}
);

fn empty_options() -> Retained<NSDictionary<NSString, AnyObject>> {
    NSDictionary::new()
}

impl ANEInMemoryModel {
    pub fn with_descriptor(
        descriptor: &ANEInMemoryModelDescriptor,
    ) -> Option<Retained<ANEInMemoryModel>> {
        unsafe { msg_send![Self::class(), inMemoryModelWithDescriptor: descriptor] }
    }

    pub fn model_attributes(&self) -> Retained<NSDictionary<NSString, AnyObject>> {
        unsafe { msg_send![self, modelAttributes] }
    }

    pub fn hex_string_identifier(&self) -> Option<Retained<NSString>> {
        unsafe { msg_send![self, hexStringIdentifier] }
    }

    pub fn compile(&self, qos: NSQualityOfService) -> Result<(), Retained<NSError>> {
        let opts = empty_options();
        unsafe { msg_send![self, compileWithQoS: qos.0 as u32, options: &*opts, error: _] }
    }

    pub fn load(&self, qos: NSQualityOfService) -> Result<(), Retained<NSError>> {
        let opts = empty_options();
        unsafe { msg_send![self, loadWithQoS: qos.0 as u32, options: &*opts, error: _] }
    }

    pub fn evaluate(
        &self,
        qos: NSQualityOfService,
        request: &ANERequest,
    ) -> Result<(), Retained<NSError>> {
        let opts = empty_options();
        unsafe {
            msg_send![self, evaluateWithQoS: qos.0 as u32, options: &*opts, request: request, error: _]
        }
    }

    pub fn unload(&self, qos: NSQualityOfService) {
        let mut err: *mut NSError = std::ptr::null_mut();
        let _: Bool = unsafe { msg_send![self, unloadWithQoS: qos.0 as u32, error: &mut err] };
    }

    pub fn map(&self, request: &ANERequest) -> Result<(), Retained<NSError>> {
        unsafe {
            msg_send![self, mapIOSurfacesWithRequest: request, cacheInference: true, error: _]
        }
    }

    pub fn unmap(&self, request: &ANERequest) {
        unsafe {
            let _: () = msg_send![self, unmapIOSurfacesWithRequest: request];
        }
    }
}
