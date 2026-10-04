use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSData, NSDictionary, NSNumber, NSString};

use crate::apple_neural_engine::{AneError, ensure_model_interfaces};
use crate::ir::MilProgram;

raw_class!(ANEInMemoryModelDescriptor, c"_ANEInMemoryModelDescriptor");

impl ANEInMemoryModelDescriptor {
    pub fn new(text: &NSData, weights: &NSData) -> Result<Retained<Self>, AneError> {
        ensure_model_interfaces()?;
        let weights: Retained<NSDictionary<NSString, AnyObject>> = if weights.is_empty() {
            NSDictionary::new()
        } else {
            let offset = NSNumber::new_u64(0);
            let entry: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
                &[&*NSString::from_str("offset"), &*NSString::from_str("data")],
                &[offset.as_ref(), weights.as_ref()],
            );
            NSDictionary::from_slices(
                &[&*NSString::from_str(MilProgram::WEIGHT_PATH)],
                &[entry.as_ref()],
            )
        };
        unsafe {
            Retained::retain_autoreleased(
                raw_message!(Self::class(), c"modelWithMILText:weights:optionsPlist:",
                text => &NSData,
                &*weights => &NSDictionary<NSString, AnyObject>,
                std::ptr::null::<NSDictionary>() => *const NSDictionary;
                *mut Self),
            )
        }
        .ok_or(AneError::ObjectCreation("MIL model descriptor"))
    }
}
