use std::ffi::CString;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSData, NSDictionary, NSNumber, NSQualityOfService, NSString};

use crate::Error;
use crate::MilProgram;
use crate::ane_in_memory_model::ANEInMemoryModel;
use crate::ane_in_memory_model_descriptor::ANEInMemoryModelDescriptor;

static FRAMEWORK_INIT: Once = Once::new();
static FRAMEWORK_OK: AtomicBool = AtomicBool::new(false);

fn ensure_framework() -> Result<(), Error> {
    FRAMEWORK_INIT.call_once(|| {
        let Ok(path) = CString::new(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        ) else {
            return;
        };
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW) };
        if !handle.is_null() {
            FRAMEWORK_OK.store(true, Ordering::Release);
        }
    });
    if FRAMEWORK_OK.load(Ordering::Acquire) {
        Ok(())
    } else {
        Err(Error::FrameworkLoad)
    }
}

pub fn compile_model(
    program: &MilProgram,
    quality_of_service: NSQualityOfService,
) -> Result<Retained<ANEInMemoryModel>, Error> {
    ensure_framework()?;

    let mil_text = &program.text;
    let weight_bytes = &program.weights;
    let mil_data = NSData::with_bytes(mil_text.as_bytes());
    let weights_dict: Retained<NSDictionary<NSString, AnyObject>> = if weight_bytes.is_empty() {
        NSDictionary::new()
    } else {
        let weight_data = NSData::with_bytes(weight_bytes);
        let offset = NSNumber::new_u64(0);
        let entry: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
            &[&*NSString::from_str("offset"), &*NSString::from_str("data")],
            &[
                offset.as_ref() as &AnyObject,
                weight_data.as_ref() as &AnyObject,
            ],
        );
        let key = NSString::from_str("@model_path/weights/weight.bin");
        NSDictionary::from_slices(&[&*key], &[entry.as_ref() as &AnyObject])
    };

    let descriptor = ANEInMemoryModelDescriptor::new(&mil_data, Some(&weights_dict))
        .ok_or(Error::ModelCreation)?;

    let model = ANEInMemoryModel::with_descriptor(&descriptor).ok_or(Error::ModelCreation)?;

    if let Some(hex_id) = model.hex_string_identifier() {
        let model_dir = std::env::temp_dir().join(hex_id.to_string());
        std::fs::create_dir_all(&model_dir)?;
        std::fs::write(model_dir.join("model.mil"), mil_text.as_bytes())?;
        if !weight_bytes.is_empty() {
            let weights_dir = model_dir.join("weights");
            std::fs::create_dir_all(&weights_dir)?;
            std::fs::write(weights_dir.join("weight.bin"), weight_bytes)?;
        }
    }

    model.compile(quality_of_service)?;
    model.load(quality_of_service)?;
    Ok(model)
}
