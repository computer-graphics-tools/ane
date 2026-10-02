use objc2::runtime::Bool;

use crate::apple_neural_engine::{AneError, ensure_device_interfaces};

raw_class!(ANEDeviceInfo, c"_ANEDeviceInfo");

impl ANEDeviceInfo {
    pub fn has_ane() -> Result<bool, AneError> {
        ensure_device_interfaces()?;
        Ok(unsafe { raw_message!(Self::class(), c"hasANE"; Bool) }.as_bool())
    }

    pub fn engine_count() -> Result<u32, AneError> {
        ensure_device_interfaces()?;
        Ok(unsafe { raw_message!(Self::class(), c"numANEs"; u32) })
    }

    pub fn core_count() -> Result<u32, AneError> {
        ensure_device_interfaces()?;
        Ok(unsafe { raw_message!(Self::class(), c"numANECores"; u32) })
    }
}
