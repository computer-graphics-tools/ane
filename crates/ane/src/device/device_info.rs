use crate::apple_neural_engine::ANEDeviceInfo;
use crate::device::DeviceError;
use std::ffi::CStr;

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub os_build: String,
    pub has_ane: bool,
    pub engine_count: u32,
    pub core_count: u32,
}

impl DeviceInfo {
    pub fn current() -> Result<Self, DeviceError> {
        Ok(Self {
            name: system_string(if cfg!(target_os = "ios") {
                c"hw.machine"
            } else {
                c"machdep.cpu.brand_string"
            })?,
            os_build: system_string(c"kern.osversion")?,
            has_ane: ANEDeviceInfo::has_ane()?,
            engine_count: ANEDeviceInfo::engine_count()?,
            core_count: ANEDeviceInfo::core_count()?,
        })
    }
}

fn system_string(name: &CStr) -> Result<String, DeviceError> {
    let mut length = 0;
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            std::ptr::null_mut(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let mut bytes = vec![0u8; length];
    let result = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            bytes.as_mut_ptr().cast(),
            &mut length,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    bytes.truncate(length);
    if bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8(bytes).map_err(|_| DeviceError::InvalidEncoding)
}
