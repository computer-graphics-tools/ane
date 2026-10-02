use std::{sync::OnceLock, time::Duration};

use objc2::{rc::Retained, runtime::Bool};

use obfstr::obfcstr;

use crate::io_surface::IOSurfaceError;
use crate::unavailable_interface::ensure_interfaces;

raw_class!(SurfaceEvent, c"IOSurfaceSharedEvent");

unsafe impl Send for SurfaceEvent {}
unsafe impl Sync for SurfaceEvent {}

impl SurfaceEvent {
    pub fn signaled_value(&self) -> u64 {
        unsafe { raw_message!(self, c"signaledValue"; u64) }
    }

    pub fn set_signaled_value(&self, value: u64) {
        unsafe { raw_message!(self, c"setSignaledValue:", value => u64; ()) }
    }

    pub fn new() -> Result<Retained<Self>, IOSurfaceError> {
        static AVAILABILITY: OnceLock<Result<(), IOSurfaceError>> = OnceLock::new();
        AVAILABILITY
            .get_or_init(|| {
                Ok(ensure_interfaces(&[(
                    obfcstr!(c"IOSurfaceSharedEvent"),
                    &[obfcstr!(c"new")],
                    &[
                        obfcstr!(c"signaledValue"),
                        obfcstr!(c"setSignaledValue:"),
                        obfcstr!(c"waitUntilSignaledValue:timeoutMS:"),
                    ],
                )])?)
            })
            .clone()?;
        unsafe { Retained::from_raw(raw_message!(Self::class(), c"new"; *mut Self)) }.ok_or(
            IOSurfaceError::ObjectCreation(
                Self::class()
                    .name()
                    .to_str()
                    .expect("invalid runtime class name"),
            ),
        )
    }

    pub fn wait_until_signaled_value(&self, value: u64, timeout: Duration) -> bool {
        let milliseconds = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        unsafe {
            raw_message!(self, c"waitUntilSignaledValue:timeoutMS:",
                value => u64, milliseconds => u64; Bool)
        }
        .as_bool()
    }
}
