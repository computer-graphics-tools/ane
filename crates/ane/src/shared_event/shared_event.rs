use std::time::Duration;

use obfstr::obfcstr;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Sel};

use crate::io_surface::SurfaceEvent;
use crate::shared_event::SharedEventError;

/// A shared timeline event for ANE and Metal.
#[derive(Clone, Debug)]
pub struct SharedEvent {
    event: Retained<SurfaceEvent>,
}

impl SharedEvent {
    pub fn new() -> Result<Self, SharedEventError> {
        Ok(Self {
            event: SurfaceEvent::new()?,
        })
    }

    /// Adopts a shared event; rejects wrappers without `eventPort`.
    ///
    /// # Safety
    ///
    /// `event` must implement `MTLSharedEvent` or `IOSurfaceSharedEvent`, with compatible
    /// signatures for the selectors checked below.
    pub unsafe fn from_object(event: Retained<AnyObject>) -> Result<Self, SharedEventError> {
        let class = event.class();
        for selector in [
            obfcstr!(c"signaledValue"),
            obfcstr!(c"setSignaledValue:"),
            obfcstr!(c"waitUntilSignaledValue:timeoutMS:"),
            obfcstr!(c"eventPort"),
        ] {
            if !class.responds_to(Sel::register(selector)) {
                return Err(SharedEventError::Unsupported(
                    selector.to_string_lossy().into_owned(),
                ));
            }
        }
        // SAFETY: the caller guarantees compatible selectors.
        Ok(Self {
            event: unsafe { Retained::cast_unchecked(event) },
        })
    }

    pub fn signaled_value(&self) -> u64 {
        self.event.signaled_value()
    }

    pub fn set_signaled_value(&self, value: u64) {
        self.event.set_signaled_value(value);
    }

    pub fn wait_until_signaled_value(&self, value: u64, timeout: Duration) -> bool {
        self.event.wait_until_signaled_value(value, timeout)
    }
}

pub fn io_surface_shared_event(event: &SharedEvent) -> &SurfaceEvent {
    &event.event
}

#[cfg(test)]
mod tests {
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2_foundation::NSObject;

    use super::*;

    #[test]
    fn rejects_objects_that_are_not_shared_events() {
        let object: Retained<AnyObject> = NSObject::new().into();
        let error = unsafe { SharedEvent::from_object(object) }.unwrap_err();
        assert!(matches!(error, SharedEventError::Unsupported(_)));
    }
}
