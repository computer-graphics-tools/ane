use objc2::rc::Retained;

use crate::apple_neural_engine::AneError;
use crate::io_surface::SurfaceEvent;

raw_class!(ANESharedWaitEvent, c"_ANESharedWaitEvent");

impl ANESharedWaitEvent {
    pub fn new(value: u64, event: &SurfaceEvent) -> Result<Retained<Self>, AneError> {
        unsafe {
            Retained::retain_autoreleased(
                raw_message!(Self::class(), c"waitEventWithValue:sharedEvent:",
                value => u64, event => &SurfaceEvent; *mut Self),
            )
        }
        .ok_or(AneError::ObjectCreation(
            Self::class()
                .name()
                .to_str()
                .expect("invalid runtime class name"),
        ))
    }
}
