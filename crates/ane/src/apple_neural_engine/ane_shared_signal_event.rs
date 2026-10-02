use objc2::rc::Retained;

use crate::apple_neural_engine::AneError;
use crate::io_surface::SurfaceEvent;

raw_class!(ANESharedSignalEvent, c"_ANESharedSignalEvent");

impl ANESharedSignalEvent {
    pub fn new(value: u64, event: &SurfaceEvent) -> Result<Retained<Self>, AneError> {
        unsafe {
            Retained::retain_autoreleased(raw_message!(Self::class(), c"signalEventWithValue:symbolIndex:eventType:sharedEvent:",
                value => u64, 0 => u32, 0 => i64, event => &SurfaceEvent; *mut Self))
        }.ok_or(AneError::ObjectCreation(Self::class().name().to_str().expect("invalid runtime class name")))
    }
}
