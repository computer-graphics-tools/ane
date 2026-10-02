use objc2::rc::Retained;
use objc2_foundation::NSArray;

use crate::apple_neural_engine::{ANESharedSignalEvent, ANESharedWaitEvent, AneError};

raw_class!(ANESharedEvents, c"_ANESharedEvents");

impl ANESharedEvents {
    pub fn new(
        signal_events: &[Retained<ANESharedSignalEvent>],
        wait_events: &[Retained<ANESharedWaitEvent>],
    ) -> Result<Retained<Self>, AneError> {
        let signals = NSArray::from_retained_slice(signal_events);
        let waits = NSArray::from_retained_slice(wait_events);
        unsafe {
            Retained::retain_autoreleased(
                raw_message!(Self::class(), c"sharedEventsWithSignalEvents:waitEvents:",
                &*signals => &NSArray<ANESharedSignalEvent>,
                &*waits => &NSArray<ANESharedWaitEvent>; *mut Self),
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
