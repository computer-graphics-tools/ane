use std::time::Duration;

use objc2::rc::Retained;

use crate::io_surface::SurfaceEvent;
use crate::shared_event::SharedEventError;

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
