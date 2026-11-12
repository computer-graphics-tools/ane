use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::{BoundExecution, Error, ExecutionDescriptor, SharedEvent};

pub(crate) struct Timeline {
    event: SharedEvent,
    next_value: AtomicU64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Trip {
    pub ready: u64,
    pub done: u64,
}

impl Timeline {
    pub fn new(event: SharedEvent) -> Self {
        Self {
            event,
            next_value: AtomicU64::new(0),
        }
    }

    pub fn reserve(&self) -> Trip {
        let ready = self.next_value.fetch_add(2, Ordering::AcqRel) + 1;
        Trip {
            ready,
            done: ready + 1,
        }
    }

    pub fn wait_until_signaled(&self, value: u64) {
        while !self
            .event
            .wait_until_signaled_value(value, Duration::from_secs(60))
        {}
    }

    /// Failed executions still signal `done`; their results are invalid.
    /// Queueing errors do not signal; caller must signal `done` after `ready`.
    /// `on_failure` must not panic or wait for ANE work.
    /// # Safety
    /// Use a trip reserved here. Trips on one timeline must not overlap; keep event values ordered.
    /// Write inputs before `ready`; access results after `done`. See [`BoundExecution::submit_ordered`].
    pub unsafe fn submit(
        &self,
        trip: Trip,
        execution: &BoundExecution,
        on_failure: impl FnOnce(&Error) + Send + 'static,
    ) -> Result<(), Error> {
        let descriptor = ExecutionDescriptor {
            wait_events: &[(&self.event, trip.ready)],
            signal_events: &[(&self.event, trip.done)],
        };
        let event = self.event.clone();
        // SAFETY: forwarded to the caller; the failure path signals `done` itself.
        unsafe {
            execution.submit_ordered(&descriptor, move |result| {
                if let Err(error) = result {
                    on_failure(&error);
                    event.set_signaled_value(trip.done);
                }
            })
        }
    }
}
