use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use crate::{BoundRequest, Error, ExecutionDescriptor, SharedEvent};

/// GPU signals `ready`, ANE signals `done`, then GPU consumes the result.
/// Trips on one timeline must be ordered; overlapping trips need separate timelines.
pub struct Timeline {
    event: SharedEvent,
    next_value: AtomicU64,
}

/// Event values for one round trip.
#[derive(Clone, Copy, Debug)]
pub struct Trip {
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

    /// Blocks for exceptional cleanup; normal completion waits on the GPU queue.
    pub fn wait_until_signaled(&self, value: u64) {
        while !self
            .event
            .wait_until_signaled_value(value, Duration::from_secs(60))
        {}
    }

    /// Accepted requests signal `done` even on failure; their results are then invalid.
    /// Queueing errors signal nothing; the caller must signal `done` after `ready`.
    /// `on_failure` may run on this thread or an ANE thread; it must not panic or block ANE work.
    ///
    /// # Safety
    ///
    /// Reserve `trip` from this timeline and use its event values in increasing order.
    /// Write inputs before `ready`; access results after `done`. Follow
    /// [`BoundRequest::submit_unchecked`]'s surface-ordering requirements.
    pub unsafe fn submit(
        &self,
        trip: Trip,
        request: &BoundRequest,
        on_failure: impl FnOnce(&Error) + Send + 'static,
    ) -> Result<(), Error> {
        let descriptor = ExecutionDescriptor {
            wait_events: &[(&self.event, trip.ready)],
            signal_events: &[(&self.event, trip.done)],
        };
        let event = self.event.clone();
        // Shared so that a queueing error can take the callback back from the request.
        let on_failure = Arc::new(Mutex::new(Some(on_failure)));
        let queued = {
            let on_failure = on_failure.clone();
            // SAFETY: forwarded to the caller; the failure path signals `done` itself.
            unsafe {
                request.submit_unchecked(&descriptor, move |outcome| {
                    if let Err(error) = outcome {
                        notify(&on_failure, &error);
                        event.set_signaled_value(trip.done);
                    }
                })
            }
        };
        if let Err(error) = &queued {
            notify(&on_failure, error);
        }
        queued
    }
}

fn notify<F: FnOnce(&Error)>(slot: &Mutex<Option<F>>, error: &Error) {
    let Ok(mut slot) = slot.lock() else {
        return;
    };
    if let Some(on_failure) = slot.take() {
        on_failure(error);
    }
}
