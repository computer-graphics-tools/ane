use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::{BoundExecution, Error, ExecutionDescriptor, SharedEvent};

pub struct EventSequence {
    event: SharedEvent,
    next_value: AtomicU64,
}

#[derive(Clone, Copy, Debug)]
pub struct EventValues {
    pub inputs_ready: u64,
    pub completed: u64,
}

impl EventSequence {
    pub fn new(event: SharedEvent) -> Self {
        Self {
            event,
            next_value: AtomicU64::new(0),
        }
    }

    pub fn reserve(&self) -> EventValues {
        let inputs_ready = self.next_value.fetch_add(2, Ordering::AcqRel) + 1;
        EventValues {
            inputs_ready,
            completed: inputs_ready + 1,
        }
    }

    pub fn wait_until_signaled(&self, value: u64) {
        while !self
            .event
            .wait_until_signaled_value(value, Duration::from_secs(60))
        {}
    }

    /// # Safety
    /// Use this sequence's values in order: finish writes before `inputs_ready`,
    /// and read outputs after successful completion.
    /// On queueing failure, signal `completed` after `inputs_ready`.
    /// `on_failure` must not panic or wait for ANE work.
    pub unsafe fn submit(
        &self,
        values: EventValues,
        execution: &BoundExecution,
        on_failure: impl FnOnce(&Error) + Send + 'static,
    ) -> Result<(), Error> {
        let descriptor = ExecutionDescriptor {
            wait_events: &[(&self.event, values.inputs_ready)],
            signal_events: &[(&self.event, values.completed)],
        };
        let event = self.event.clone();
        // SAFETY: the caller guarantees surface access follows event ordering.
        unsafe {
            execution.submit_ordered(&descriptor, move |result| {
                if let Err(error) = result {
                    on_failure(&error);
                    event.set_signaled_value(values.completed);
                }
            })
        }
    }
}
