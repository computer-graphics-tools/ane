use std::sync::{Arc, Mutex};

use crate::{Error, ExecutionDescriptor, Request};

/// Reusable surface bindings; overlapping submissions use separate request slots.
pub struct BoundRequest {
    slots: Mutex<Vec<Arc<Request>>>,
}

impl BoundRequest {
    pub(crate) fn new(request: Arc<Request>) -> Self {
        Self {
            slots: Mutex::new(vec![request]),
        }
    }

    /// Calls `on_complete` after completion. On queueing error, no callback or signals fire.
    /// The callback must be short, must not panic, and must not wait for ANE work.
    ///
    /// # Safety
    ///
    /// No surface leases are acquired. Order all conflicting CPU/GPU/ANE accesses, including
    /// overlapping submissions, with shared events. Write inputs before wait events fire;
    /// read results only after signal events fire or the callback runs.
    pub unsafe fn submit_unchecked(
        &self,
        descriptor: &ExecutionDescriptor<'_>,
        on_complete: impl FnOnce(Result<(), Error>) + Send + 'static,
    ) -> Result<(), Error> {
        let mut slots = self.slots.lock().map_err(|_| Error::Synchronization)?;
        let index = match slots.iter().position(|request| request.is_idle()) {
            Some(index) => index,
            None => {
                let copy = slots[0].duplicate()?;
                slots.push(copy);
                slots.len() - 1
            }
        };
        // SAFETY: forwarded to the caller; the request is idle and the lock is held, so no other
        // submission can take it before this one begins.
        unsafe { slots[index].submit_callback(descriptor, Box::new(on_complete)) }
    }
}
