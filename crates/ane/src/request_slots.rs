use std::sync::{Arc, Mutex};

use crate::{Error, ExecutionDescriptor, Request};

pub(crate) struct RequestSlots {
    slots: Mutex<Vec<Arc<Request>>>,
}

impl RequestSlots {
    pub(crate) fn new(request: Arc<Request>) -> Self {
        Self {
            slots: Mutex::new(vec![request]),
        }
    }

    /// Queueing errors fire neither callback nor signals.
    /// `on_complete` must be short, must not panic, and must not wait for ANE work.
    /// # Safety
    /// Order all conflicting CPU/GPU/ANE accesses with shared events; no leases are taken.
    /// Write inputs before wait events fire; read results after signals or the callback.
    pub unsafe fn submit(
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
        unsafe { slots[index].submit_callback(descriptor, Box::new(on_complete)) }
    }
}
