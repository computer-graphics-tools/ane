use std::sync::{Arc, Mutex};

use crate::{Error, ExecutionDescriptor, Request, Submission, TensorData};

pub struct BoundExecution {
    request: Arc<Request>,
    results: Box<[TensorData]>,
    spare_requests: Mutex<Vec<Arc<Request>>>,
}

impl BoundExecution {
    pub(crate) fn new(request: Arc<Request>, results: Box<[TensorData]>) -> Self {
        Self {
            request,
            results,
            spare_requests: Mutex::new(Vec::new()),
        }
    }

    pub fn submit(&self) -> Result<Submission, Error> {
        Ok(Submission::new(self.request.submit(None)?, self.results.clone()))
    }

    /// Queueing errors fire neither handler nor signals.
    /// `on_complete` must not panic or wait for ANE work.
    /// # Safety
    /// Order conflicting surface accesses with events; no leases are taken.
    /// Write inputs before wait events fire; read results after signals or the handler.
    pub(crate) unsafe fn submit_ordered(
        &self,
        descriptor: &ExecutionDescriptor<'_>,
        on_complete: impl FnOnce(Result<(), Error>) + Send + 'static,
    ) -> Result<(), Error> {
        let mut requests = self
            .spare_requests
            .lock()
            .map_err(|_| Error::Synchronization)?;
        let request = if self.request.is_idle() {
            &self.request
        } else {
            let index = match requests.iter().position(|request| request.is_idle()) {
                Some(index) => index,
                None => {
                    requests.push(self.request.duplicate()?);
                    requests.len() - 1
                }
            };
            &requests[index]
        };
        unsafe { request.submit_callback(descriptor, Box::new(on_complete)) }
    }
}
