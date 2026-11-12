use std::sync::{Arc, Mutex};

use crate::{Error, ExecutionDescriptor, Request, Submission, TensorData};

pub struct BoundExecution {
    requests: Mutex<Vec<Arc<Request>>>,
    results: Box<[TensorData]>,
}

impl BoundExecution {
    pub(crate) fn new(request: Arc<Request>, results: Box<[TensorData]>) -> Self {
        Self {
            requests: Mutex::new(vec![request]),
            results,
        }
    }

    pub fn submit(&self) -> Result<Submission, Error> {
        let requests = self.requests.lock().map_err(|_| Error::Synchronization)?;
        Ok(Submission::new(requests[0].submit(None)?, self.results.clone()))
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
        let mut requests = self.requests.lock().map_err(|_| Error::Synchronization)?;
        let index = match requests.iter().position(|request| request.is_idle()) {
            Some(index) => index,
            None => {
                let request = requests[0].duplicate()?;
                requests.push(request);
                requests.len() - 1
            }
        };
        unsafe { requests[index].submit_callback(descriptor, Box::new(on_complete)) }
    }
}
