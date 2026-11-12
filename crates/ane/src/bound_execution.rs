use std::sync::{Arc, Mutex};

use crate::{Error, ExecutionDescriptor, Procedure, Request, Submission, TensorData};

pub struct BoundExecution {
    procedure: Arc<Procedure>,
    inputs: Box<[TensorData]>,
    requests: Mutex<Vec<Arc<Request>>>,
    results: Box<[TensorData]>,
}

impl BoundExecution {
    pub fn new(
        procedure: Arc<Procedure>,
        inputs: &[&TensorData],
        results: &[&TensorData],
    ) -> Result<Self, Error> {
        let request = procedure.request(inputs, results)?;
        Ok(Self {
            procedure,
            inputs: inputs.iter().map(|data| (*data).clone()).collect(),
            requests: Mutex::new(vec![request]),
            results: results.iter().map(|data| (*data).clone()).collect(),
        })
    }

    pub fn submit(&self) -> Result<Submission, Error> {
        let requests = self.requests.lock().map_err(|_| Error::Synchronization)?;
        Ok(Submission::new(requests[0].submit(None)?, self.results.clone()))
    }

    /// # Safety
    /// Synchronize access to the bound surfaces with the descriptor's events.
    pub unsafe fn submit_ordered(
        &self,
        descriptor: &ExecutionDescriptor<'_>,
        on_complete: impl FnOnce(Result<(), Error>) + Send + 'static,
    ) -> Result<(), Error> {
        let mut requests = self.requests.lock().map_err(|_| Error::Synchronization)?;
        let index = match requests.iter().position(|request| request.is_idle()) {
            Some(index) => index,
            None => {
                let inputs: Vec<_> = self.inputs.iter().collect();
                let results: Vec<_> = self.results.iter().collect();
                let request = self.procedure.request(&inputs, &results)?;
                requests.push(request);
                requests.len() - 1
            }
        };
        requests[index].submit_callback(descriptor, Box::new(on_complete))
    }
}
