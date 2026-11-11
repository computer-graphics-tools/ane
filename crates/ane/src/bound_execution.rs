use std::sync::Arc;

use crate::{Error, Request, Submission, TensorData};

pub struct BoundExecution {
    request: Arc<Request>,
    results: Box<[TensorData]>,
}

impl BoundExecution {
    pub(crate) fn new(request: Arc<Request>, results: Box<[TensorData]>) -> Self {
        Self { request, results }
    }

    pub fn submit(&self) -> Result<Submission, Error> {
        Ok(Submission::new(self.request.submit(None)?, self.results.clone()))
    }
}
