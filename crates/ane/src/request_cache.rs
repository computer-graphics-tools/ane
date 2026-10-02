use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::Error;
use crate::request::Request;

#[derive(Default)]
pub struct RequestCache {
    requests: Mutex<VecDeque<Arc<Request>>>,
}

impl RequestCache {
    const CAPACITY: usize = 64;

    pub fn get_or_insert(
        &self,
        bindings: &[u32],
        create: impl FnOnce() -> Result<Request, Error>,
    ) -> Result<Arc<Request>, Error> {
        let mut requests = self.requests.lock().map_err(|_| Error::Synchronization)?;
        if let Some(index) = requests.iter().position(|r| r.bindings() == bindings) {
            let request = requests.remove(index).unwrap();
            requests.push_back(request.clone());
            return Ok(request);
        }
        let request = Arc::new(create()?);
        if requests.len() == Self::CAPACITY {
            requests.pop_front();
        }
        requests.push_back(request.clone());
        Ok(request)
    }
}
