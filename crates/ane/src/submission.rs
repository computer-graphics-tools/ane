use std::sync::Arc;

use crate::completion::Completion;
use crate::request::Request;
use crate::{Error, Executable};

pub struct Submission {
    state: Arc<Completion>,
    request: Option<Request>,
    executable: Arc<Executable>,
}

impl Submission {
    pub fn new(executable: Arc<Executable>, state: Arc<Completion>, request: Request) -> Self {
        Self {
            state,
            request: Some(request),
            executable,
        }
    }

    pub fn executable(&self) -> &Executable {
        &self.executable
    }

    pub fn is_finished(&self) -> bool {
        self.state.0.lock().unwrap().is_some()
    }

    pub fn wait(mut self) -> Result<(), Error> {
        self.finish()
    }

    fn finish(&mut self) -> Result<(), Error> {
        let (slot, done) = &*self.state;
        let mut result = done
            .wait_while(slot.lock().unwrap(), |r| r.is_none())
            .unwrap();
        self.request = None;
        result.take().unwrap()
    }
}

impl Drop for Submission {
    fn drop(&mut self) {
        if self.request.is_some() {
            let _ = self.finish();
        }
    }
}
