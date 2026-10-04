use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::thread::Builder;

use crate::io_surface::SurfaceLease;
use crate::{CompletionState, Error, Outcome, Request, SharedEvent};

pub struct Completion {
    state: Mutex<CompletionState>,
    pub event: SharedEvent,
    cleanup: mpsc::Sender<Request>,
}

impl Completion {
    pub fn new() -> Result<Self, Error> {
        static CLEANUP: OnceLock<mpsc::Sender<Request>> = OnceLock::new();
        let cleanup = if let Some(sender) = CLEANUP.get() {
            sender.clone()
        } else {
            let (sender, receiver) = mpsc::channel();
            Builder::new().name("ane-cleanup".into()).spawn(move || {
                for request in receiver {
                    objc2::rc::autoreleasepool(|_| drop(request));
                }
            })?;
            CLEANUP.get_or_init(|| sender).clone()
        };
        Ok(Self {
            state: Mutex::new(CompletionState::default()),
            event: SharedEvent::new()?,
            cleanup,
        })
    }

    pub fn begin(
        &self,
        request: Arc<Request>,
        leases: Vec<SurfaceLease>,
    ) -> Result<(u64, Arc<Outcome>), Error> {
        let mut state = self.state.lock().map_err(|_| Error::Synchronization)?;
        if state.request.is_some() {
            return Err(Error::BufferBusy);
        }
        let value = state
            .event_value
            .checked_add(1)
            .ok_or(Error::Synchronization)?;
        state.leases = leases;
        let outcome = Arc::new(Outcome::default());
        state.request = Some(request);
        state.outcome = Some(outcome.clone());
        state.event_value = value;
        Ok((value, outcome))
    }

    pub fn complete(&self, result: Result<(), Error>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(request) = state.request.take() else {
            return;
        };
        state.leases.clear();
        let outcome = state.outcome.take();
        drop(state);
        if let Some(request) = Arc::into_inner(request) {
            self.cleanup
                .send(request)
                .unwrap_or_else(|_| panic!("ANE cleanup worker stopped"));
        }
        if let Some(outcome) = outcome {
            outcome.finish(result);
        }
    }

    pub fn abort(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.leases.clear();
        let request = state.request.take();
        let outcome = state.outcome.take();
        drop(state);
        drop((request, outcome));
    }
}
