use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::thread::Builder;

use crate::completion_state::CompletionAction;
use crate::io_surface::SurfaceLease;
use crate::{CompletionState, Error, Request, SharedEvent};

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
        action: CompletionAction,
    ) -> Result<u64, Error> {
        let mut state = self.state.lock().map_err(|_| Error::Synchronization)?;
        if state.request.is_some() {
            return Err(Error::BufferBusy);
        }
        let value = state
            .event_value
            .checked_add(1)
            .ok_or(Error::Synchronization)?;
        state.leases = leases;
        state.request = Some(request);
        state.action = Some(action);
        state.event_value = value;
        Ok(value)
    }

    pub fn is_idle(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .request
            .is_none()
    }

    pub fn complete(&self, result: Result<(), Error>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(request) = state.request.take() else {
            return;
        };
        state.leases.clear();
        let action = state.action.take();
        drop(state);
        if let Some(request) = Arc::into_inner(request) {
            self.cleanup
                .send(request)
                .unwrap_or_else(|_| panic!("ANE cleanup worker stopped"));
        }
        // After the state is released, so a handler can submit the request again.
        match action {
            Some(CompletionAction::StoreResult(submission_result)) => {
                submission_result.finish(result)
            }
            Some(CompletionAction::Notify(handler)) => handler(result),
            None => {}
        }
    }

    pub fn abort(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.leases.clear();
        let request = state.request.take();
        let action = state.action.take();
        drop(state);
        drop((request, action));
    }
}
