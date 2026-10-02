use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};

use crate::{Error, SharedEvent, completion_state::CompletionState, request::Request};

pub struct Completion {
    pub state: Mutex<CompletionState>,
    pub done: Condvar,
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
            std::thread::Builder::new()
                .name("ane-cleanup".into())
                .spawn(move || {
                    for request in receiver {
                        objc2::rc::autoreleasepool(|_| drop(request));
                    }
                })?;
            CLEANUP.get_or_init(|| sender).clone()
        };
        Ok(Self {
            state: Mutex::new(CompletionState::default()),
            done: Condvar::new(),
            event: SharedEvent::new()?,
            cleanup,
        })
    }

    pub fn begin(&self, request: Arc<Request>) -> Result<u64, Error> {
        let mut state = self.state.lock().map_err(|_| Error::Synchronization)?;
        if state.request.is_some() {
            return Err(Error::BufferBusy);
        }
        let value = state
            .event_value
            .checked_add(1)
            .ok_or(Error::Synchronization)?;
        request.acquire_into(&mut state.leases)?;
        state.result = None;
        state.waker = None;
        state.request = Some(request);
        state.event_value = value;
        Ok(value)
    }

    pub fn complete(&self, result: Result<(), Error>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some(request) = state.request.take() else {
            return;
        };
        state.leases.clear();
        state.result = Some(result);
        let waker = state.waker.take();
        drop(state);
        if let Some(request) = Arc::into_inner(request) {
            self.cleanup
                .send(request)
                .unwrap_or_else(|_| panic!("ANE cleanup worker stopped"));
        }
        self.done.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    pub fn wait(&self) -> Result<(), Error> {
        let mut state = self
            .done
            .wait_while(
                self.state.lock().unwrap_or_else(|e| e.into_inner()),
                |state| state.result.is_none(),
            )
            .unwrap_or_else(|e| e.into_inner());
        state.result.take().unwrap()
    }

    pub fn abort(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.leases.clear();
        let request = state.request.take();
        state.result = None;
        drop(state);
        drop(request);
    }
}
