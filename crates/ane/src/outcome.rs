use std::sync::{Condvar, Mutex, MutexGuard};
use std::task::Waker;
use std::time::Duration;

use crate::Error;

#[derive(Default)]
pub struct Outcome {
    result: Mutex<Option<Result<(), Error>>>,
    waker: Mutex<Option<Waker>>,
    done: Condvar,
}

impl Outcome {
    pub fn finish(&self, result: Result<(), Error>) {
        let mut slot = self.result();
        *slot = Some(result);
        let waker = self.waker.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(slot);
        self.done.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    pub fn is_finished(&self) -> bool {
        self.result().is_some()
    }

    pub fn wait(&self) -> Result<(), Error> {
        self.done
            .wait_while(self.result(), |slot| slot.is_none())
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .unwrap_or(Err(Error::Synchronization))
    }

    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let (slot, _) = self
            .done
            .wait_timeout_while(self.result(), timeout, |slot| slot.is_none())
            .unwrap_or_else(|e| e.into_inner());
        slot.is_some()
    }

    pub fn poll(&self, waker: &Waker) -> Option<Result<(), Error>> {
        let mut slot = self.result();
        let result = slot.take();
        if result.is_none() {
            *self.waker.lock().unwrap_or_else(|e| e.into_inner()) = Some(waker.clone());
        }
        result
    }

    fn result(&self) -> MutexGuard<'_, Option<Result<(), Error>>> {
        self.result.lock().unwrap_or_else(|e| e.into_inner())
    }
}
