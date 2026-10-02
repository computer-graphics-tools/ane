use std::sync::Arc;

use crate::completion::Completion;
use crate::{Error, TensorData};

pub struct Submission {
    state: Arc<Completion>,
    results: Box<[TensorData]>,
    consumed: bool,
}

impl Submission {
    pub fn new(state: Arc<Completion>, results: Box<[TensorData]>) -> Self {
        Self {
            state,
            results,
            consumed: false,
        }
    }

    pub fn results(&self) -> &[TensorData] {
        &self.results
    }

    pub fn is_finished(&self) -> bool {
        self.consumed
            || self
                .state
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .result
                .is_some()
    }

    pub fn wait(mut self) -> Result<Box<[TensorData]>, Error> {
        if !self.consumed {
            self.consumed = true;
            self.state.wait()?;
        }
        Ok(std::mem::take(&mut self.results))
    }

    pub fn wait_timeout(&mut self, timeout: std::time::Duration) -> Result<bool, Error> {
        if self.consumed {
            return Ok(true);
        }
        let (mut state, _) = self
            .state
            .done
            .wait_timeout_while(
                self.state
                    .state
                    .lock()
                    .map_err(|_| Error::Synchronization)?,
                timeout,
                |state| state.result.is_none(),
            )
            .map_err(|_| Error::Synchronization)?;
        let Some(result) = state.result.take() else {
            return Ok(false);
        };
        self.consumed = true;
        result.map(|()| true)
    }
}

impl std::future::Future for Submission {
    type Output = Result<Box<[TensorData]>, Error>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.get_mut();
        if !this.consumed {
            let mut state = this.state.state.lock().unwrap_or_else(|e| e.into_inner());
            match state.result.take() {
                Some(Err(error)) => {
                    this.consumed = true;
                    return std::task::Poll::Ready(Err(error));
                }
                Some(Ok(())) => this.consumed = true,
                None => {
                    state.waker = Some(context.waker().clone());
                    return std::task::Poll::Pending;
                }
            }
        }
        std::task::Poll::Ready(Ok(std::mem::take(&mut this.results)))
    }
}
