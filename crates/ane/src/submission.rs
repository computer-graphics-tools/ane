use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use crate::{Error, Outcome, TensorData};

pub struct Submission {
    outcome: Arc<Outcome>,
    results: Box<[TensorData]>,
    consumed: bool,
}

impl Submission {
    pub fn new(outcome: Arc<Outcome>, results: Box<[TensorData]>) -> Self {
        Self {
            outcome,
            results,
            consumed: false,
        }
    }

    pub fn results(&self) -> &[TensorData] {
        &self.results
    }

    pub fn is_finished(&self) -> bool {
        self.consumed || self.outcome.is_finished()
    }

    pub fn wait(mut self) -> Result<Box<[TensorData]>, Error> {
        if !self.consumed {
            self.consumed = true;
            self.outcome.wait()?;
        }
        Ok(std::mem::take(&mut self.results))
    }

    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        self.consumed || self.outcome.wait_timeout(timeout)
    }
}

impl Future for Submission {
    type Output = Result<Box<[TensorData]>, Error>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if !this.consumed {
            match this.outcome.poll(context.waker()) {
                None => return Poll::Pending,
                Some(result) => {
                    this.consumed = true;
                    result?;
                }
            }
        }
        Poll::Ready(Ok(std::mem::take(&mut this.results)))
    }
}
