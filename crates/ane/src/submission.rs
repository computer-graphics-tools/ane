use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use crate::{Error, SubmissionResult, TensorData};

pub struct Submission {
    submission_result: Arc<SubmissionResult>,
    results: Box<[TensorData]>,
    consumed: bool,
}

impl Submission {
    pub fn new(submission_result: Arc<SubmissionResult>, results: Box<[TensorData]>) -> Self {
        Self {
            submission_result,
            results,
            consumed: false,
        }
    }

    pub fn results(&self) -> &[TensorData] {
        &self.results
    }

    pub fn is_finished(&self) -> bool {
        self.consumed || self.submission_result.is_finished()
    }

    pub fn wait(mut self) -> Result<Box<[TensorData]>, Error> {
        if !self.consumed {
            self.consumed = true;
            self.submission_result.wait()?;
        }
        Ok(std::mem::take(&mut self.results))
    }

    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        self.consumed || self.submission_result.wait_timeout(timeout)
    }
}

impl Future for Submission {
    type Output = Result<Box<[TensorData]>, Error>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        if !this.consumed {
            match this.submission_result.poll(context.waker()) {
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
