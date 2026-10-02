use std::sync::Arc;
use std::task::Waker;

use crate::io_surface::SurfaceLease;
use crate::{Error, request::Request};

#[derive(Default)]
pub struct CompletionState {
    pub request: Option<Arc<Request>>,
    pub leases: Vec<SurfaceLease>,
    pub result: Option<Result<(), Error>>,
    pub waker: Option<Waker>,
    pub event_value: u64,
}
