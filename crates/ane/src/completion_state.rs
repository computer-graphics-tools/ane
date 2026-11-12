use std::sync::Arc;

use crate::io_surface::SurfaceLease;
use crate::{Outcome, Request};

pub type Callback = Box<dyn FnOnce(Result<(), crate::Error>) + Send>;

pub enum Sink {
    Outcome(Arc<Outcome>),
    Callback(Callback),
}

#[derive(Default)]
pub struct CompletionState {
    pub request: Option<Arc<Request>>,
    pub sink: Option<Sink>,
    pub leases: Vec<SurfaceLease>,
    pub event_value: u64,
}
