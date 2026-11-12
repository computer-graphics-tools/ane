use std::sync::Arc;

use crate::io_surface::SurfaceLease;
use crate::{Request, SubmissionResult};

pub type CompletionHandler = Box<dyn FnOnce(Result<(), crate::Error>) + Send>;

pub enum CompletionAction {
    StoreResult(Arc<SubmissionResult>),
    Notify(CompletionHandler),
}

#[derive(Default)]
pub struct CompletionState {
    pub request: Option<Arc<Request>>,
    pub action: Option<CompletionAction>,
    pub leases: Vec<SurfaceLease>,
    pub event_value: u64,
}
