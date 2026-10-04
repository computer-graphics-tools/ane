use std::sync::Arc;

use crate::io_surface::SurfaceLease;
use crate::{Outcome, Request};

#[derive(Default)]
pub struct CompletionState {
    pub request: Option<Arc<Request>>,
    pub outcome: Option<Arc<Outcome>>,
    pub leases: Vec<SurfaceLease>,
    pub event_value: u64,
}
