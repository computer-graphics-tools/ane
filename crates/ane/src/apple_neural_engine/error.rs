use std::sync::Arc;

use objc2::rc::Retained;
use objc2_foundation::NSError;

use crate::UnavailableInterface;

#[derive(Clone, Debug, thiserror::Error)]
pub enum AneError {
    #[error("ANE framework was not found")]
    FrameworkNotFound,
    #[error(transparent)]
    Unavailable(#[from] UnavailableInterface),
    #[error("{0} returned no object")]
    ObjectCreation(&'static str),
    #[error("invalid compiled model metadata")]
    Metadata(#[source] Arc<plist::Error>),
    #[error("ANE operation failed without an error")]
    EvaluationFailed,
    #[error(transparent)]
    NSError(#[from] Retained<NSError>),
}
