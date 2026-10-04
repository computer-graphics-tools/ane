use std::sync::Arc;

use objc2::{Message, rc::Retained};
use objc2_foundation::{NSError, NSString};

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
    #[error("{}", describe(.0))]
    NSError(#[from] Retained<NSError>),
}

/// The error's description followed by its `NSUnderlyingError` chain, where the ANE compiler
/// reports which layer it rejected.
fn describe(error: &NSError) -> String {
    let mut text = error.localizedDescription().to_string();
    let mut current = error.retain();
    let key = NSString::from_str("NSUnderlyingError");
    while let Some(underlying) = current
        .userInfo()
        .objectForKey(&key)
        .and_then(|value| value.downcast::<NSError>().ok())
    {
        let detail = underlying.localizedDescription().to_string();
        let detail = detail
            .split_once("err=(")
            .map_or(detail.as_str(), |(_, detail)| detail);
        let detail = detail.replace("\\n", " ").replace("\\\"", "\"");
        let detail = detail.trim().trim_end_matches(')').trim().trim_matches('"');
        text.push_str(": ");
        text.push_str(&detail.split_whitespace().collect::<Vec<_>>().join(" "));
        current = underlying;
    }
    text
}
