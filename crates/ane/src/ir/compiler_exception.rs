use std::panic::AssertUnwindSafe;

use objc2::exception::{Exception, catch};
use objc2::rc::Retained;

use crate::ir::IrError;

#[derive(Clone, Debug, thiserror::Error)]
#[error("{message}")]
pub struct CompilerException {
    message: Box<str>,
}

impl From<Retained<Exception>> for CompilerException {
    fn from(exception: Retained<Exception>) -> Self {
        Self {
            message: exception.to_string().into(),
        }
    }
}

pub fn catch_native<T>(f: impl FnOnce() -> T) -> Result<T, IrError> {
    catch(AssertUnwindSafe(f)).map_err(|exception| {
        exception
            .map(|exception| IrError::from(CompilerException::from(exception)))
            .unwrap_or(IrError::CompilerFailed)
    })
}
