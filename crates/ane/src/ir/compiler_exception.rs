#[derive(Clone, Debug, thiserror::Error)]
#[error("{message}")]
pub struct CompilerException {
    message: Box<str>,
}

impl From<objc2::rc::Retained<objc2::exception::Exception>> for CompilerException {
    fn from(exception: objc2::rc::Retained<objc2::exception::Exception>) -> Self {
        Self {
            message: exception.to_string().into(),
        }
    }
}
