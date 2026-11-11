use crate::io_surface::IOSurfaceError;

#[derive(Clone, Debug, thiserror::Error)]
pub enum SharedEventError {
    #[error(transparent)]
    IOSurface(#[from] IOSurfaceError),
    #[error(
        "object cannot be used as a shared event: it does not respond to `{0}` (is Metal API validation on?)"
    )]
    Unsupported(String),
}
