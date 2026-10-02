use crate::io_surface::IOSurfaceError;

#[derive(Clone, Debug, thiserror::Error)]
pub enum SharedEventError {
    #[error(transparent)]
    IOSurface(#[from] IOSurfaceError),
}
