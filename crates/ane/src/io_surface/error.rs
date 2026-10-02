use crate::UnavailableInterface;

#[derive(Clone, Debug, thiserror::Error)]
pub enum IOSurfaceError {
    #[error(transparent)]
    Unavailable(#[from] UnavailableInterface),
    #[error("{0} returned no object")]
    ObjectCreation(&'static str),
    #[error("invalid IOSurface allocation size {0}")]
    InvalidSize(usize),
    #[error("IOSurface allocation failed")]
    Allocation,
    #[error("IOSurface lock failed with code {0}")]
    Lock(i32),
    #[error("IOSurface unlock failed with code {0}")]
    Unlock(i32),
    #[error("{length}-byte access exceeds the {allocation}-byte IOSurface allocation")]
    OutOfBounds { length: usize, allocation: usize },
    #[error("IOSurface is already borrowed or in use by ANE")]
    Busy,
    #[error("IOSurface access registry is poisoned")]
    Synchronization,
}
