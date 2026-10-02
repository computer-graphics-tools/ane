use crate::{
    AneError, DeviceError, GraphError, IOSurfaceError, IrError, SharedEventError, TensorDataError,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("ANE program ({operations}) failed: {source}")]
    Compilation {
        operations: String,
        #[source]
        source: Box<Error>,
    },
    #[error("buffer is already borrowed or in use by ANE")]
    BufferBusy,
    #[error("buffer synchronization failed")]
    Synchronization,
    #[error("expected {expected} {kind}, got {actual}")]
    Count {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("tensor is not bound as an executable {0}")]
    Unbound(&'static str),
    #[error("a written buffer cannot alias another binding")]
    Alias,
    #[error("compilation cache limits must be positive")]
    CacheLimits,
    #[error("compiled source of {bytes} bytes exceeds the cache limit of {limit} bytes")]
    CacheOverflow { bytes: usize, limit: usize },
    #[error(transparent)]
    Ane(#[from] AneError),

    #[error(transparent)]
    IOSurface(#[from] IOSurfaceError),

    #[error(transparent)]
    TensorData(#[from] TensorDataError),

    #[error(transparent)]
    Ir(#[from] IrError),

    #[error(transparent)]
    Graph(#[from] GraphError),

    #[error(transparent)]
    Device(#[from] DeviceError),

    #[error(transparent)]
    SharedEvent(#[from] SharedEventError),

    #[error("compiled {field} of {symbol} differs from the graph")]
    Layout { symbol: String, field: &'static str },
    #[error("invalid compiled program metadata: {0}")]
    Metadata(&'static str),
    #[error(transparent)]
    PropertyList(#[from] plist::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub fn require(condition: bool, error: Error) -> Result<(), Error> {
    if condition { Ok(()) } else { Err(error) }
}
