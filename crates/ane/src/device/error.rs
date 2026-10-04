use std::io;

use crate::AneError;

#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error(transparent)]
    Ane(#[from] AneError),
    #[error("sysctl query failed")]
    Sysctl(#[from] io::Error),
    #[error("device identity is not valid UTF-8")]
    InvalidEncoding,
}
