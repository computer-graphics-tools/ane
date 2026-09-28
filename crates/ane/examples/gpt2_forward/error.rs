#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Ane(#[from] ane::Error),
    #[error(transparent)]
    Weights(#[from] safetensors::SafeTensorError),
    #[error(transparent)]
    Download(#[from] hf_hub::api::sync::ApiError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Config(#[from] serde_json::Error),
    #[cfg(target_os = "ios")]
    #[error("app cache directory unavailable")]
    CacheDirectory,
    #[error("invalid GPT-2 input: {0}")]
    Input(&'static str),
}
