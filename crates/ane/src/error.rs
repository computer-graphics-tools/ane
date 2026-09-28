#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to load AppleNeuralEngine.framework")]
    FrameworkLoad,

    #[error("ANE framework failed: {0}")]
    Framework(#[from] objc2::rc::Retained<objc2_foundation::NSError>),

    #[error("ANE surface binding mismatch: {0}")]
    Binding(String),

    #[error("unsupported ANE compiler composition: {0}")]
    UnsupportedComposition(&'static str),

    #[error("ANE evaluation failed")]
    Evaluation,

    #[error("failed to create ANERequest")]
    RequestCreation,

    #[error("failed to wrap IOSurface for ANE")]
    SurfaceWrap,

    #[error("failed to create _ANEInMemoryModel")]
    ModelCreation,

    #[error(
        "placeholder \"{name}\" has spatial width {width}, \
         but the dense graph API requires at least {min} (pad the width dimension to {min} or larger)"
    )]
    SpatialWidthTooSmall {
        name: String,
        width: usize,
        min: usize,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
