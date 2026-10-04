/// Coordinate mapping of `resize_bilinear`, as MIL's `sampling_mode` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeSamplingMode {
    /// `x_in = x_out · (in − 1) / (out − 1)`; a size-1 output samples the first pixel.
    StrictAlignCorners,
    /// Same mapping as `StrictAlignCorners` on the ANE.
    AlignCorners,
    /// `x_in = x_out · in / out`, clamped to the last pixel.
    Default,
    /// `x_in = (x_out + 0.5) · (in − 1) / out`.
    OffsetCorners,
    /// Half-pixel centres: `x_in = (x_out + 0.5) · in / out − 0.5`, clamped to the input, as
    /// `align_corners = false` in PyTorch.
    UnalignCorners,
}

impl ResizeSamplingMode {
    /// MIL name of the mode.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::StrictAlignCorners => "STRICT_ALIGN_CORNERS",
            Self::AlignCorners => "ALIGN_CORNERS",
            Self::Default => "DEFAULT",
            Self::OffsetCorners => "OFFSET_CORNERS",
            Self::UnalignCorners => "UNALIGN_CORNERS",
        }
    }
}
