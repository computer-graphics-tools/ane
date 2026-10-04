use std::fmt::{self, Display, Formatter};

macro_rules! names {
    ($($variant:ident => $name:literal),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Parameter { $($variant),* }
        impl Parameter {
            pub fn as_str(self) -> &'static str { match self { $(Self::$variant => $name),* } }
        }
    };
}

names! {
    A => "a",
    B => "b",
    X => "x",
    Y => "y",
    Input => "input",
    Weight => "weight",
    Bias => "bias",
    Cond => "cond",
    Query => "query",
    Key => "key",
    Value => "value",
    AttnMask => "attn_mask",
    Coordinates => "coordinates",
    Values => "values",
    Data => "data",
    Scale => "scale",
    Offset => "offset",
    Indices => "indices",
    Lut => "lut",
    LutScale => "lut_scale",
    LutOffset => "lut_offset",
    VectorAxis => "vector_axis",
    Mask => "mask",
    NonzeroData => "nonzero_data",
    DataMask => "data_mask",
    IndicesMask => "indices_mask",
    IndicesNonzeroData => "indices_nonzero_data",
    Alpha => "alpha",
    Beta => "beta",
    Epsilon => "epsilon",
    K => "k",
    Axis => "axis",
    Axes => "axes",
    KeepDims => "keep_dims",
    TransposeX => "transpose_x",
    TransposeY => "transpose_y",
    Dtype => "dtype",
    Shape => "shape",
    Perm => "perm",
    Begin => "begin",
    Size => "size",
    End => "end",
    Stride => "stride",
    BeginMask => "begin_mask",
    EndMask => "end_mask",
    SqueezeMask => "squeeze_mask",
    Interleave => "interleave",
    Reps => "reps",
    BlockSize => "block_size",
    UpscaleFactor => "upscale_factor",
    BlockShape => "block_shape",
    Crops => "crops",
    Pad => "pad",
    Mode => "mode",
    ConstantVal => "constant_val",
    Strides => "strides",
    Dilations => "dilations",
    Groups => "groups",
    PadType => "pad_type",
    OutputShape => "output_shape",
    KernelSizes => "kernel_sizes",
    CeilMode => "ceil_mode",
    ExcludePaddingFromAverage => "exclude_padding_from_average",
    BatchDims => "batch_dims",
    ValidateIndices => "validate_indices",
    Ascending => "ascending",
    Sort => "sort",
    ReturnIndices => "return_indices",
    OutputIndicesDtype => "output_indices_dtype",
    TargetSizeHeight => "target_size_height",
    TargetSizeWidth => "target_size_width",
    SamplingMode => "sampling_mode",
    PaddingMode => "padding_mode",
    PaddingValue => "padding_value",
    CoordinatesMode => "coordinates_mode",
    AlignCorners => "align_corners",
    ZeroPoint => "zero_point",
    OutputDtype => "output_dtype",
    Gamma => "gamma",
    Mean => "mean",
    Variance => "variance",
    Update => "update",
    Paddings => "paddings",
    SplitSizes => "split_sizes",
}

impl Display for Parameter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
