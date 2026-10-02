use crate::ir::{ArgumentType, Parameter};

macro_rules! operators {
    ($( [$($variant:ident => $name:literal),+ $(,)?] ($($p:ident : $t:ident),* $(; $($optional:ident : $ot:ident),*)?); )*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Operator { $($($variant),+),* }
        impl Operator {
            pub fn as_str(self) -> &'static str { match self { $($(Self::$variant => $name),+),* } }
            pub fn schema(self) -> &'static [(Parameter, ArgumentType, bool)] {
                match self { $($(Self::$variant)|+ => &[$((Parameter::$p, ArgumentType::$t, true)),* $(,$((Parameter::$optional, ArgumentType::$ot, false)),*)?]),* }
            }
        }
    };
}

operators! {
    [
        Abs => "abs",
        Atan => "atan",
        Ceil => "ceil",
        Cos => "cos",
        Erf => "erf",
        Exp => "exp",
        Exp2 => "exp2",
        Floor => "floor",
        Relu => "relu",
        Round => "round",
        Sigmoid => "sigmoid",
        Sign => "sign",
        Silu => "silu",
        Sin => "sin",
        Softplus => "softplus",
        Softsign => "softsign",
        Sqrt => "sqrt",
        Tanh => "tanh",
    ] (
        X: FloatTensor
    );
    [Log => "log", Inverse => "inverse", Rsqrt => "rsqrt"] (X: FloatTensor, Epsilon: Fp16);
    [
        Add => "add",
        Sub => "sub",
        Mul => "mul",
        RealDiv => "real_div",
        Pow => "pow",
        Maximum => "maximum",
        Minimum => "minimum",
    ] (
        X: Scalar,
        Y: Scalar
    );
    [
        Equal => "equal",
        NotEqual => "not_equal",
        Less => "less",
        LessEqual => "less_equal",
        Greater => "greater",
        GreaterEqual => "greater_equal",
    ] (
        X: Tensor,
        Y: Tensor
    );
    [LogicalNot => "logical_not"] (X: BoolTensor);
    [LogicalAnd => "logical_and", LogicalOr => "logical_or"] (X: BoolTensor, Y: BoolTensor);
    [Select => "select"] (Cond: BoolTensor, A: Tensor, B: Tensor);
    [Matmul => "matmul"] (X: FloatTensor, Y: FloatTensor, TransposeX: Bool, TransposeY: Bool);
    [ScaledDotProductAttention => "scaled_dot_product_attention"] (
        Query: FloatTensor,
        Key: FloatTensor,
        Value: FloatTensor;
        AttnMask: FloatTensor
    );
    [LeakyRelu => "leaky_relu", Elu => "elu"] (X: FloatTensor, Alpha: Fp32);
    [LinearActivation => "linear_activation"] (X: FloatTensor, Alpha: Fp32, Beta: Fp32);
    [Threshold => "threshold", ThresholdedRelu => "thresholded_relu"] (X: FloatTensor, Alpha: Fp16);
    [Gelu => "gelu"] (X: FloatTensor, Mode: String);
    [Softmax => "softmax"] (X: FloatTensor, Axis: Int32);
    [LocalResponseNorm => "local_response_norm"] (
        X: FloatTensor,
        Size: Int32,
        Alpha: Fp16,
        Beta: Fp16,
        K: Fp16
    );
    [
        ReduceSum => "reduce_sum",
        ReduceMean => "reduce_mean",
        ReduceMin => "reduce_min",
        ReduceMax => "reduce_max",
    ] (
        X: FloatTensor,
        Axes: Int32List,
        KeepDims: Bool
    );
    [Reshape => "reshape"] (X: Tensor, Shape: Int32List);
    [Transpose => "transpose"] (X: Tensor, Perm: Int32List);
    [Cast => "cast"] (X: Tensor, Dtype: String);
    [SliceBySize => "slice_by_size"] (X: Tensor, Begin: Int32List, Size: Int32List);
    [DynamicSlice => "dynamic_slice"] (X: Tensor, Begin: Int32Tensor, Axis: Int32, Size: Int32);
    [SliceByIndex => "slice_by_index"] (
        X: Tensor,
        Begin: Int32List,
        End: Int32List,
        Stride: Int32List,
        BeginMask: BoolList,
        EndMask: BoolList,
        SqueezeMask: BoolList
    );
    [Concat => "concat"] (Values: Tensor, Axis: Int32, Interleave: Bool);
    [Tile => "tile"] (X: Tensor, Reps: Int32List);
    [Reverse => "reverse"] (X: Tensor, Axes: Int32List);
    [Pad => "pad"] (X: FloatTensor, Pad: Int32List, Mode: String, ConstantVal: Fp16);
    [DepthToSpace => "depth_to_space", SpaceToDepth => "space_to_depth"] (X: FloatTensor, BlockSize: Int32);
    [PixelShuffle => "pixel_shuffle"] (X: FloatTensor, UpscaleFactor: Int32);
    [PixelUnshuffle => "pixel_unshuffle"] (X: FloatTensor, DownscaleFactor: Int32);
    [BatchToSpace => "batch_to_space"] (X: FloatTensor, BlockShape: Int32List, Crops: Int32Matrix);
    [Conv => "conv"] (
        X: FloatTensor,
        Weight: FloatTensor,
        Strides: Int32List,
        Dilations: Int32List,
        Groups: Int32,
        Pad: Int32List,
        PadType: String;
        Bias: FloatBlob
    );
    [ConvTranspose => "conv_transpose"] (
        X: FloatTensor,
        Weight: FloatTensor,
        Strides: Int32List,
        Dilations: Int32List,
        Groups: Int32,
        Pad: Int32List,
        PadType: String,
        OutputShape: Int32List;
        Bias: FloatBlob
    );
    [AvgPool => "avg_pool"] (
        X: FloatTensor,
        KernelSizes: Int32List,
        Strides: Int32List,
        Pad: Int32List,
        PadType: String,
        CeilMode: Bool,
        ExcludePaddingFromAverage: Bool
    );
    [MaxPool => "max_pool", L2Pool => "l2_pool"] (
        X: FloatTensor,
        KernelSizes: Int32List,
        Strides: Int32List,
        Pad: Int32List,
        PadType: String,
        CeilMode: Bool
    );
    [Topk => "topk"] (
        X: FloatTensor,
        K: Int32,
        Axis: Int32,
        Ascending: Bool,
        Sort: Bool,
        ReturnIndices: Bool,
        OutputIndicesDtype: String
    );
    [Gather => "gather"] (
        X: Tensor,
        Indices: IndexTensor,
        Axis: Int32,
        BatchDims: Int32,
        ValidateIndices: Bool
    );
    [GatherAlongAxis => "gather_along_axis"] (
        X: Tensor,
        Indices: IndexTensor,
        Axis: Int32,
        ValidateIndices: Bool
    );
    [GatherNd => "gather_nd"] (X: Tensor, Indices: IndexTensor, BatchDims: Int32, ValidateIndices: Bool);
    [ResizeNearestNeighbor => "resize_nearest_neighbor"] (
        X: FloatTensor,
        TargetSizeHeight: Int32,
        TargetSizeWidth: Int32
    );
    [ResizeBilinear => "resize_bilinear"] (
        X: FloatTensor,
        TargetSizeHeight: Int32,
        TargetSizeWidth: Int32,
        SamplingMode: String
    );
    [Resample => "resample"] (
        X: FloatTensor,
        Coordinates: FloatTensor,
        SamplingMode: String,
        PaddingMode: String,
        PaddingValue: Fp16,
        CoordinatesMode: String,
        AlignCorners: Bool
    );
    [Affine => "affine"] (
        X: FloatTensor,
        TransformMatrix: FloatTensor,
        SamplingMode: String,
        PaddingMode: String,
        PaddingValue: Fp16,
        CoordinatesMode: String,
        AlignCorners: Bool,
        OutputHeight: Int32,
        OutputWidth: Int32
    );
    [Quantize => "quantize"] (
        Input: FloatTensor,
        Scale: Scale,
        OutputDtype: String;
        ZeroPoint: ZeroPoint,
        Axis: Int32
    );
    [Dequantize => "dequantize"] (Input: Tensor, Scale: Scale; ZeroPoint: ZeroPoint, Axis: Int32);
    [ConstexprBlockwiseShiftScale => "constexpr_blockwise_shift_scale"] (
        Data: Blob,
        Scale: FloatBlob;
        Offset: FloatBlob
    );
    [ConstexprLutToDense => "constexpr_lut_to_dense"] (Indices: Blob, Lut: FloatBlob);
    [ConstexprSparseToDense => "constexpr_sparse_to_dense"] (Mask: MaskBlob, NonzeroData: FloatBlob);
    [SparseBlockwiseWeights => "sparse_blockwise_weights"] (
        DataMask: MaskBlob,
        NonzeroData: Blob,
        Scale: FloatBlob;
        Offset: FloatBlob
    );
    [SparsePaletteWeights => "sparse_palette_weights"] (
        IndicesMask: MaskBlob,
        IndicesNonzeroData: Blob,
        Lut: FloatBlob
    );
}

impl Operator {
    pub fn is_constant(self) -> bool {
        matches!(
            self,
            Self::ConstexprBlockwiseShiftScale
                | Self::ConstexprLutToDense
                | Self::ConstexprSparseToDense
                | Self::SparseBlockwiseWeights
                | Self::SparsePaletteWeights
        )
    }
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
