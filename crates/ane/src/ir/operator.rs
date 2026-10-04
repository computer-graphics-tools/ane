use std::fmt::{self, Display, Formatter};

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
        Relu6 => "relu6",
        Round => "round",
        Sigmoid => "sigmoid",
        Sign => "sign",
        Silu => "silu",
        Sin => "sin",
        Softplus => "softplus",
        Softsign => "softsign",
        Sqrt => "sqrt",
        Square => "square",
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
    [LogicalAnd => "logical_and"] (X: BoolTensor, Y: BoolTensor);
    [Select => "select"] (Cond: BoolTensor, A: Tensor, B: Tensor);
    [Matmul => "matmul"] (X: FloatTensor, Y: FloatTensor, TransposeX: Bool, TransposeY: Bool);
    [ScaledDotProductAttention => "scaled_dot_product_attention"] (
        Query: FloatTensor,
        Key: FloatTensor,
        Value: FloatTensor;
        AttnMask: FloatTensor
    );
    [LeakyRelu => "leaky_relu", Elu => "elu"] (X: FloatTensor, Alpha: Fp16);
    [
        LinearActivation => "linear_activation",
        SigmoidHard => "sigmoid_hard",
        ScaledTanh => "scaled_tanh",
        ClampedRelu => "clamped_relu",
        Clip => "clip",
    ] (
        X: FloatTensor,
        Alpha: Fp16,
        Beta: Fp16
    );
    [Prelu => "prelu"] (X: FloatTensor, Alpha: FloatBlob);
    [SoftplusParametric => "softplus_parametric"] (X: FloatTensor, Alpha: FloatBlob, Beta: FloatBlob);
    [Linear => "linear"] (X: FloatTensor, Weight: FloatBlob; Bias: FloatBlob);
    [LayerNorm => "layer_norm"] (X: FloatTensor, Axes: Int32List, Epsilon: Fp16; Gamma: FloatBlob, Beta: FloatBlob);
    [InstanceNorm => "instance_norm"] (X: FloatTensor, Epsilon: Fp16; Gamma: FloatBlob, Beta: FloatBlob);
    [BatchNorm => "batch_norm"] (
        X: FloatTensor,
        Mean: FloatBlob,
        Variance: FloatBlob,
        Epsilon: Fp16;
        Gamma: FloatBlob,
        Beta: FloatBlob
    );
    [L2Norm => "l2_norm"] (X: FloatTensor, Epsilon: Fp16);
    [ThresholdedRelu => "thresholded_relu"] (X: FloatTensor, Alpha: Fp16);
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
        ReduceL1Norm => "reduce_l1_norm",
        ReduceL2Norm => "reduce_l2_norm",
        ReduceLogSum => "reduce_log_sum",
        ReduceLogSumExp => "reduce_log_sum_exp",
        ReduceSumSquare => "reduce_sum_square",
    ] (
        X: FloatTensor,
        Axes: Int32List,
        KeepDims: Bool
    );
    [ReduceArgmax => "reduce_argmax", ReduceArgmin => "reduce_argmin"] (
        X: FloatTensor,
        Axis: Int32,
        KeepDims: Bool,
        OutputDtype: String
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
    [SliceUpdate => "slice_update"] (X: Tensor, Update: Tensor, Begin: Int32List, End: Int32List);
    [Split => "split"] (X: Tensor, SplitSizes: Int32List, Axis: Int32);
    [Stack => "stack"] (Values: Tensor, Axis: Int32);
    [Concat => "concat"] (Values: Tensor, Axis: Int32, Interleave: Bool);
    [Tile => "tile"] (X: Tensor, Reps: Int32List);
    [Reverse => "reverse"] (X: Tensor, Axes: Int32List);
    [Pad => "pad"] (X: FloatTensor, Pad: Int32List, Mode: String, ConstantVal: Fp16);
    [DepthToSpace => "depth_to_space", SpaceToDepth => "space_to_depth"] (X: FloatTensor, BlockSize: Int32);
    [PixelShuffle => "pixel_shuffle"] (X: FloatTensor, UpscaleFactor: Int32);
    [SpaceToBatch => "space_to_batch"] (X: FloatTensor, BlockShape: Int32List, Paddings: Int32Matrix);
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
        Offset: Blob
    );
    [ConstexprLutToDense => "constexpr_lut_to_dense"] (
        Indices: Blob,
        Lut: Blob;
        LutScale: FloatBlob,
        LutOffset: Blob,
        VectorAxis: Int32
    );
    [ConstexprSparseToDense => "constexpr_sparse_to_dense"] (Mask: MaskBlob, NonzeroData: FloatBlob);
    [ConstexprSparseBlockwiseShiftScale => "constexpr_sparse_blockwise_shift_scale"] (
        DataMask: MaskBlob,
        NonzeroData: Blob,
        Scale: FloatBlob;
        Offset: Blob
    );
    [ConstexprLutToSparse => "constexpr_lut_to_sparse"] (
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
                | Self::ConstexprSparseBlockwiseShiftScale
                | Self::ConstexprLutToSparse
        )
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
