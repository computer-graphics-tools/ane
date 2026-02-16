use crate::{
    ActivationOp, ActivationMode, ConcatOp, ConvOp, DeconvOp,
    ElementwiseOp, ElementwiseOpType, FlattenOp, InstanceNormOp,
    MatmulOp, Op, PadFillMode, PadMode, PaddingOp, PoolType, PoolingOp,
    ReductionOp, ReductionMode, ReshapeOp, Shape,
    SliceBySizeOp, SoftmaxOp, TransposeOp,
};
use crate::ops::weights::WeightBlob;

use super::descriptor::{Convolution2dDescriptor, ConvolutionTranspose2dDescriptor};
use super::tensor::Tensor;
use super::{Graph, GraphOp};

impl Graph {
    /// Declare a placeholder tensor fed via IOSurface at runtime.
    pub fn placeholder(&mut self, shape: Shape) -> Tensor {
        let tensor = self.alloc(shape);
        self.inputs.push(tensor);
        tensor
    }

    /// Create a constant tensor from f32 data (converted to fp16 internally).
    pub fn constant(&mut self, data: &[f32], shape: Shape) -> Tensor {
        let tensor = self.alloc(shape);
        self.constants.insert(tensor.id, (WeightBlob::from_f32(data).data, shape));
        tensor
    }

    /// Create a constant tensor from pre-encoded fp16 bytes.
    pub fn constant_with_f16_bytes(&mut self, data: &[u8], shape: Shape) -> Tensor {
        let tensor = self.alloc(shape);
        self.constants.insert(tensor.id, (data.into(), shape));
        tensor
    }

    /// Create a constant tensor filled with a scalar value.
    pub fn constant_with_scalar(&mut self, scalar: f32, shape: Shape) -> Tensor {
        let count = shape.total_elements();
        let data = vec![scalar; count];
        self.constant(&data, shape)
    }

    pub(crate) fn resolve_constant(&self, tensor: Tensor) -> WeightBlob {
        let (bytes, _) = self.constants.get(&tensor.id).expect("tensor is not a constant");
        WeightBlob::from_f16_bytes(bytes.clone())
    }

    pub fn convolution_2d_1x1(
        &mut self,
        source: Tensor,
        weights: Tensor,
        bias: Option<Tensor>,
    ) -> Tensor {
        self.convolution_2d(source, weights, bias, &Convolution2dDescriptor::default())
    }

    pub fn convolution_2d(
        &mut self,
        source: Tensor,
        weights: Tensor,
        bias: Option<Tensor>,
        descriptor: &Convolution2dDescriptor,
    ) -> Tensor {
        let out_channels = weights.shape.channels;
        let kernel_h = weights.shape.height;
        let kernel_w = weights.shape.width;
        let out_h = match descriptor.pad_mode {
            PadMode::Valid => source.shape.height.saturating_sub(kernel_h) + 1,
            PadMode::Same => source.shape.height,
        };
        let out_w = match descriptor.pad_mode {
            PadMode::Valid => source.shape.width.saturating_sub(kernel_w) + 1,
            PadMode::Same => source.shape.width,
        };
        let output = self.alloc(Shape { channels: out_channels, height: out_h, width: out_w, batch: 1 });
        self.ops.push(GraphOp {
            op: Op::Conv(ConvOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                input_channels: source.shape.channels,
                output_channels: out_channels,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                groups: descriptor.groups,
                pad_mode: descriptor.pad_mode.clone(),
                pad_top: 0,
                pad_bottom: 0,
                pad_left: 0,
                pad_right: 0,
                weights: self.resolve_constant(weights),
                bias: bias.map(|b| self.resolve_constant(b)),
                fused_relu: false,
                fused_tanh: false,
            }),
            output,
        });
        output
    }

    pub fn convolution_transpose_2d(
        &mut self,
        source: Tensor,
        weights: Tensor,
        bias: Option<Tensor>,
        descriptor: &ConvolutionTranspose2dDescriptor,
    ) -> Tensor {
        let out_channels = weights.shape.channels;
        let kernel_h = weights.shape.height;
        let kernel_w = weights.shape.width;
        let out_h = source.shape.height * descriptor.stride_height;
        let out_w = source.shape.width * descriptor.stride_width;
        let output = self.alloc(Shape { channels: out_channels, height: out_h, width: out_w, batch: 1 });
        self.ops.push(GraphOp {
            op: Op::Deconv(DeconvOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                input_channels: source.shape.channels,
                output_channels: out_channels,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                stride_height: descriptor.stride_height,
                stride_width: descriptor.stride_width,
                groups: descriptor.groups,
                pad_mode: descriptor.pad_mode.clone(),
                pad_top: 0,
                pad_bottom: 0,
                pad_left: 0,
                pad_right: 0,
                output_padding_height: 0,
                output_padding_width: 0,
                weights: self.resolve_constant(weights),
                bias: bias.map(|b| self.resolve_constant(b)),
                fused_relu: false,
                fused_tanh: false,
            }),
            output,
        });
        output
    }


    fn activation(&mut self, input: Tensor, mode: ActivationMode) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Activation(ActivationOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                mode,
            }),
            output,
        });
        output
    }

    pub fn relu(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::Relu)
    }

    pub fn tanh(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::Tanh)
    }

    pub fn sigmoid(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::Sigmoid)
    }

    pub fn leaky_relu(&mut self, input: Tensor, negative_slope: f64) -> Tensor {
        self.activation(input, ActivationMode::LeakyRelu { negative_slope })
    }

    pub fn elu(&mut self, input: Tensor, alpha: f64) -> Tensor {
        self.activation(input, ActivationMode::Elu { alpha })
    }

    pub fn hard_sigmoid(&mut self, input: Tensor, alpha: f64, beta: f64) -> Tensor {
        self.activation(input, ActivationMode::SigmoidHard { alpha, beta })
    }

    pub fn linear(&mut self, input: Tensor, alpha: f64, beta: f64) -> Tensor {
        self.activation(input, ActivationMode::Linear { alpha, beta })
    }

    pub fn softplus(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::SoftPlus)
    }

    pub fn softsign(&mut self, input: Tensor) -> Tensor {
        self.activation(input, ActivationMode::SoftSign)
    }


    fn elementwise_binary(
        &mut self,
        left_hand_side: Tensor,
        right_hand_side: Tensor,
        op: ElementwiseOpType,
    ) -> Tensor {
        let output = self.alloc(Shape {
            batch: left_hand_side.shape.batch.max(right_hand_side.shape.batch),
            channels: left_hand_side.shape.channels.max(right_hand_side.shape.channels),
            height: left_hand_side.shape.height.max(right_hand_side.shape.height),
            width: left_hand_side.shape.width.max(right_hand_side.shape.width),
        });
        let left_hand_side_name = Self::tensor_name(left_hand_side);
        let right_hand_side_name = Self::tensor_name(right_hand_side);
        self.ops.push(GraphOp {
            op: Op::Elementwise(ElementwiseOp {
                name: Self::op_name(output),
                bottoms: vec![left_hand_side_name, right_hand_side_name].into_boxed_slice(),
                top: Self::tensor_name(output),
                operation: op,
                alpha: 1.0,
                beta: 0.0,
                fused_relu: false,
            }),
            output,
        });
        output
    }

    fn elementwise_unary(&mut self, input: Tensor, op: ElementwiseOpType) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Elementwise(ElementwiseOp {
                name: Self::op_name(output),
                bottoms: vec![Self::tensor_name(input)].into_boxed_slice(),
                top: Self::tensor_name(output),
                operation: op,
                alpha: 1.0,
                beta: 0.0,
                fused_relu: false,
            }),
            output,
        });
        output
    }

    pub fn addition(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Add)
    }

    pub fn subtraction(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Sub)
    }

    pub fn multiplication(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Multiply)
    }

    pub fn division(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Div)
    }

    pub fn power(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Pow)
    }

    pub fn maximum(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Max)
    }

    pub fn minimum(&mut self, left_hand_side: Tensor, right_hand_side: Tensor) -> Tensor {
        self.elementwise_binary(left_hand_side, right_hand_side, ElementwiseOpType::Min)
    }

    pub fn absolute(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Abs)
    }

    pub fn square_root(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Sqrt)
    }

    pub fn reciprocal_square_root(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Rsqrt)
    }

    pub fn exponent(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Exp)
    }

    pub fn logarithm(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Log)
    }

    pub fn reciprocal(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Inverse)
    }


    /// Softmax along the specified axis.
    pub fn soft_max(&mut self, input: Tensor, axis: i64) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push(GraphOp {
            op: Op::Softmax(SoftmaxOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axis,
            }),
            output,
        });
        output
    }

    /// Concatenate tensors along the specified axis.
    pub fn concat(&mut self, inputs: &[Tensor], axis: usize) -> Tensor {
        assert!(!inputs.is_empty(), "concat requires at least one input");
        let base = inputs[0].shape;
        let sum_dim: usize = inputs.iter().map(|t| {
            let dims = [t.shape.batch, t.shape.channels, t.shape.height, t.shape.width];
            dims[axis]
        }).sum();
        let mut out_dims = [base.batch, base.channels, base.height, base.width];
        out_dims[axis] = sum_dim;
        let output = self.alloc(Shape {
            batch: out_dims[0],
            channels: out_dims[1],
            height: out_dims[2],
            width: out_dims[3],
        });
        let bottoms: Box<[String]> = inputs.iter().map(|t| Self::tensor_name(*t)).collect();
        self.ops.push(GraphOp {
            op: Op::Concat(ConcatOp {
                name: Self::op_name(output),
                bottoms,
                top: Self::tensor_name(output),
                axis,
            }),
            output,
        });
        output
    }


    /// Matrix multiplication: `out = x @ y` (with optional transposes on last two dims).
    pub fn matrix_multiplication(
        &mut self,
        left_hand_side: Tensor,
        right_hand_side: Tensor,
        transpose_x: bool,
        transpose_y: bool,
    ) -> Tensor {
        let out_h = if transpose_x { left_hand_side.shape.width } else { left_hand_side.shape.height };
        let out_w = if transpose_y { right_hand_side.shape.height } else { right_hand_side.shape.width };
        let output = self.alloc(Shape {
            batch: left_hand_side.shape.batch,
            channels: left_hand_side.shape.channels,
            height: out_h,
            width: out_w,
        });
        self.ops.push(GraphOp {
            op: Op::Matmul(MatmulOp {
                name: Self::op_name(output),
                bottom_x: Self::tensor_name(left_hand_side),
                bottom_y: Self::tensor_name(right_hand_side),
                top: Self::tensor_name(output),
                transpose_x,
                transpose_y,
            }),
            output,
        });
        output
    }


    /// Permute the dimensions of `x` according to `perm` (4-element NCHW permutation).
    pub fn transpose(&mut self, input: Tensor, perm: [usize; 4]) -> Tensor {
        let dimensions = [input.shape.batch, input.shape.channels, input.shape.height, input.shape.width];
        let output = self.alloc(Shape {
            batch: dimensions[perm[0]],
            channels: dimensions[perm[1]],
            height: dimensions[perm[2]],
            width: dimensions[perm[3]],
        });
        self.ops.push(GraphOp {
            op: Op::Transpose(TransposeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                perm,
            }),
            output,
        });
        output
    }


    /// Extract a sub-tensor starting at `begin` with dimensions `size` (both in NCHW order).
    pub fn slice(&mut self, input: Tensor, begin: [usize; 4], size: [usize; 4]) -> Tensor {
        let output = self.alloc(Shape {
            batch: size[0],
            channels: size[1],
            height: size[2],
            width: size[3],
        });
        self.ops.push(GraphOp {
            op: Op::SliceBySize(SliceBySizeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                begin,
                size,
            }),
            output,
        });
        output
    }


    pub fn reshape(&mut self, input: Tensor, target: Shape) -> Tensor {
        let output = self.alloc(target);
        self.ops.push(GraphOp {
            op: Op::Reshape(ReshapeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                target_shape: [target.batch, target.channels, target.height, target.width],
            }),
            output,
        });
        output
    }

    /// Flatten all dimensions into a single channel axis: `[1, k*h*w, 1, 1]`.
    pub fn flatten_2d(&mut self, input: Tensor) -> Tensor {
        let flat_k = input.shape.total_elements();
        let output = self.alloc(Shape::channels(flat_k));
        self.ops.push(GraphOp {
            op: Op::Flatten(FlattenOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
            }),
            output,
        });
        output
    }


    fn pool(
        &mut self,
        input: Tensor,
        pool_type: PoolType,
        kernel_h: usize,
        kernel_w: usize,
        stride_height: usize,
        stride_width: usize,
        pad_mode: PadMode,
        global: bool,
    ) -> Tensor {
        let (out_h, out_w) = if global {
            (1, 1)
        } else {
            match pad_mode {
                PadMode::Valid => (
                    (input.shape.height.saturating_sub(kernel_h)) / stride_height + 1,
                    (input.shape.width.saturating_sub(kernel_w)) / stride_width + 1,
                ),
                PadMode::Same => (
                    (input.shape.height + stride_height - 1) / stride_height,
                    (input.shape.width + stride_width - 1) / stride_width,
                ),
            }
        };
        let output = self.alloc(Shape { channels: input.shape.channels, height: out_h, width: out_w, batch: 1 });
        self.ops.push(GraphOp {
            op: Op::Pooling(PoolingOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                pool_type,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                stride_height,
                stride_width,
                pad_mode,
                pad_top: 0,
                pad_bottom: 0,
                pad_left: 0,
                pad_right: 0,
                global_pooling: global,
            }),
            output,
        });
        output
    }

    pub fn max_pool(
        &mut self,
        input: Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_height: usize,
        stride_width: usize,
        pad_mode: PadMode,
    ) -> Tensor {
        self.pool(input, PoolType::Max, kernel_h, kernel_w, stride_height, stride_width, pad_mode, false)
    }

    pub fn avg_pool(
        &mut self,
        input: Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_height: usize,
        stride_width: usize,
        pad_mode: PadMode,
    ) -> Tensor {
        self.pool(input, PoolType::Average, kernel_h, kernel_w, stride_height, stride_width, pad_mode, false)
    }

    pub fn global_avg_pool(&mut self, input: Tensor) -> Tensor {
        let kh = input.shape.height;
        let kw = input.shape.width;
        self.pool(input, PoolType::Average, kh, kw, 1, 1, PadMode::Valid, true)
    }


    pub fn pad(
        &mut self,
        input: Tensor,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
        mode: PadFillMode,
        value: f64,
    ) -> Tensor {
        let output = self.alloc(Shape {
            channels: input.shape.channels,
            height: input.shape.height + top + bottom,
            width: input.shape.width + left + right,
            batch: 1,
        });
        self.ops.push(GraphOp {
            op: Op::Padding(PaddingOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                pad_top: top,
                pad_bottom: bottom,
                pad_left: left,
                pad_right: right,
                pad_fill_mode: mode,
                pad_value: value,
            }),
            output,
        });
        output
    }


    fn reduce(&mut self, input: Tensor, mode: ReductionMode, axis: i64) -> Tensor {
        let mut out_shape = input.shape;
        match axis.rem_euclid(4) {
            0 => out_shape.batch = 1,
            1 => out_shape.channels = 1,
            2 => out_shape.height = 1,
            3 => out_shape.width = 1,
            _ => {}
        }
        let output = self.alloc(out_shape);
        self.ops.push(GraphOp {
            op: Op::Reduction(ReductionOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                mode,
                axis,
            }),
            output,
        });
        output
    }

    pub fn reduce_sum(&mut self, input: Tensor, axis: i64) -> Tensor {
        self.reduce(input, ReductionMode::Sum, axis)
    }

    pub fn reduce_mean(&mut self, input: Tensor, axis: i64) -> Tensor {
        self.reduce(input, ReductionMode::Mean, axis)
    }

    pub fn reduce_min(&mut self, input: Tensor, axis: i64) -> Tensor {
        self.reduce(input, ReductionMode::Min, axis)
    }

    pub fn reduce_max(&mut self, input: Tensor, axis: i64) -> Tensor {
        self.reduce(input, ReductionMode::Max, axis)
    }


    pub fn instance_norm(
        &mut self,
        source: Tensor,
        params: Tensor,
        epsilon: f64,
    ) -> Tensor {
        let output = self.alloc(source.shape);
        self.ops.push(GraphOp {
            op: Op::InstanceNorm(InstanceNormOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                channels: source.shape.channels,
                epsilon,
                params: self.resolve_constant(params),
            }),
            output,
        });
        output
    }
}
