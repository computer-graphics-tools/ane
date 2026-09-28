use crate::ops::weights::WeightBlob;
use crate::{
    ActivationMode, ActivationOp, ConcatOp, ConvOp, DeconvOp, ElementwiseOp, ElementwiseOpType,
    FlattenOp, InstanceNormOp, MatmulOp, Op, PadFillMode, PadMode, PaddingOp, PoolType, PoolingOp,
    ReductionMode, ReductionOp, ReshapeOp, SliceBySizeOp, SoftmaxOp, TransposeOp,
};

use super::Graph;
use super::Tensor;
use crate::graph::{Convolution2dDescriptor, ConvolutionTranspose2dDescriptor};
use crate::{DataType, InnerProductOp, ScalarOp, ScalarOpType};

impl Graph {
    pub fn integer_parameter(&mut self) -> Tensor {
        self.placeholder_with_type(&[1, 1, 1, 1], DataType::Int32)
    }

    pub fn placeholder(&mut self, shape: &[usize]) -> Tensor {
        self.placeholder_with_type(shape, DataType::Float32)
    }

    pub fn placeholder_with_type(&mut self, shape: &[usize], data_type: DataType) -> Tensor {
        let shape = crate::dimensions(shape);
        assert!(shape.iter().product::<usize>() > 0, "empty input");
        let tensor = self.alloc(shape);
        self.inputs.push((tensor, data_type));
        tensor
    }

    pub fn constant(&mut self, data: &[f32], shape: &[usize]) -> Tensor {
        let shape = crate::dimensions(shape);
        let tensor = self.alloc(shape);
        self.constants
            .insert(tensor.id, (WeightBlob::from_f32(data), shape));
        tensor
    }

    pub fn constant_with_f16_bytes(&mut self, data: &[u8], shape: &[usize]) -> Tensor {
        let shape = crate::dimensions(shape);
        let tensor = self.alloc(shape);
        self.constants
            .insert(tensor.id, (WeightBlob { data: data.into() }, shape));
        tensor
    }

    pub fn constant_with_scalar(&mut self, scalar: f32, shape: &[usize]) -> Tensor {
        let shape = crate::dimensions(shape);
        let count = shape.iter().product::<usize>();
        let data = vec![scalar; count];
        self.constant(&data, &shape)
    }

    fn resolve_constant(&self, tensor: Tensor) -> WeightBlob {
        let (weights, _) = self
            .constants
            .get(&tensor.id)
            .expect("tensor is not a constant");
        weights.clone()
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
        let out_channels = weights.shape[1];
        let kernel_h = weights.shape[2];
        let kernel_w = weights.shape[3];
        let out_h = match descriptor.pad_mode {
            PadMode::Valid => source.shape[2].saturating_sub(kernel_h) + 1,
            PadMode::Same => source.shape[2],
        };
        let out_w = match descriptor.pad_mode {
            PadMode::Valid => source.shape[3].saturating_sub(kernel_w) + 1,
            PadMode::Same => source.shape[3],
        };
        let output = self.alloc([source.shape[0], out_channels, out_h, out_w]);
        self.ops.push((
            Op::Conv(ConvOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                input_channels: source.shape[1],
                output_channels: out_channels,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                groups: descriptor.groups,
                pad_mode: descriptor.pad_mode,
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
        ));
        output
    }

    pub fn convolution_transpose_2d(
        &mut self,
        source: Tensor,
        weights: Tensor,
        bias: Option<Tensor>,
        descriptor: &ConvolutionTranspose2dDescriptor,
    ) -> Tensor {
        let out_channels = weights.shape[1];
        let kernel_h = weights.shape[2];
        let kernel_w = weights.shape[3];
        let out_h = source.shape[2] * descriptor.stride_height;
        let out_w = source.shape[3] * descriptor.stride_width;
        let output = self.alloc([source.shape[0], out_channels, out_h, out_w]);
        self.ops.push((
            Op::Deconv(DeconvOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                input_channels: source.shape[1],
                output_channels: out_channels,
                kernel_height: kernel_h,
                kernel_width: kernel_w,
                stride_height: descriptor.stride_height,
                stride_width: descriptor.stride_width,
                groups: descriptor.groups,
                pad_mode: descriptor.pad_mode,
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
        ));
        output
    }

    fn activation(&mut self, input: Tensor, mode: ActivationMode) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push((
            Op::Activation(ActivationOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                mode,
            }),
            output,
        ));
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
        let l = left_hand_side.shape;
        let r = right_hand_side.shape;
        assert!(
            l.iter().zip(r).all(|(&a, b)| a == b || a == 1 || b == 1),
            "incompatible broadcast shapes"
        );
        let output = self.alloc(std::array::from_fn(|i| l[i].max(r[i])));
        let left_hand_side_name = Self::tensor_name(left_hand_side);
        let right_hand_side_name = Self::tensor_name(right_hand_side);
        self.ops.push((
            Op::Elementwise(ElementwiseOp {
                name: Self::op_name(output),
                bottoms: vec![left_hand_side_name, right_hand_side_name].into_boxed_slice(),
                top: Self::tensor_name(output),
                operation: op,
                alpha: 1.0,
                beta: 0.0,
                fused_relu: false,
            }),
            output,
        ));
        output
    }

    fn elementwise_unary(&mut self, input: Tensor, op: ElementwiseOpType, alpha: f64) -> Tensor {
        let output = self.alloc(input.shape);
        self.ops.push((
            Op::Elementwise(ElementwiseOp {
                name: Self::op_name(output),
                bottoms: vec![Self::tensor_name(input)].into_boxed_slice(),
                top: Self::tensor_name(output),
                operation: op,
                alpha,
                beta: 0.0,
                fused_relu: false,
            }),
            output,
        ));
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
        self.elementwise_unary(input, ElementwiseOpType::Abs, 1.0)
    }

    pub fn square_root(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Sqrt, 1.0)
    }

    pub fn reciprocal_square_root(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Rsqrt, 1.0)
    }

    pub fn exponent(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Exp, 1.0)
    }

    pub fn logarithm(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Log, 1.0)
    }

    pub fn reciprocal(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Inverse, 1.0)
    }

    pub fn soft_max(&mut self, input: Tensor, axis: i64) -> Tensor {
        assert!((-4..4).contains(&axis), "invalid softmax axis");
        let output = self.alloc(input.shape);
        self.ops.push((
            Op::Softmax(SoftmaxOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                axis,
            }),
            output,
        ));
        output
    }

    pub fn concat(&mut self, inputs: &[Tensor], axis: usize) -> Tensor {
        assert!(!inputs.is_empty(), "concat requires at least one input");
        assert!(axis < 4, "concat axis must be in 0..4");
        let base = inputs[0].shape;
        assert!(
            inputs
                .iter()
                .all(|t| (0..4).all(|i| i == axis || t.shape[i] == base[i])),
            "concat dimensions differ"
        );
        let mut out_dims = base;
        out_dims[axis] = inputs.iter().map(|t| t.shape[axis]).sum();
        let output = self.alloc(out_dims);
        let bottoms: Box<[String]> = inputs.iter().map(|t| Self::tensor_name(*t)).collect();
        self.ops.push((
            Op::Concat(ConcatOp {
                name: Self::op_name(output),
                bottoms,
                top: Self::tensor_name(output),
                axis,
            }),
            output,
        ));
        output
    }

    pub fn matrix_multiplication(
        &mut self,
        left_hand_side: Tensor,
        right_hand_side: Tensor,
        transpose_x: bool,
        transpose_y: bool,
    ) -> Tensor {
        let lx = left_hand_side.shape;
        let ry = right_hand_side.shape;
        assert_eq!(
            if transpose_x { lx[2] } else { lx[3] },
            if transpose_y { ry[3] } else { ry[2] },
            "matmul contraction dimensions differ"
        );
        assert!(
            (lx[0] == ry[0] || lx[0] == 1 || ry[0] == 1)
                && (lx[1] == ry[1] || lx[1] == 1 || ry[1] == 1),
            "matmul batch dimensions differ"
        );
        let out_h = if transpose_x {
            left_hand_side.shape[3]
        } else {
            left_hand_side.shape[2]
        };
        let out_w = if transpose_y {
            right_hand_side.shape[2]
        } else {
            right_hand_side.shape[3]
        };
        let output = self.alloc([lx[0].max(ry[0]), lx[1].max(ry[1]), out_h, out_w]);
        self.ops.push((
            Op::Matmul(MatmulOp {
                name: Self::op_name(output),
                bottom_x: Self::tensor_name(left_hand_side),
                bottom_y: Self::tensor_name(right_hand_side),
                top: Self::tensor_name(output),
                transpose_x,
                transpose_y,
            }),
            output,
        ));
        output
    }

    pub fn transpose(&mut self, input: Tensor, perm: [usize; 4]) -> Tensor {
        let mut sorted = perm;
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3], "invalid permutation");
        let output = self.alloc(perm.map(|axis| input.shape[axis]));
        self.ops.push((
            Op::Transpose(TransposeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                perm,
            }),
            output,
        ));
        output
    }

    pub fn slice(&mut self, input: Tensor, begin: [usize; 4], size: [usize; 4]) -> Tensor {
        assert!(
            (0..4).all(|i| size[i] > 0
                && begin[i] <= input.shape[i]
                && size[i] <= input.shape[i] - begin[i]),
            "slice exceeds input"
        );
        let output = self.alloc(size);
        self.ops.push((
            Op::SliceBySize(SliceBySizeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                begin,
                size,
            }),
            output,
        ));
        output
    }

    pub fn reshape(&mut self, input: Tensor, target: &[usize]) -> Tensor {
        let target = crate::dimensions(target);
        assert_eq!(
            input.shape.iter().product::<usize>(),
            target.iter().product::<usize>(),
            "reshape element count differs"
        );
        let output = self.alloc(target);
        self.ops.push((
            Op::Reshape(ReshapeOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                target_shape: target,
            }),
            output,
        ));
        output
    }

    pub fn flatten_2d(&mut self, input: Tensor) -> Tensor {
        let flat_k = input.shape.iter().product::<usize>();
        let output = self.alloc([1, flat_k, 1, 1]);
        self.ops.push((
            Op::Flatten(FlattenOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
            }),
            output,
        ));
        output
    }

    fn pool(
        &mut self,
        input: Tensor,
        pool_type: PoolType,
        kernel: [usize; 2],
        strides: [usize; 2],
        pad_mode: PadMode,
        global: bool,
    ) -> Tensor {
        let [kernel_h, kernel_w] = kernel;
        let [stride_height, stride_width] = strides;
        assert!(
            kernel_h > 0 && kernel_w > 0 && stride_height > 0 && stride_width > 0,
            "empty pooling geometry"
        );
        let (out_h, out_w) = if global {
            (1, 1)
        } else {
            match pad_mode {
                PadMode::Valid => (
                    (input.shape[2].saturating_sub(kernel_h)) / stride_height + 1,
                    (input.shape[3].saturating_sub(kernel_w)) / stride_width + 1,
                ),
                PadMode::Same => (
                    input.shape[2].div_ceil(stride_height),
                    input.shape[3].div_ceil(stride_width),
                ),
            }
        };
        let output = self.alloc([input.shape[0], input.shape[1], out_h, out_w]);
        self.ops.push((
            Op::Pooling(PoolingOp {
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
        ));
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
        self.pool(
            input,
            PoolType::Max,
            [kernel_h, kernel_w],
            [stride_height, stride_width],
            pad_mode,
            false,
        )
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
        self.pool(
            input,
            PoolType::Average,
            [kernel_h, kernel_w],
            [stride_height, stride_width],
            pad_mode,
            false,
        )
    }

    pub fn global_avg_pool(&mut self, input: Tensor) -> Tensor {
        let kh = input.shape[2];
        let kw = input.shape[3];
        self.pool(
            input,
            PoolType::Average,
            [kh, kw],
            [1, 1],
            PadMode::Valid,
            true,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "preserves the existing public padding API"
    )]
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
        let output = self.alloc([
            input.shape[0],
            input.shape[1],
            input.shape[2] + top + bottom,
            input.shape[3] + left + right,
        ]);
        self.ops.push((
            Op::Padding(PaddingOp {
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
        ));
        output
    }

    fn reduce(&mut self, input: Tensor, mode: ReductionMode, axis: i64) -> Tensor {
        assert!((-4..4).contains(&axis), "invalid reduction axis");
        let mut out_shape = input.shape;
        out_shape[axis.rem_euclid(4) as usize] = 1;
        let output = self.alloc(out_shape);
        self.ops.push((
            Op::Reduction(ReductionOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                mode,
                axis,
            }),
            output,
        ));
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

    pub fn instance_norm(&mut self, source: Tensor, params: Tensor, epsilon: f64) -> Tensor {
        let output = self.alloc(source.shape);
        self.ops.push((
            Op::InstanceNorm(InstanceNormOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(source),
                top: Self::tensor_name(output),
                channels: source.shape[1],
                epsilon,
                params: self.resolve_constant(params),
            }),
            output,
        ));
        output
    }
}

impl Graph {
    pub fn inner_product(
        &mut self,
        input: Tensor,
        weights: Tensor,
        bias: Option<Tensor>,
    ) -> Tensor {
        let data = self.resolve_constant(weights);
        assert_eq!(data.data.len(), input.shape[1] * weights.shape[1] * 2);
        let bias = bias.map(|b| self.resolve_constant(b));
        if let Some(b) = &bias {
            assert_eq!(b.data.len(), weights.shape[1] * 2);
        }
        let output = self.alloc([
            input.shape[0],
            weights.shape[1],
            input.shape[2],
            input.shape[3],
        ]);
        self.ops.push((
            Op::InnerProduct(InnerProductOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                input_channels: input.shape[1],
                output_channels: output.shape[1],
                weights: data,
                bias,
                has_relu: false,
                has_tanh: false,
            }),
            output,
        ));
        output
    }

    fn scalar(&mut self, input: Tensor, op: ScalarOpType, scalar: f32) -> Tensor {
        assert!(scalar.is_finite(), "scalar must be finite");
        let output = self.alloc(input.shape);
        self.ops.push((
            Op::ScalarOp(ScalarOp {
                name: Self::op_name(output),
                bottom: Self::tensor_name(input),
                top: Self::tensor_name(output),
                op,
                scalar,
            }),
            output,
        ));
        output
    }

    pub fn multiply_scalar(&mut self, input: Tensor, value: f32) -> Tensor {
        self.scalar(input, ScalarOpType::Mul, value)
    }
    pub fn add_scalar(&mut self, input: Tensor, value: f32) -> Tensor {
        self.scalar(input, ScalarOpType::Add, value)
    }
    pub fn reverse_subtract_scalar(&mut self, input: Tensor, value: f32) -> Tensor {
        self.scalar(input, ScalarOpType::RSub, value)
    }
    pub fn power_scalar(&mut self, input: Tensor, value: f32) -> Tensor {
        self.scalar(input, ScalarOpType::Pow, value)
    }
    pub fn minimum_scalar(&mut self, input: Tensor, value: f32) -> Tensor {
        self.scalar(input, ScalarOpType::Min, value)
    }
    pub fn maximum_scalar(&mut self, input: Tensor, value: f32) -> Tensor {
        self.scalar(input, ScalarOpType::Max, value)
    }
    pub fn clip(&mut self, input: Tensor, minimum: f32, maximum: f32) -> Tensor {
        assert!(minimum <= maximum, "invalid clipping interval");
        let lower = self.maximum_scalar(input, minimum);
        self.minimum_scalar(lower, maximum)
    }
    pub fn floor(&mut self, input: Tensor) -> Tensor {
        self.elementwise_unary(input, ElementwiseOpType::Floor, 1.0)
    }
    pub fn threshold(&mut self, input: Tensor, minimum: f64) -> Tensor {
        assert!(minimum.is_finite(), "threshold must be finite");
        self.elementwise_unary(input, ElementwiseOpType::Threshold, minimum)
    }
    pub fn global_max_pool(&mut self, input: Tensor) -> Tensor {
        self.pool(
            input,
            PoolType::Max,
            [input.shape[2], input.shape[3]],
            [1, 1],
            PadMode::Valid,
            true,
        )
    }

    pub fn unpack_int4(&mut self, bytes: Tensor) -> Tensor {
        let divided = self.multiply_scalar(bytes, 1.0 / 16.0);
        let high = self.floor(divided);
        let shifted = self.multiply_scalar(high, 16.0);
        let low = self.subtraction(bytes, shifted);
        let lanes = [
            bytes.shape[0]
                .checked_mul(bytes.shape[1])
                .expect("unpacked batch overflow"),
            bytes.shape[2],
            bytes.shape[3],
            1,
        ];
        let low = self.reshape(low, &lanes);
        let high = self.reshape(high, &lanes);
        let pairs = self.concat(&[low, high], 3);
        self.reshape(
            pairs,
            &[
                bytes.shape[0],
                bytes.shape[1],
                bytes.shape[2],
                bytes.shape[3]
                    .checked_mul(2)
                    .expect("unpacked width overflow"),
            ],
        )
    }

    pub fn unpack_signed_int4(&mut self, bytes: Tensor) -> Tensor {
        let codes = self.unpack_int4(bytes);
        let sign = self.add_scalar(codes, -7.0);
        let sign = self.clip(sign, 0.0, 1.0);
        let correction = self.multiply_scalar(sign, 16.0);
        self.subtraction(codes, correction)
    }

    pub fn pack_signed_int4(&mut self, codes: Tensor) -> Tensor {
        assert!(
            codes.shape[3].is_multiple_of(2),
            "packing needs an even width"
        );
        let negative = self.multiply_scalar(codes, -1.0);
        let negative = self.clip(negative, 0.0, 1.0);
        let correction = self.multiply_scalar(negative, 16.0);
        let unsigned = self.addition(codes, correction);
        let lanes = [
            codes.shape[0]
                .checked_mul(codes.shape[1])
                .expect("packing batch overflow"),
            codes.shape[2],
            codes.shape[3] / 2,
            2,
        ];
        let pairs = self.reshape(unsigned, &lanes);
        let size = [lanes[0], lanes[1], lanes[2], 1];
        let low = self.slice(pairs, [0, 0, 0, 0], size);
        let high = self.slice(pairs, [0, 0, 0, 1], size);
        let high = self.multiply_scalar(high, 16.0);
        let bytes = self.addition(low, high);
        self.reshape(
            bytes,
            &[
                codes.shape[0],
                codes.shape[1],
                codes.shape[2],
                codes.shape[3] / 2,
            ],
        )
    }

    pub fn dequantize_groupwise(
        &mut self,
        codes: Tensor,
        scales: Tensor,
        zero_points: Option<Tensor>,
        biases: Option<Tensor>,
        group_size: usize,
    ) -> Tensor {
        assert!(
            codes.shape[0] == 1 && codes.shape[1] == 1,
            "codes must be [1,1,N,K]"
        );
        assert!(
            group_size > 0 && codes.shape[3].is_multiple_of(group_size),
            "group size must divide K"
        );
        let groups = codes.shape[3] / group_size;
        let metadata = [1, codes.shape[2], groups, 1];
        assert_eq!(
            scales.shape.iter().product::<usize>(),
            metadata.iter().product::<usize>(),
            "scale count differs"
        );
        let shape = [metadata[0], metadata[1], metadata[2], group_size];
        let mut values = self.reshape(codes, &shape);
        if let Some(offsets) = zero_points {
            assert_eq!(
                offsets.shape.iter().product::<usize>(),
                metadata.iter().product::<usize>(),
                "zero-point count differs"
            );
            let offsets = self.reshape(offsets, &metadata);
            values = self.subtraction(values, offsets);
        }
        let scales = self.reshape(scales, &metadata);
        values = self.multiplication(values, scales);
        if let Some(biases) = biases {
            assert_eq!(
                biases.shape.iter().product::<usize>(),
                metadata.iter().product::<usize>(),
                "bias count differs"
            );
            let biases = self.reshape(biases, &metadata);
            values = self.addition(values, biases);
        }
        self.reshape(values, &codes.shape)
    }
}
