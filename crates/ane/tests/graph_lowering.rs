use ane::{
    BlockwiseQuantization, Convolution2dDescriptor, ConvolutionTranspose2dDescriptor, DataType,
    Graph, GraphError, PadFillMode, PadMode, PoolType, Pooling2dDescriptor, SamplingDescriptor,
    SamplingMode, Tensor, TensorData, WeightDataType,
};

#[path = "graph_lowering/case.rs"]
mod case;
use case::Case;

fn h<const RANK: usize>(g: &Graph, shape: [usize; RANK]) -> Result<Tensor, GraphError> {
    g.placeholder(shape, DataType::Float16)
}
fn x4(g: &Graph) -> Result<Tensor, GraphError> {
    h(g, [1, 2, 4, 4])
}
fn typed<const RANK: usize>(
    g: &Graph,
    shape: [usize; RANK],
    dtype: DataType,
) -> Result<Tensor, GraphError> {
    g.placeholder(shape, dtype)
}
fn c<const RANK: usize>(g: &Graph, shape: [usize; RANK]) -> Result<Tensor, GraphError> {
    let count = shape.iter().product::<usize>();
    let data: Vec<f32> = (0..count).map(|i| 0.25 + i as f32 * 0.125).collect();
    g.constant(&data, shape)
}

macro_rules! unary {
    ($($name:ident),* $(,)?) => {
        &[$( (stringify!($name), |g: &Graph| { let x = x4(g)?; Ok(vec![g.$name(&x)?]) }) ),*]
    };
}
macro_rules! binary {
    ($($name:ident),* $(,)?) => {
        &[$( (stringify!($name), |g: &Graph| { let x = x4(g)?; let y = x4(g)?; Ok(vec![g.$name(&x, &y)?]) }) ),*]
    };
}
macro_rules! scalar {
    ($($name:ident($value:expr)),* $(,)?) => {
        &[$( (stringify!($name), |g: &Graph| { let x = x4(g)?; Ok(vec![g.$name(&x, $value)?]) }) ),*]
    };
}
macro_rules! axis {
    ($($name:ident),* $(,)?) => {
        &[$( (stringify!($name), |g: &Graph| { let x = x4(g)?; Ok(vec![g.$name(&x, -1)?]) }) ),*]
    };
}
macro_rules! boolean {
    ($($name:ident),* $(,)?) => {
        &[$( (stringify!($name), |g: &Graph| {
            let x = typed(g, [2, 3], DataType::Bool)?;
            let y = typed(g, [2, 3], DataType::Bool)?;
            Ok(vec![g.$name(&x, &y)?])
        }) ),*]
    };
}

const UNARY: &[(&str, Case)] = unary!(
    relu,
    tanh,
    sigmoid,
    softplus,
    softsign,
    absolute,
    square_root,
    reciprocal_square_root,
    exponent,
    logarithm,
    reciprocal,
    floor,
    ceil,
    round,
    sign,
    square,
    negative,
    erf,
    exponent_base2,
    sin,
    cos,
    atan,
    tan,
    relu6,
    silu,
    gelu,
    gelu_exact,
    truncate,
    logarithm_base2,
    logarithm_base10,
    exponent_base10,
    hard_swish,
    identity,
    global_avg_pool,
    global_max_pool,
);
const BINARY: &[(&str, Case)] = binary!(
    addition,
    subtraction,
    multiplication,
    division,
    power,
    maximum,
    minimum,
    floor_divide,
    prelu,
    equal,
    not_equal,
    less_than,
    less_than_or_equal_to,
    greater_than,
    greater_than_or_equal_to,
    reshape_like,
);
const SCALAR: &[(&str, Case)] = scalar!(
    multiply_scalar(2.5),
    add_scalar(-1.0),
    reverse_subtract_scalar(3.0),
    power_scalar(2.0),
    minimum_scalar(0.5),
    maximum_scalar(0.5),
    thresholded_relu(0.5),
    fill_like(0.75),
    leaky_relu(0.1),
    elu(1.0),
    threshold(0.5),
);
const AXIS: &[(&str, Case)] = axis!(
    soft_max,
    reduction_sum,
    reduction_mean,
    reduction_minimum,
    reduction_maximum,
    reduction_sum_square,
    reduction_l1_norm,
    reduction_l2_norm,
    reduction_log_sum,
    reduction_log_sum_exp,
    log_softmax,
    reduction_arg_maximum,
    reduction_arg_minimum,
);
const BOOLEAN: &[(&str, Case)] = boolean!(logical_and, logical_or, logical_xor);

const SPECIAL: &[(&str, Case)] = &[
    ("placeholder_f32", |g| {
        let x = g.placeholder([2, 3], DataType::Float32)?;
        Ok(vec![g.relu(&x)?])
    }),
    ("integer_parameter", |g| {
        Ok(vec![g.placeholder([1, 1, 1, 1], DataType::Int32)?])
    }),
    ("constant", |g| {
        let x = h(g, [2, 3])?;
        let k = c(g, [2, 3])?;
        Ok(vec![g.addition(&x, &k)?])
    }),
    ("constant_f16_bytes", |g| {
        let x = h(g, [2, 3])?;
        let k = g.constant_with_bytes(&[0x3c; 12], [2, 3], DataType::Float16)?;
        Ok(vec![g.addition(&x, &k)?])
    }),
    ("constant_scalar", |g| {
        let x = h(g, [2, 3])?;
        let k = g.constant_with_scalar(2.0, [2, 3])?;
        Ok(vec![g.multiplication(&x, &k)?])
    }),
    ("fill", |g| {
        let x = h(g, [2, 3])?;
        let k = g.constant_with_scalar(1.5, [2, 3])?;
        Ok(vec![g.addition(&x, &k)?])
    }),
    ("range", |g| {
        let x = h(g, [4])?;
        let k = g.range(0.0, 1.0, 4)?;
        Ok(vec![g.addition(&x, &k)?])
    }),
    ("coordinate", |g| {
        let x = h(g, [2, 3])?;
        let k = g.coordinate_along_axis([2, 3], 1)?;
        Ok(vec![g.addition(&x, &k)?])
    }),
    ("boolean_constant", |g| {
        let x = x4(g)?;
        let y = x4(g)?;
        let b = g.boolean_constant(true)?;
        Ok(vec![g.select(&b, &x, &y)?])
    }),
    ("hard_sigmoid", |g| {
        let x = x4(g)?;
        Ok(vec![g.hard_sigmoid(&x, 0.2, 0.5)?])
    }),
    ("linear", |g| {
        let x = x4(g)?;
        Ok(vec![g.linear_activation(&x, 2.0, 1.0)?])
    }),
    ("clip", |g| {
        let x = x4(g)?;
        Ok(vec![g.clamp(&x, -1.0, 1.0)?])
    }),
    ("clamped_relu", |g| {
        let x = x4(g)?;
        Ok(vec![g.clamped_relu(&x, 0.1, 6.0)?])
    }),
    ("scaled_tanh", |g| {
        let x = x4(g)?;
        Ok(vec![g.scaled_tanh(&x, 1.0, 2.0)?])
    }),
    ("softplus_parametric", |g| {
        let x = x4(g)?;
        let a = c(g, [1, 2, 1, 1])?;
        let b = c(g, [1, 2, 1, 1])?;
        Ok(vec![g.softplus_parametric(&x, &a, &b)?])
    }),
    ("logical_not", |g| {
        let b = typed(g, [2, 3], DataType::Bool)?;
        Ok(vec![g.not(&b)?])
    }),
    ("select", |g| {
        let b = typed(g, [1, 2, 4, 4], DataType::Bool)?;
        let x = x4(g)?;
        let y = x4(g)?;
        Ok(vec![g.select(&b, &x, &y)?])
    }),
    ("cast_uint8", |g| {
        let x = x4(g)?;
        Ok(vec![g.cast(&x, DataType::UInt8)?])
    }),
    ("cast_int32", |g| {
        let x = x4(g)?;
        Ok(vec![g.cast(&x, DataType::Int32)?])
    }),
    ("sum", |g| {
        let x = x4(g)?;
        Ok(vec![g.sum(&x, &[-1, -2])?])
    }),
    ("mean", |g| {
        let x = x4(g)?;
        Ok(vec![g.mean(&x, &[-1, -2])?])
    }),
    ("variance", |g| {
        let x = x4(g)?;
        Ok(vec![g.variance(&x, &[-1])?])
    }),
    ("layer_norm", |g| {
        let x = x4(g)?;
        let s = c(g, [4])?;
        let b = c(g, [4])?;
        Ok(vec![g.layer_norm(&x, &[-1], &s, Some(&b), 1e-5)?])
    }),
    ("layer_norm_no_bias", |g| {
        let x = x4(g)?;
        let s = c(g, [4])?;
        Ok(vec![g.layer_norm(&x, &[-1], &s, None, 1e-5)?])
    }),
    ("rms_norm", |g| {
        let x = x4(g)?;
        let s = c(g, [4])?;
        Ok(vec![g.rms_norm(&x, &[-1], &s, 1e-5)?])
    }),
    ("group_norm", |g| {
        let x = x4(g)?;
        let s = c(g, [1, 2, 1, 1])?;
        let b = c(g, [1, 2, 1, 1])?;
        Ok(vec![g.group_norm(&x, 2, &s, Some(&b), 1e-5)?])
    }),
    ("batch_norm", |g| {
        let x = x4(g)?;
        let m = c(g, [1, 2, 1, 1])?;
        let v = c(g, [1, 2, 1, 1])?;
        let s = c(g, [1, 2, 1, 1])?;
        let b = c(g, [1, 2, 1, 1])?;
        Ok(vec![g.batch_norm(&x, &m, &v, &s, &b, 1e-5)?])
    }),
    ("instance_norm", |g| {
        let x = x4(g)?;
        let s = c(g, [1, 2, 1, 1])?;
        let b = c(g, [1, 2, 1, 1])?;
        Ok(vec![g.instance_norm(&x, &s, Some(&b), 1e-5)?])
    }),
    ("l2_normalize", |g| {
        let x = x4(g)?;
        Ok(vec![g.l2_normalize(&x, -1, 1e-5)?])
    }),
    ("local_response_norm", |g| {
        let x = x4(g)?;
        Ok(vec![g.local_response_norm(&x, 3, 1e-4, 0.75, 1.0)?])
    }),
    ("matmul", |g| {
        let a = h(g, [2, 3])?;
        let b = h(g, [3, 4])?;
        Ok(vec![g.matrix_multiplication(&a, &b, false, false)?])
    }),
    ("matmul_transposed", |g| {
        let a = h(g, [3, 2])?;
        let b = h(g, [4, 3])?;
        Ok(vec![g.matrix_multiplication(&a, &b, true, true)?])
    }),
    ("quantized_matmul", |g| {
        let a = typed(g, [2, 3], DataType::Int8)?;
        let b = typed(g, [4, 3], DataType::Int8)?;
        Ok(vec![g.quantized_matmul(&a, &b, 0.1, 0.2, true)?])
    }),
    ("attention", |g| {
        let q = h(g, [1, 2, 4, 8])?;
        let k = h(g, [1, 2, 4, 8])?;
        let v = h(g, [1, 2, 4, 8])?;
        Ok(vec![g.scaled_dot_product_attention(&q, &k, &v, None)?])
    }),
    ("attention_mask", |g| {
        let q = h(g, [1, 2, 4, 8])?;
        let k = h(g, [1, 2, 4, 8])?;
        let v = h(g, [1, 2, 4, 8])?;
        let m = h(g, [1, 1, 4, 4])?;
        Ok(vec![g.scaled_dot_product_attention(
            &q,
            &k,
            &v,
            Some(&m),
        )?])
    }),
    ("rotary", |g| {
        let x = h(g, [1, 2, 4, 8])?;
        let cs = h(g, [4, 8])?;
        let sn = h(g, [4, 8])?;
        Ok(vec![g.rotary_embedding(&x, &cs, &sn, 8)?])
    }),
    ("einsum", |g| {
        let a = h(g, [2, 3])?;
        let b = h(g, [3, 4])?;
        Ok(vec![g.einsum(&[&a, &b], "ij,jk->ik")?])
    }),
    ("einsum_batched", |g| {
        let a = h(g, [2, 3, 5])?;
        let b = h(g, [2, 5, 4])?;
        Ok(vec![g.einsum(&[&a, &b], "bij,bjk->bik")?])
    }),
    ("pooling_2d", |g| {
        let x = x4(g)?;
        Ok(vec![g.pooling_2d(
            &x,
            PoolType::Max,
            &Pooling2dDescriptor::new([2, 2], [2, 2]),
        )?])
    }),
    ("pooling_2d_average", |g| {
        let x = x4(g)?;
        Ok(vec![g.pooling_2d(
            &x,
            PoolType::Average,
            &Pooling2dDescriptor::new([3, 3], [1, 1]),
        )?])
    }),
    ("l2_pool", |g| {
        let x = x4(g)?;
        Ok(vec![g.pooling_2d(
            &x,
            PoolType::L2,
            &Pooling2dDescriptor::new([2, 2], [2, 2]),
        )?])
    }),
    ("max_pool_same", |g| {
        let x = x4(g)?;
        let mut descriptor = Pooling2dDescriptor::new([2, 2], [2, 2]);
        descriptor.pad_mode = PadMode::Same;
        Ok(vec![g.pooling_2d(&x, PoolType::Max, &descriptor)?])
    }),
    ("avg_pool", |g| {
        let x = x4(g)?;
        Ok(vec![g.pooling_2d(
            &x,
            PoolType::Average,
            &Pooling2dDescriptor::new([2, 2], [1, 1]),
        )?])
    }),
    ("pad_constant", |g| {
        let x = x4(g)?;
        Ok(vec![g.pad(&x, 1, 1, 1, 1, PadFillMode::Constant, 0.5)?])
    }),
    ("pad_reflect", |g| {
        let x = x4(g)?;
        Ok(vec![g.pad(&x, 1, 1, 1, 1, PadFillMode::Reflect, 0.0)?])
    }),
    ("reshape", |g| {
        let x = h(g, [2, 3])?;
        Ok(vec![g.reshape(&x, [3, 2])?])
    }),
    ("transpose", |g| {
        let x = x4(g)?;
        Ok(vec![g.transpose(&x, [0, 2, 1, 3])?])
    }),
    ("slice", |g| {
        let x = x4(g)?;
        Ok(vec![g.slice(&x, [0, 0, 1, 1], [1, 2, 2, 2])?])
    }),
    ("strided_slice", |g| {
        let x = h(g, [4, 8])?;
        Ok(vec![g.strided_slice(&x, &[0, 0], &[2, 4], &[2, 2])?])
    }),
    ("concat", |g| {
        let a = h(g, [2, 3])?;
        let b = h(g, [2, 5])?;
        Ok(vec![g.concat(&[&a, &b], 1)?])
    }),
    ("split", |g| {
        let x = h(g, [2, 6])?;
        g.split(&x, &[2, 4], 1)
    }),
    ("flatten_2d", |g| {
        let x = x4(g)?;
        Ok(vec![g.flatten_2d(&x, 2)?])
    }),
    ("expand_dims", |g| {
        let x = h(g, [2, 3])?;
        Ok(vec![g.expand_dims(&x, &[0])?])
    }),
    ("squeeze", |g| {
        let x = h(g, [1, 2, 3])?;
        Ok(vec![g.squeeze(&x, &[0])?])
    }),
    ("stack", |g| {
        let a = h(g, [2, 3])?;
        let b = h(g, [2, 3])?;
        Ok(vec![g.stack(&[&a, &b], 0)?])
    }),
    ("tile", |g| {
        let x = h(g, [2, 3])?;
        Ok(vec![g.tile(&x, &[2, 1])?])
    }),
    ("broadcast_to", |g| {
        let x = h(g, [1, 3])?;
        Ok(vec![g.broadcast_to(&x, [2, 3])?])
    }),
    ("reverse", |g| {
        let x = h(g, [2, 3])?;
        Ok(vec![g.reverse(&x, &[1])?])
    }),
    ("slice_update", |g| {
        let x = h(g, [4, 4])?;
        let u = h(g, [2, 2])?;
        Ok(vec![g.slice_update(&x, &u, &[1, 1])?])
    }),
    ("depth_to_space", |g| {
        let x = h(g, [1, 4, 2, 2])?;
        Ok(vec![g.depth_to_space(&x, 2)?])
    }),
    ("space_to_depth", |g| {
        let x = h(g, [1, 1, 4, 4])?;
        Ok(vec![g.space_to_depth(&x, 2)?])
    }),
    ("pixel_shuffle", |g| {
        let x = h(g, [1, 4, 2, 2])?;
        Ok(vec![g.pixel_shuffle(&x, 2)?])
    }),
    ("pixel_unshuffle", |g| {
        let x = h(g, [1, 1, 4, 4])?;
        Ok(vec![g.pixel_unshuffle(&x, 2)?])
    }),
    ("space_to_batch", |g| {
        let x = h(g, [1, 1, 4, 4])?;
        Ok(vec![g.space_to_batch(&x, [2, 2], [0, 0, 0, 0])?])
    }),
    ("batch_to_space", |g| {
        let x = h(g, [4, 1, 2, 2])?;
        Ok(vec![g.batch_to_space(&x, [2, 2], [0, 0, 0, 0])?])
    }),
    ("crop", |g| {
        let x = x4(g)?;
        Ok(vec![g.crop(&x, [1, 1, 1, 1])?])
    }),
    ("gather", |g| {
        let x = h(g, [4, 8])?;
        let i = typed(g, [3], DataType::UInt16)?;
        Ok(vec![g.gather(&x, &i, 0)?])
    }),
    ("gather_along_axis", |g| {
        let x = h(g, [4, 8])?;
        let i = typed(g, [4, 2], DataType::UInt16)?;
        Ok(vec![g.gather_along_axis(&x, &i, 1)?])
    }),
    ("band_part", |g| {
        let x = x4(g)?;
        Ok(vec![g.band_part(&x, -1, 0)?])
    }),
    ("one_hot", |g| {
        let indices = typed(g, [3], DataType::UInt16)?;
        Ok(vec![g.one_hot(&indices, 5)?])
    }),
    ("gather_nd", |g| {
        let x = h(g, [4, 8])?;
        let i = typed(g, [3, 2], DataType::UInt16)?;
        Ok(vec![g.gather_nd(&x, &i)?])
    }),
    ("top_k", |g| {
        let x = h(g, [4, 8])?;
        let (v, i) = g.top_k(&x, 2, -1)?;
        Ok(vec![v, i])
    }),
    ("bottom_k", |g| {
        let x = h(g, [4, 8])?;
        let (v, i) = g.bottom_k(&x, 3, 0)?;
        Ok(vec![v, i])
    }),
    ("sort", |g| {
        let x = h(g, [4, 8])?;
        Ok(vec![g.sort(&x, -1, true)?])
    }),
    ("argsort", |g| {
        let x = h(g, [4, 8])?;
        Ok(vec![g.argsort(&x, -1, false)?])
    }),
    ("cumulative_sum", |g| {
        let x = h(g, [2, 5])?;
        Ok(vec![g.cumulative_sum(&x, 1, false, false)?])
    }),
    ("cumulative_product", |g| {
        let x = h(g, [2, 5])?;
        Ok(vec![g.cumulative_product(&x, 1, true, false)?])
    }),
    ("cumulative_min", |g| {
        let x = h(g, [2, 5])?;
        Ok(vec![g.cumulative_min(&x, 1, false, true)?])
    }),
    ("cumulative_max", |g| {
        let x = h(g, [2, 5])?;
        Ok(vec![g.cumulative_max(&x, 1, true, true)?])
    }),
    ("reduce_product", |g| {
        let x = h(g, [2, 5])?;
        Ok(vec![g.reduction_product(&x, 1)?])
    }),
    ("quantize", |g| {
        let x = h(g, [2, 3])?;
        Ok(vec![g.quantize(&x, &[0.1], None, None, DataType::Int8)?])
    }),
    ("quantize_axis", |g| {
        let x = h(g, [2, 3])?;
        Ok(vec![g.quantize(
            &x,
            &[0.1, 0.2],
            Some(&[1, 2]),
            Some(0),
            DataType::UInt8,
        )?])
    }),
    ("dequantize", |g| {
        let q = typed(g, [2, 3], DataType::Int8)?;
        Ok(vec![g.dequantize(&q, &[0.1], Some(&[1]), None)?])
    }),
    ("quantize_int4", |g| {
        let x = h(g, [2, 4])?;
        Ok(vec![g.quantize_int4(&x, 0.1)?])
    }),
    ("unpack_int4", |g| {
        let b = typed(g, [2, 2], DataType::UInt8)?;
        Ok(vec![g.unpack_int4(&b)?])
    }),
    ("unpack_signed_int4", |g| {
        let b = typed(g, [2, 2], DataType::UInt8)?;
        Ok(vec![g.unpack_signed_int4(&b)?])
    }),
    ("pack_signed_int4", |g| {
        let x = h(g, [2, 4])?;
        Ok(vec![g.pack_signed_int4(&x)?])
    }),
    ("dequantize_groupwise", |g| {
        let q = h(g, [2, 8])?;
        let s = h(g, [2, 2])?;
        Ok(vec![g.dequantize_groupwise(&q, &s, None, None, 4)?])
    }),
    ("blockwise_weights", |g| {
        let x = h(g, [2, 4])?;
        let w = g.blockwise_weights(
            &[1; 16],
            [4, 4],
            &BlockwiseQuantization::new(WeightDataType::Int8, &[0.1, 0.2, 0.3, 0.4], [4, 1]),
        )?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("palettized_weights", |g| {
        let x = h(g, [2, 4])?;
        let w = g.palettized_weights(&[0x1b; 4], 2, [4, 4], &[0.0, 0.5, 1.0, 1.5])?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("sparse_weights", |g| {
        let x = h(g, [2, 4])?;
        let w = g.sparse_weights(&[0x55, 0x55], [4, 4], &[1.0; 8])?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("sparse_blockwise_weights", |g| {
        let x = h(g, [2, 4])?;
        let w = g.sparse_blockwise_weights(
            &[1; 8],
            &[0x55, 0x55],
            [4, 4],
            &BlockwiseQuantization::new(WeightDataType::Int8, &[0.1, 0.2, 0.3, 0.4], [4, 1]),
        )?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("sparse_palettized_weights", |g| {
        let x = h(g, [2, 4])?;
        let w = g.sparse_palettized_weights(
            &[0x1b; 2],
            &[0x55, 0x55],
            2,
            [4, 4],
            &[0.0, 0.5, 1.0, 1.5],
        )?;
        Ok(vec![g.matrix_multiplication(&x, &w, false, false)?])
    }),
    ("convolution_1x1", |g| {
        let x = x4(g)?;
        let w = c(g, [3, 2, 1, 1])?;
        Ok(vec![g.convolution_2d_1x1(&x, &w, None)?])
    }),
    ("convolution", |g| {
        let x = x4(g)?;
        let w = c(g, [3, 2, 3, 3])?;
        let b = c(g, [3])?;
        Ok(vec![g.convolution_2d(
            &x,
            &w,
            Some(&b),
            &Convolution2dDescriptor {
                pad_mode: PadMode::Same,
                ..Default::default()
            },
        )?])
    }),
    ("convolution_transpose", |g| {
        let x = x4(g)?;
        let w = c(g, [2, 3, 3, 3])?;
        Ok(vec![g.convolution_transpose_2d(
            &x,
            &w,
            None,
            &ConvolutionTranspose2dDescriptor {
                strides: [2, 2],
                ..Default::default()
            },
        )?])
    }),
    ("convolution_1x1_bias", |g| {
        let x = h(g, [1, 2, 1, 1])?;
        let w = c(g, [3, 2, 1, 1])?;
        let b = c(g, [3])?;
        Ok(vec![g.convolution_2d_1x1(&x, &w, Some(&b))?])
    }),
    ("resize", |g| {
        let x = x4(g)?;
        Ok(vec![g.resize(&x, [8, 8], SamplingMode::Bilinear)?])
    }),
    ("upsample", |g| {
        let x = x4(g)?;
        Ok(vec![g.upsample(&x, [2, 2], SamplingMode::Nearest)?])
    }),
    ("sample_grid", |g| {
        let x = x4(g)?;
        let k = h(g, [1, 4, 4, 2])?;
        Ok(vec![g.sample_grid(
            &x,
            &k,
            &SamplingDescriptor::default(),
        )?])
    }),
    ("affine", |g| {
        let x = x4(g)?;
        let m = h(g, [1, 6])?;
        Ok(vec![g.affine(
            &x,
            &m,
            [4, 4],
            &SamplingDescriptor {
                align_corners: true,
                ..Default::default()
            },
        )?])
    }),
    ("crop_resize", |g| {
        let x = x4(g)?;
        let b = h(g, [1, 4])?;
        let i = typed(g, [1], DataType::UInt16)?;
        Ok(vec![g.crop_resize(&x, &b, &i, [2, 2], true)?])
    }),
    ("variable", |g| {
        let x = h(g, [2, 3])?;
        let v = g.variable_with_data(&[0.5; 6], [2, 3])?;
        let r = g.read_variable(&v)?;
        Ok(vec![g.addition(&x, &r)?])
    }),
    ("variable_from_surface", |g| {
        let x = h(g, [2, 3])?;
        let d = TensorData::with_type([2, 3], DataType::Float16)?;
        let v = g.variable_with_tensor_data(&d)?;
        let r = g.read_variable(&v)?;
        Ok(vec![g.addition(&x, &r)?])
    }),
    ("state_assign", |g| {
        let x = h(g, [2, 3])?;
        let s = g.variable_placeholder([2, 3])?;
        let r = g.read_variable(&s)?;
        let y = g.addition(&x, &r)?;
        g.assign_variable(&s, &y)?;
        Ok(vec![g.read_variable(&s)?])
    }),
    ("state_update_rows", |g| {
        let x = h(g, [1, 1, 1, 4])?;
        let p = g.placeholder([1, 1, 1, 1], DataType::Int32)?;
        let s = g.variable_placeholder([1, 1, 8, 4])?;
        g.assign_variable_rows(&s, &x, &p, 0)?;
        Ok(vec![g.read_variable(&s)?])
    }),
    ("state_update_channel", |g| {
        let x = h(g, [1, 1, 1, 4])?;
        let p = g.placeholder([1, 1, 1, 1], DataType::Int32)?;
        let s = g.variable_placeholder([1, 2, 8, 4])?;
        g.assign_variable_rows(&s, &x, &p, 1)?;
        Ok(vec![g.read_variable(&s)?])
    }),
    ("foreign_tensor", |g| {
        let x = x4(g)?;
        let other = Graph::new();
        let y = x4(&other)?;
        Ok(vec![g.addition(&x, &y)?])
    }),
];

#[test]
fn all_operations_lower_to_apple_mlir() {
    for (name, case) in [UNARY, BINARY, SCALAR, AXIS, BOOLEAN, SPECIAL].concat() {
        println!("lowering {name}");
        let graph = Graph::new();
        let output = case(&graph);
        if matches!(name, "foreign_tensor" | "cast_int32") {
            assert!(output.is_err());
            continue;
        }
        let output = output.unwrap_or_else(|error| panic!("{name}: {error}"));
        let program = graph
            .program(&output, &[DataType::Float32])
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let text = program
            .mlir()
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(text.contains("func.func @main"), "{name}: {text}");
        assert!(!text.contains("BLOBFILE"), "{name}: {text}");
    }
}

#[test]
fn invalid_parameter_does_not_change_the_graph() {
    let graph = Graph::new();
    let input = h(&graph, [2, 64]).unwrap();
    for value in [f64::MAX, 65536.0, -65536.0] {
        assert!(matches!(
            graph.threshold(&input, value),
            Err(GraphError::Ir(ane::IrError::InvalidValue("fp16")))
        ));
        assert!(matches!(
            graph.linear_activation(&input, value, 0.0),
            Err(GraphError::Ir(ane::IrError::InvalidValue("fp32")))
        ));
    }
    let output = graph.relu(&input).unwrap();
    let actual = graph.program(&[output], &[DataType::Float16]).unwrap();
    let clean = Graph::new();
    let input = h(&clean, [2, 64]).unwrap();
    let output = clean.relu(&input).unwrap();
    let expected = clean.program(&[output], &[DataType::Float16]).unwrap();
    assert!(actual == expected);
}

#[test]
fn matmul_validation_and_transpose_share_shape_rules() {
    let graph = Graph::new();
    let x = h(&graph, [3, 4]).unwrap();
    let wrong = h(&graph, [5, 2]).unwrap();
    assert!(matches!(
        graph.matrix_multiplication(&x, &wrong, false, false),
        Err(GraphError::Ir(ane::IrError::InvalidProgram(_)))
    ));
    let y = h(&graph, [2, 4]).unwrap();
    let result = graph.matrix_multiplication(&x, &y, false, true).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    graph.program(&[result], &[DataType::Float16]).unwrap();
}

#[test]
fn boolean_gather_data_is_rejected() {
    let graph = Graph::new();
    let data = typed(&graph, [4], DataType::Bool).unwrap();
    let indices = typed(&graph, [2], DataType::UInt16).unwrap();
    assert!(matches!(
        graph.gather(&data, &indices, 0),
        Err(GraphError::UnsupportedDataType(_))
    ));
}

#[test]
fn all_operations_compile_for_ane() {
    let expected = [
        ("fill_like", "the output does not depend on a live input"),
        ("integer_parameter", "Int32 is only a state position type"),
        (
            "boolean_constant",
            "Apple folds the unused select branch input",
        ),
        (
            "variable_from_surface",
            "the bound surface uses an unpadded layout",
        ),
    ];
    let mut failures = Vec::new();
    for (name, case) in [UNARY, BINARY, SCALAR, AXIS, BOOLEAN, SPECIAL].concat() {
        if matches!(name, "foreign_tensor" | "cast_int32") {
            continue;
        }
        let graph = Graph::new();
        let output = case(&graph).unwrap();
        if let Err(error) = graph.compile(&output, &[], None) {
            failures.push(name);
            assert!(
                expected.iter().any(|(case, _)| *case == name),
                "{name}: {error}"
            );
        }
    }
    assert_eq!(failures, expected.map(|(name, _)| name));
}

#[test]
fn constants_must_fit_fp16_unless_explicitly_infinite() {
    let graph = Graph::new();
    assert!(matches!(
        graph.constant(&[70_000.0], [1]),
        Err(GraphError::Ir(ane::IrError::InvalidValue("fp16")))
    ));
    assert!(graph.constant(&[f32::NEG_INFINITY, 65_504.0], [2]).is_ok());
}
