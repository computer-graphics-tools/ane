use crate::BlockwiseQuantization;
use crate::DataType;
use crate::WeightDataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, checked_shape, ensure};
use crate::ir::{Operator, Parameter, Value, WeightBlob};
use crate::logical_shape;

impl Graph {
    fn quantization_scale(
        &self,
        input: &Tensor,
        scale: &Tensor,
        axis: Option<i64>,
    ) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        self.numeric(*scale)?;
        let count = scale.physical_shape().iter().product::<usize>();
        let mut shape = [1; 4];
        if let Some(axis) = axis {
            let axis = self.axis(*input, axis)?;
            ensure(
                count == 1 || (scale.rank() == 1 && count == input.physical_shape()[axis]),
                GraphError::ShapeMismatch("quantization scale length differs from its axis"),
            )?;
            shape[axis] = count;
        } else {
            ensure(
                count == 1,
                GraphError::ShapeMismatch("per-tensor quantization needs a scalar scale"),
            )?;
        }
        self.reshape_to(*scale, &shape)
    }
    pub fn quantize_with_scale_tensor(
        &self,
        input: &Tensor,
        scale: &Tensor,
        zero_point: i32,
        axis: Option<i64>,
        dtype: DataType,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        let (min, max) = match dtype {
            DataType::Int8 => (-128, 127),
            DataType::UInt8 => (0, 255),
            _ => {
                return Err(GraphError::UnsupportedDataType(
                    "quantization output requires Int8 or UInt8",
                ));
            }
        };
        ensure(
            (min..=max).contains(&zero_point),
            GraphError::InvalidArgument("quantization zero point is outside its storage range"),
        )?;
        let scale = self.quantization_scale(input, scale, axis)?;
        let value = self.division(input, &scale)?;
        let value = self.add_scalar(&value, zero_point as f32)?;
        let value = self.clamp(&value, min as f32, max as f32)?;
        let value = self.round(&value)?;
        self.quantize(&value, &[1.0], Some(&[0]), None, dtype)
    }
    pub fn dequantize_with_scale_tensor(
        &self,
        input: &Tensor,
        scale: &Tensor,
        zero_point: i32,
        axis: Option<i64>,
    ) -> Result<Tensor, GraphError> {
        let (min, max) = match input.data_type() {
            DataType::Int8 => (-128, 127),
            DataType::UInt8 => (0, 255),
            _ => {
                return Err(GraphError::UnsupportedDataType(
                    "dequantization input requires Int8 or UInt8",
                ));
            }
        };
        ensure(
            (min..=max).contains(&zero_point),
            GraphError::InvalidArgument("quantization zero point is outside its storage range"),
        )?;
        let scale = self.quantization_scale(input, scale, axis)?;
        let value = self.cast(input, DataType::Float16)?;
        let value = self.add_scalar(&value, -zero_point as f32)?;
        self.multiplication(&value, &scale)
    }
    pub fn unpack_int4(&self, bytes: &Tensor) -> Result<Tensor, GraphError> {
        let bytes = *bytes;
        self.check_tensor(bytes)?;
        let bytes = if bytes.data_type() == DataType::Float16 {
            bytes
        } else {
            self.cast(&bytes, DataType::Float16)?
        };
        let divided = self.multiply_scalar(&bytes, 1.0 / 16.0)?;
        let high = self.floor(&divided)?;
        let shifted = self.multiply_scalar(&high, 16.0)?;
        let low = self.subtraction(&bytes, &shifted)?;
        let lanes = [
            bytes.physical_shape()[0]
                .checked_mul(bytes.physical_shape()[1])
                .ok_or(GraphError::Overflow)?,
            bytes.physical_shape()[2],
            bytes.physical_shape()[3],
            1,
        ];
        let low = self.reshape_to(low, &lanes)?;
        let high = self.reshape_to(high, &lanes)?;
        let pairs = self.concat(&[&low, &high], 3)?;
        self.reshape_to(
            pairs,
            &[
                bytes.physical_shape()[0],
                bytes.physical_shape()[1],
                bytes.physical_shape()[2],
                bytes.physical_shape()[3]
                    .checked_mul(2)
                    .ok_or(GraphError::Overflow)?,
            ][4 - bytes.rank()..],
        )
    }

    pub fn unpack_signed_int4(&self, bytes: &Tensor) -> Result<Tensor, GraphError> {
        self.check_tensor(*bytes)?;
        let codes = self.unpack_int4(bytes)?;
        let sign = self.add_scalar(&codes, -7.0)?;
        let sign = self.clamp(&sign, 0.0, 1.0)?;
        let correction = self.multiply_scalar(&sign, 16.0)?;
        self.subtraction(&codes, &correction)
    }

    pub fn pack_signed_int4(&self, codes: &Tensor) -> Result<Tensor, GraphError> {
        self.check_tensor(*codes)?;
        ensure(
            codes.physical_shape()[3].is_multiple_of(2),
            GraphError::ShapeMismatch("packing needs an even width"),
        )?;
        let negative = self.multiply_scalar(codes, -1.0)?;
        let negative = self.clamp(&negative, 0.0, 1.0)?;
        let correction = self.multiply_scalar(&negative, 16.0)?;
        let unsigned = self.addition(codes, &correction)?;
        let lanes = [
            codes.physical_shape()[0]
                .checked_mul(codes.physical_shape()[1])
                .ok_or(GraphError::Overflow)?,
            codes.physical_shape()[2],
            codes.physical_shape()[3] / 2,
            2,
        ];
        let pairs = self.reshape_to(unsigned, &lanes)?;
        let size = [lanes[0], lanes[1], lanes[2], 1];
        let low = self.slice(&pairs, [0, 0, 0, 0], size)?;
        let high = self.slice(&pairs, [0, 0, 0, 1], size)?;
        let high = self.multiply_scalar(&high, 16.0)?;
        let bytes = self.addition(&low, &high)?;
        let bytes = self.quantize(&bytes, &[1.0], Some(&[0]), None, DataType::UInt8)?;
        self.reshape_to(
            bytes,
            &[
                codes.physical_shape()[0],
                codes.physical_shape()[1],
                codes.physical_shape()[2],
                codes.physical_shape()[3] / 2,
            ][4 - codes.rank()..],
        )
    }

    pub fn dequantize_groupwise(
        &self,
        codes: &Tensor,
        scales: &Tensor,
        zero_points: Option<&Tensor>,
        biases: Option<&Tensor>,
        group_size: usize,
    ) -> Result<Tensor, GraphError> {
        let scales = *scales;
        let biases = biases.copied();
        self.check_tensor(*codes)?;
        self.check_tensor(scales)?;
        if let Some(tensor) = zero_points.copied() {
            self.check_tensor(tensor)?;
        }
        if let Some(tensor) = biases {
            self.check_tensor(tensor)?;
        }
        ensure(
            codes.physical_shape()[0] == 1 && codes.physical_shape()[1] == 1,
            GraphError::ShapeMismatch("codes must be [1,1,N,K]"),
        )?;
        ensure(
            group_size > 0 && codes.physical_shape()[3].is_multiple_of(group_size),
            GraphError::ShapeMismatch("group size must divide K"),
        )?;
        let groups = codes.physical_shape()[3] / group_size;
        let metadata = [1, codes.physical_shape()[2], groups, 1];
        ensure(
            (scales.physical_shape().iter().product::<usize>())
                == (metadata.iter().product::<usize>()),
            GraphError::ShapeMismatch("scale count differs"),
        )?;
        let shape = [metadata[0], metadata[1], metadata[2], group_size];
        let mut values = self.reshape_to(*codes, &shape)?;
        if let Some(offsets) = zero_points.copied() {
            ensure(
                (offsets.physical_shape().iter().product::<usize>())
                    == (metadata.iter().product::<usize>()),
                GraphError::ShapeMismatch("zero-point count differs"),
            )?;
            let offsets = self.reshape_to(offsets, &metadata)?;
            values = self.subtraction(&values, &offsets)?;
        }
        let scales = self.reshape_to(scales, &metadata)?;
        values = self.multiplication(&values, &scales)?;
        if let Some(biases) = biases {
            ensure(
                (biases.physical_shape().iter().product::<usize>())
                    == (metadata.iter().product::<usize>()),
                GraphError::ShapeMismatch("bias count differs"),
            )?;
            let biases = self.reshape_to(biases, &metadata)?;
            values = self.addition(&values, &biases)?;
        }
        self.reshape_to(values, codes.shape())
    }

    fn quantization_parameters(
        &self,
        input: Tensor,
        scales: &[f32],
        zeros: Option<&[i32]>,
        axis: Option<i64>,
        dtype: DataType,
    ) -> Result<Vec<(Parameter, Value)>, GraphError> {
        self.check_tensor(input)?;
        ensure(
            matches!(dtype, DataType::Int8 | DataType::UInt8),
            GraphError::UnsupportedDataType("quantization requires Int8 or UInt8 storage"),
        )?;
        ensure(
            !scales.is_empty()
                && scales.iter().all(|&s| {
                    s.is_finite()
                        && s > 0.0
                        && half::f16::from_f32(s).is_finite()
                        && half::f16::from_f32(s).to_f32() > 0.0
                }),
            GraphError::InvalidArgument("quantization scales must be finite and positive in FP16"),
        )?;
        let axis = axis.map(|axis| self.axis(input, axis)).transpose()?;
        ensure(
            axis.map_or(scales.len() == 1, |a| {
                scales.len() == input.physical_shape()[a]
            }),
            GraphError::ShapeMismatch("quantization scale count differs from axis"),
        )?;
        let scalar = axis.is_none();
        let mut attrs = vec![(
            Parameter::Scale,
            if scalar {
                Value::Fp16(scales[0])
            } else {
                Value::Fp16List(scales.into())
            },
        )];
        if let Some(axis) = axis {
            attrs.push((Parameter::Axis, Value::Int32(axis)));
        }
        if let Some(zeros) = zeros {
            ensure(
                zeros.len() == scales.len()
                    && zeros.iter().all(|&z| {
                        if dtype == DataType::Int8 {
                            (-128..128).contains(&z)
                        } else {
                            (0..256).contains(&z)
                        }
                    }),
                GraphError::InvalidArgument("quantization zero points have wrong count or range"),
            )?;
            attrs.push((
                Parameter::ZeroPoint,
                if scalar {
                    Value::Integer(dtype, zeros[0])
                } else {
                    Value::IntegerList(dtype, zeros.into())
                },
            ));
        }
        Ok(attrs)
    }

    pub fn quantize(
        &self,
        input: &Tensor,
        scales: &[f32],
        zeros: Option<&[i32]>,
        axis: Option<i64>,
        dtype: DataType,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        self.quantization(*input, scales, zeros, axis, dtype)
    }

    pub fn dequantize(
        &self,
        input: &Tensor,
        scales: &[f32],
        zeros: Option<&[i32]>,
        axis: Option<i64>,
    ) -> Result<Tensor, GraphError> {
        self.quantization(*input, scales, zeros, axis, DataType::Float16)
    }

    fn quantization(
        &self,
        input: Tensor,
        scales: &[f32],
        zeros: Option<&[i32]>,
        axis: Option<i64>,
        output_type: DataType,
    ) -> Result<Tensor, GraphError> {
        let quantize = output_type != DataType::Float16;
        let integer_type = if quantize {
            output_type
        } else {
            input.data_type()
        };
        let mut attrs = self.quantization_parameters(input, scales, zeros, axis, integer_type)?;
        if quantize {
            attrs.push((Parameter::OutputDtype, Value::String(output_type.as_str())));
        }
        let mut permutation = [0, 1, 2, 3];
        if let Some(axis) = axis {
            permutation.swap(1, self.axis(input, axis)?);
            for (key, value) in &mut attrs {
                if *key == Parameter::Axis {
                    *value = Value::Int32(1);
                }
            }
        }
        let reordered = permutation != [0, 1, 2, 3];
        let source = if reordered {
            let physical = self.reshape_to(input, &input.physical_shape())?;
            self.transpose(&physical, permutation)?
        } else {
            input
        };
        let output = self.builtin(
            if quantize {
                Operator::Quantize
            } else {
                Operator::Dequantize
            },
            &[(Parameter::Input, source)],
            &attrs,
            source.shape(),
            output_type,
        )?;
        if reordered {
            let restored = self.transpose(&output, permutation)?;
            self.reshape_to(restored, input.shape())
        } else {
            Ok(output)
        }
    }

    pub fn quantized_matmul(
        &self,
        left: &Tensor,
        right: &Tensor,
        left_scale: f32,
        right_scale: f32,
        transpose_right: bool,
    ) -> Result<Tensor, GraphError> {
        let left = *left;
        let right = *right;
        let left = self.dequantize(&left, &[left_scale], None, None)?;
        let right = self.dequantize(&right, &[right_scale], None, None)?;
        self.matrix_multiplication(&left, &right, false, transpose_right)
    }

    pub fn quantize_int4(&self, input: &Tensor, scale: f32) -> Result<Tensor, GraphError> {
        ensure(
            scale.is_finite() && scale > 0.0,
            GraphError::InvalidArgument("quantization scale must be positive and finite"),
        )?;
        let scaled = self.multiply_scalar(input, scale.recip())?;
        let rounded = self.round(&scaled)?;
        let clipped = self.clamp(&rounded, -8.0, 7.0)?;
        let packed = self.pack_signed_int4(&clipped)?;
        self.cast(&packed, DataType::UInt8)
    }

    fn constant_operation(
        &self,
        operation: Operator,
        shape: &[usize],
        blobs: Vec<(Parameter, Box<[usize]>, WeightBlob)>,
        attributes: &[(Parameter, Value)],
    ) -> Result<Tensor, GraphError> {
        let blobs = blobs.into_iter().map(Into::into).collect();
        Ok(self.state().builtin_many(
            operation,
            &[],
            attributes,
            &[(DataType::Float16, shape)],
            blobs,
        )?[0])
    }

    pub fn blockwise_weights<const RANK: usize>(
        &self,
        packed: &[u8],
        shape: [usize; RANK],
        config: &BlockwiseQuantization<'_, RANK>,
    ) -> Result<Tensor, GraphError> {
        let shape = logical_shape(&shape);
        let BlockwiseQuantization {
            data_type: dtype,
            scales,
            scale_shape,
            offsets,
        } = *config;
        ensure(
            matches!(
                dtype,
                WeightDataType::Int4
                    | WeightDataType::UInt4
                    | WeightDataType::Int8
                    | WeightDataType::UInt8
            ),
            GraphError::UnsupportedDataType("blockwise weights require 4-bit or 8-bit integers"),
        )?;
        let physical = checked_shape(shape)?;
        let scale_dimensions = checked_shape(&scale_shape)?;
        ensure(
            (0..4).all(|a| physical[a].is_multiple_of(scale_dimensions[a]))
                && scales.len() == scale_dimensions.iter().product::<usize>()
                && scales.iter().all(|v| v.is_finite()),
            GraphError::ShapeMismatch("blockwise scales do not divide the weight shape"),
        )?;
        let data = WeightBlob::from_bytes(packed, physical.iter().product(), dtype)?;
        let mut blobs = vec![
            (Parameter::Data, physical.to_vec().into(), data),
            (
                Parameter::Scale,
                scale_dimensions.to_vec().into(),
                WeightBlob::from_f32(scales)?,
            ),
        ];
        if let Some(offsets) = offsets {
            ensure(
                offsets.len() == scales.len() && offsets.iter().all(|v| v.is_finite()),
                GraphError::ShapeMismatch("blockwise offsets differ from scales"),
            )?;
            blobs.push((
                Parameter::Offset,
                scale_dimensions.to_vec().into(),
                WeightBlob::from_f32(offsets)?,
            ));
        }
        self.constant_operation(Operator::ConstexprBlockwiseShiftScale, shape, blobs, &[])
    }

    pub fn palettized_weights<const RANK: usize>(
        &self,
        indices: &[u8],
        bits: usize,
        shape: [usize; RANK],
        palette: &[f32],
    ) -> Result<Tensor, GraphError> {
        self.palettized_weights_with_groups(indices, bits, shape, [1; RANK], palette)
    }
    pub fn palettized_weights_with_groups<const RANK: usize>(
        &self,
        indices: &[u8],
        bits: usize,
        shape: [usize; RANK],
        group_shape: [usize; RANK],
        palette: &[f32],
    ) -> Result<Tensor, GraphError> {
        let shape = logical_shape(&shape);
        let physical = checked_shape(shape)?;
        let groups = checked_shape(logical_shape(&group_shape))?;
        let dtype = match bits {
            1 => WeightDataType::UInt1,
            2 => WeightDataType::UInt2,
            3 => WeightDataType::UInt3,
            4 => WeightDataType::UInt4,
            6 => WeightDataType::UInt6,
            8 => WeightDataType::UInt8,
            _ => {
                return Err(GraphError::UnsupportedDataType(
                    "palette indices require 1, 2, 3, 4, 6 or 8 bits",
                ));
            }
        };
        ensure(
            groups.iter().product::<usize>().checked_mul(1usize << bits) == Some(palette.len())
                && (0..4).all(|axis| physical[axis].is_multiple_of(groups[axis]))
                && palette.iter().all(|v| v.is_finite()),
            GraphError::ShapeMismatch(
                "palette groups must divide the weight shape and contain complete codebooks",
            ),
        )?;
        self.constant_operation(
            Operator::ConstexprLutToDense,
            shape,
            vec![
                (
                    Parameter::Indices,
                    physical.to_vec().into(),
                    WeightBlob::from_bytes(indices, physical.iter().product(), dtype)?,
                ),
                (
                    Parameter::Lut,
                    vec![
                        groups[0],
                        groups[1],
                        groups[2],
                        groups[3],
                        1usize << bits,
                        1,
                    ]
                    .into(),
                    WeightBlob::from_f32(palette)?,
                ),
            ],
            &[],
        )
    }

    pub fn sparse_weights<const RANK: usize>(
        &self,
        mask: &[u8],
        shape: [usize; RANK],
        nonzero: &[f32],
    ) -> Result<Tensor, GraphError> {
        let shape = logical_shape(&shape);
        let physical = checked_shape(shape)?;
        let count = physical.iter().product();
        let mask_blob = WeightBlob::from_bytes(mask, count, WeightDataType::UInt1)?;
        let set = (0..count)
            .filter(|&i| mask[i / 8] & (1 << (i % 8)) != 0)
            .count();
        ensure(
            set == nonzero.len(),
            GraphError::ShapeMismatch("sparse mask population differs from nonzero count"),
        )?;
        if set == 0 {
            return self.constant_scalar(0.0, shape);
        }
        self.constant_operation(
            Operator::ConstexprSparseToDense,
            shape,
            vec![
                (Parameter::Mask, physical.to_vec().into(), mask_blob),
                (
                    Parameter::NonzeroData,
                    vec![set].into(),
                    WeightBlob::from_f32(nonzero)?,
                ),
            ],
            &[],
        )
    }

    pub fn sparse_blockwise_weights<const RANK: usize>(
        &self,
        packed: &[u8],
        mask: &[u8],
        shape: [usize; RANK],
        config: &BlockwiseQuantization<'_, RANK>,
    ) -> Result<Tensor, GraphError> {
        let shape = logical_shape(&shape);
        let physical = checked_shape(shape)?;
        let scale_shape = checked_shape(&config.scale_shape)?;
        let count = physical.iter().product();
        let mask = WeightBlob::from_bytes(mask, count, WeightDataType::UInt1)?;
        let nonzero = (0..count)
            .filter(|&i| mask.bytes()[i / 8] & (1 << (i % 8)) != 0)
            .count();
        ensure(
            matches!(
                config.data_type,
                WeightDataType::Int4
                    | WeightDataType::UInt4
                    | WeightDataType::Int8
                    | WeightDataType::UInt8
            ),
            GraphError::UnsupportedDataType("sparse quantization requires 4-bit or 8-bit weights"),
        )?;
        ensure(
            (0..4).all(|i| physical[i].is_multiple_of(scale_shape[i]))
                && config.scales.len() == scale_shape.iter().product::<usize>()
                && config.scales.iter().all(|v| v.is_finite()),
            GraphError::ShapeMismatch("sparse quantization scales do not divide the weight shape"),
        )?;
        let data = WeightBlob::from_bytes(packed, nonzero, config.data_type)?;
        if nonzero == 0 {
            return self.constant_scalar(0.0, shape);
        }
        let mut blobs = vec![
            (Parameter::DataMask, physical.to_vec().into(), mask),
            (Parameter::NonzeroData, vec![nonzero].into(), data),
            (
                Parameter::Scale,
                scale_shape.to_vec().into(),
                WeightBlob::from_f32(config.scales)?,
            ),
        ];
        if let Some(offsets) = config.offsets {
            ensure(
                offsets.len() == config.scales.len() && offsets.iter().all(|v| v.is_finite()),
                GraphError::ShapeMismatch("sparse quantization offsets differ from scales"),
            )?;
            blobs.push((
                Parameter::Offset,
                scale_shape.to_vec().into(),
                WeightBlob::from_f32(offsets)?,
            ));
        }
        self.constant_operation(Operator::SparseBlockwiseWeights, shape, blobs, &[])
    }

    pub fn sparse_palettized_weights<const RANK: usize>(
        &self,
        indices: &[u8],
        mask: &[u8],
        bits: usize,
        shape: [usize; RANK],
        palette: &[f32],
    ) -> Result<Tensor, GraphError> {
        let shape = logical_shape(&shape);
        let physical = checked_shape(shape)?;
        let count = physical.iter().product();
        let mask = WeightBlob::from_bytes(mask, count, WeightDataType::UInt1)?;
        let nonzero = (0..count)
            .filter(|&i| mask.bytes()[i / 8] & (1 << (i % 8)) != 0)
            .count();
        let dtype = match bits {
            1 => WeightDataType::UInt1,
            2 => WeightDataType::UInt2,
            3 => WeightDataType::UInt3,
            4 => WeightDataType::UInt4,
            6 => WeightDataType::UInt6,
            8 => WeightDataType::UInt8,
            _ => {
                return Err(GraphError::UnsupportedDataType(
                    "palette indices require 1, 2, 3, 4, 6 or 8 bits",
                ));
            }
        };
        ensure(
            palette.len() == 1usize << bits && palette.iter().all(|v| v.is_finite()),
            GraphError::ShapeMismatch("palette size differs from index bit width"),
        )?;
        let data = WeightBlob::from_bytes(indices, nonzero, dtype)?;
        if nonzero == 0 {
            return self.constant_scalar(0.0, shape);
        }
        self.constant_operation(
            Operator::SparsePaletteWeights,
            shape,
            vec![
                (Parameter::IndicesMask, physical.to_vec().into(), mask),
                (Parameter::IndicesNonzeroData, vec![nonzero].into(), data),
                (
                    Parameter::Lut,
                    vec![1, 1, 1, 1, palette.len(), 1].into(),
                    WeightBlob::from_f32(palette)?,
                ),
            ],
            &[],
        )
    }
}
