use crate::BlockwiseQuantization;
use crate::DataType;
use crate::WeightDataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, checked_shape, ensure};
use crate::ir::{ConstantInput, Operator, Parameter, Value, WeightBlob};
use crate::logical_shape;
use crate::ops::{Palette, Palettization};

impl Graph {
    fn quantization_parameters(
        &self,
        input: Tensor,
        scales: &[f32],
        zeros: Option<&[i32]>,
        axis: Option<i64>,
        dtype: DataType,
    ) -> Result<Vec<(Parameter, Value)>, GraphError> {
        self.check_tensor(input)?;
        byte_range(dtype, 0)?;
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
                zeros.len() == scales.len(),
                GraphError::InvalidArgument("quantization zero point count differs from scales"),
            )?;
            for &zero in zeros {
                byte_range(dtype, zero)?;
            }
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

    /// `round(x / scale) + zero_point` stored as Int8 or UInt8, per tensor or per `axis`. The ANE
    /// compiles per-axis scales only on the channel axis. MIL `quantize`.
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

    /// `(x - zero_point) · scale` from Int8 or UInt8, per tensor or per `axis`. A constant input
    /// becomes a compile-time `constexpr_blockwise_shift_scale`. MIL `dequantize`.
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
        self.builtin(
            if quantize {
                Operator::Quantize
            } else {
                Operator::Dequantize
            },
            &[(Parameter::Input, input)],
            &attrs,
            input.shape(),
            output_type,
        )
    }

    fn constant_operation(
        &self,
        operation: Operator,
        shape: &[usize],
        blobs: Vec<ConstantInput>,
        attributes: &[(Parameter, Value)],
    ) -> Result<Tensor, GraphError> {
        let blobs = blobs.into();
        Ok(self.state().builtin_many(
            operation,
            &[],
            attributes,
            &[(DataType::Float16, shape)],
            blobs,
        )?[0])
    }

    /// Constant weights stored as packed 4-bit or 8-bit integers and dequantized per block.
    /// MIL `constexpr_blockwise_shift_scale`.
    pub fn blockwise_weights<const RANK: usize>(
        &self,
        packed: &[u8],
        shape: [usize; RANK],
        config: &BlockwiseQuantization<'_, RANK>,
    ) -> Result<Tensor, GraphError> {
        let shape = logical_shape(&shape);
        let physical = checked_shape(shape)?;
        let dtype = config.data_type;
        let scales = scale_blobs(physical, config)?;
        let data = WeightBlob::from_bytes(packed, physical.iter().product(), dtype)?;
        let mut blobs: Vec<ConstantInput> =
            vec![(Parameter::Data, physical.to_vec().into(), data).into()];
        blobs.extend(scales);
        self.constant_operation(Operator::ConstexprBlockwiseShiftScale, shape, blobs, &[])
    }

    /// Constant weights stored as 1-, 2-, 3-, 4-, 6- or 8-bit palette indices.
    /// MIL `constexpr_lut_to_dense`; an integer palette is dequantized first by
    /// `constexpr_blockwise_shift_scale`.
    pub fn palettized_weights<const RANK: usize>(
        &self,
        indices: &[u8],
        shape: [usize; RANK],
        config: &Palettization<'_, RANK>,
    ) -> Result<Tensor, GraphError> {
        let shape = logical_shape(&shape);
        let physical = checked_shape(shape)?;
        let groups = checked_shape(logical_shape(&config.group_shape))?;
        let index_type = palette_type(config.bits)?;
        let entries = 1usize << config.bits;
        let group_count: usize = groups.iter().product();
        let (values, data_type) = match config.palette {
            Palette::Float(values) => {
                ensure(
                    values.iter().all(|v| v.is_finite()),
                    GraphError::InvalidArgument("palette values must be finite"),
                )?;
                (values.len(), WeightDataType::Float16)
            }
            Palette::Quantized {
                data_type, codes, ..
            } => {
                ensure(
                    matches!(data_type, WeightDataType::Int8 | WeightDataType::UInt8),
                    GraphError::UnsupportedDataType("quantized palettes use Int8 or UInt8 codes"),
                )?;
                (codes.len(), data_type)
            }
        };
        let vector = values / (group_count * entries).max(1);
        let mut index_shape = physical;
        let mut attributes = Vec::new();
        if let Some(axis) = config.vector_axis {
            ensure(
                axis < RANK,
                GraphError::InvalidAxes("palette vector axis exceeds the weight rank"),
            )?;
            let axis = axis + 4 - RANK;
            ensure(
                physical[axis].is_multiple_of(vector),
                GraphError::ShapeMismatch("palette vectors must divide the vector axis"),
            )?;
            index_shape[axis] /= vector;
            attributes.push((Parameter::VectorAxis, Value::Int32(axis)));
        }
        ensure(
            values == group_count * entries * vector
                && (config.vector_axis.is_some() || vector == 1)
                && (0..4).all(|axis| physical[axis].is_multiple_of(groups[axis])),
            GraphError::ShapeMismatch(
                "palette groups must divide the weight shape and contain complete codebooks",
            ),
        )?;
        let lut_shape: Box<[usize]> =
            [groups[0], groups[1], groups[2], groups[3], entries, vector].into();
        let indices = WeightBlob::from_bytes(indices, index_shape.iter().product(), index_type)?;
        let mut blobs: Vec<ConstantInput> =
            vec![(Parameter::Indices, index_shape.to_vec().into(), indices).into()];
        match config.palette {
            Palette::Float(palette) => {
                blobs.push((Parameter::Lut, lut_shape, WeightBlob::from_f32(palette)?).into());
            }
            Palette::Quantized {
                codes,
                scales,
                zero_points,
                ..
            } => {
                ensure(
                    scales.len() == group_count && scales.iter().all(|v| v.is_finite()),
                    GraphError::ShapeMismatch("quantized palettes need one finite scale per group"),
                )?;
                let scale_shape: Box<[usize]> =
                    [groups[0], groups[1], groups[2], groups[3], 1, 1].into();
                blobs.push(
                    (
                        Parameter::Lut,
                        lut_shape,
                        WeightBlob::from_bytes(codes, values, data_type)?,
                    )
                        .into(),
                );
                blobs.push(
                    (
                        Parameter::LutScale,
                        scale_shape.clone(),
                        WeightBlob::from_f32(scales)?,
                    )
                        .into(),
                );
                if let Some(zero_points) = zero_points {
                    blobs.push(
                        (
                            Parameter::LutOffset,
                            scale_shape,
                            WeightBlob::from_bytes(zero_points, group_count, data_type)?,
                        )
                            .into(),
                    );
                }
            }
        }
        self.constant_operation(Operator::ConstexprLutToDense, shape, blobs, &attributes)
    }

    /// Constant weights stored as a bit mask and the Float16 values of its set bits.
    /// MIL `constexpr_sparse_to_dense`.
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
        let set = mask_population(&mask_blob)?;
        ensure(
            set == nonzero.len(),
            GraphError::ShapeMismatch("sparse mask population differs from nonzero count"),
        )?;
        self.constant_operation(
            Operator::ConstexprSparseToDense,
            shape,
            vec![
                (Parameter::Mask, physical.to_vec().into(), mask_blob).into(),
                (
                    Parameter::NonzeroData,
                    vec![set].into(),
                    WeightBlob::from_f32(nonzero)?,
                )
                    .into(),
            ],
            &[],
        )
    }

    /// Sparse constant weights whose nonzero values are blockwise-quantized integers.
    /// MIL `constexpr_sparse_blockwise_shift_scale` followed by `constexpr_sparse_to_dense`.
    pub fn sparse_blockwise_weights<const RANK: usize>(
        &self,
        packed: &[u8],
        mask: &[u8],
        shape: [usize; RANK],
        config: &BlockwiseQuantization<'_, RANK>,
    ) -> Result<Tensor, GraphError> {
        let shape = logical_shape(&shape);
        let physical = checked_shape(shape)?;
        let count = physical.iter().product();
        let mask = WeightBlob::from_bytes(mask, count, WeightDataType::UInt1)?;
        let nonzero = mask_population(&mask)?;
        let scales = scale_blobs(physical, config)?;
        let data = WeightBlob::from_bytes(packed, nonzero, config.data_type)?;
        let mut blobs: Vec<ConstantInput> = vec![
            (Parameter::DataMask, physical.to_vec().into(), mask).into(),
            (Parameter::NonzeroData, vec![nonzero].into(), data).into(),
        ];
        blobs.extend(scales);
        self.constant_operation(
            Operator::ConstexprSparseBlockwiseShiftScale,
            shape,
            blobs,
            &[],
        )
    }

    /// Sparse constant weights whose nonzero values are palette indices.
    /// MIL `constexpr_lut_to_sparse` followed by `constexpr_sparse_to_dense`.
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
        let nonzero = mask_population(&mask)?;
        let dtype = palette_type(bits)?;
        ensure(
            palette.len() == 1usize << bits && palette.iter().all(|v| v.is_finite()),
            GraphError::ShapeMismatch("palette size differs from index bit width"),
        )?;
        let data = WeightBlob::from_bytes(indices, nonzero, dtype)?;
        self.constant_operation(
            Operator::ConstexprLutToSparse,
            shape,
            vec![
                (Parameter::IndicesMask, physical.to_vec().into(), mask).into(),
                (Parameter::IndicesNonzeroData, vec![nonzero].into(), data).into(),
                (
                    Parameter::Lut,
                    vec![1, 1, 1, 1, palette.len(), 1].into(),
                    WeightBlob::from_f32(palette)?,
                )
                    .into(),
            ],
            &[],
        )
    }
}

fn byte_range(dtype: DataType, zero_point: i32) -> Result<(), GraphError> {
    let range = match dtype {
        DataType::Int8 => -128..=127,
        DataType::UInt8 => 0..=255,
        _ => {
            return Err(GraphError::UnsupportedDataType(
                "quantization requires Int8 or UInt8 storage",
            ));
        }
    };
    ensure(
        range.contains(&zero_point),
        GraphError::InvalidArgument("quantization zero point is outside its storage range"),
    )
}

fn palette_type(bits: usize) -> Result<WeightDataType, GraphError> {
    Ok(match bits {
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
    })
}

fn mask_population(mask: &WeightBlob) -> Result<usize, GraphError> {
    let count = (0..mask.element_count())
        .filter(|&i| mask.bytes()[i / 8] & (1 << (i % 8)) != 0)
        .count();
    ensure(
        count > 0,
        GraphError::InvalidArgument("sparse weights need at least one nonzero element"),
    )?;
    Ok(count)
}

fn scale_blobs<const RANK: usize>(
    physical: [usize; 4],
    config: &BlockwiseQuantization<'_, RANK>,
) -> Result<Vec<ConstantInput>, GraphError> {
    let BlockwiseQuantization {
        data_type: dtype,
        scales,
        scale_shape,
        offsets,
        zero_points,
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
    let blocks = checked_shape(logical_shape(&scale_shape))?;
    ensure(
        (0..4).all(|a| physical[a].is_multiple_of(blocks[a]))
            && scales.len() == blocks.iter().product::<usize>()
            && scales.iter().all(|v| v.is_finite()),
        GraphError::ShapeMismatch("blockwise scales do not divide the weight shape"),
    )?;
    let mut blobs = vec![
        (
            Parameter::Scale,
            blocks.to_vec().into(),
            WeightBlob::from_f32(scales)?,
        )
            .into(),
    ];
    let offset = match (offsets, zero_points) {
        (Some(_), Some(_)) => {
            return Err(GraphError::InvalidArgument(
                "blockwise weights take float offsets or integer zero points, not both",
            ));
        }
        (Some(offsets), None) => {
            ensure(
                offsets.len() == scales.len() && offsets.iter().all(|v| v.is_finite()),
                GraphError::ShapeMismatch("blockwise offsets differ from scales"),
            )?;
            Some(WeightBlob::from_f32(offsets)?)
        }
        (None, Some(zero_points)) => {
            Some(WeightBlob::from_bytes(zero_points, scales.len(), dtype)?)
        }
        (None, None) => None,
    };
    if let Some(offset) = offset {
        blobs.push((Parameter::Offset, blocks.to_vec().into(), offset).into());
    }
    Ok(blobs)
}
