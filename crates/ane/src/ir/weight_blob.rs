use std::sync::Arc;

use crate::WeightDataType;
use crate::ir::IrError;

#[derive(Clone, PartialEq)]
pub struct WeightBlob {
    data: Arc<[u8]>,
    dtype: WeightDataType,
    elements: usize,
}

impl WeightBlob {
    fn code(&self, index: usize) -> u16 {
        let bits = self.dtype.bit_width();
        let bit = index * bits;
        let mut code = u16::from(self.data[bit / 8]) >> (bit % 8);
        if bit % 8 + bits > 8 {
            code |= u16::from(self.data[bit / 8 + 1]) << (8 - bit % 8);
        }
        code & ((1 << bits) - 1)
    }
    fn put_code(bytes: &mut [u8], index: usize, bits: usize, code: u16) {
        let bit = index * bits;
        let code = code << (bit % 8);
        bytes[bit / 8] |= code as u8;
        if bit % 8 + bits > 8 {
            bytes[bit / 8 + 1] |= (code >> 8) as u8;
        }
    }
    pub fn native_palette(&self, lut: &Self) -> Result<(Self, Self), IrError> {
        let dtype = match self.dtype {
            WeightDataType::UInt1 => WeightDataType::UInt2,
            WeightDataType::UInt3 => WeightDataType::UInt4,
            WeightDataType::UInt6 => WeightDataType::UInt8,
            _ => return Ok((self.clone(), lut.clone())),
        };
        let entries = 1usize << self.dtype.bit_width();
        if lut.dtype != WeightDataType::Float16 || !lut.elements.is_multiple_of(entries) {
            return Err(IrError::InvalidProgram("invalid palette data"));
        }
        let bits = dtype.bit_width();
        let mut codes = vec![0; (self.elements * bits).div_ceil(8)];
        for index in 0..self.elements {
            Self::put_code(&mut codes, index, bits, self.code(index));
        }
        let groups = lut.elements / entries;
        let width = (1usize << bits) * 2;
        let mut palette = vec![0; groups * width];
        for group in 0..groups {
            palette[group * width..group * width + entries * 2]
                .copy_from_slice(&lut.data[group * entries * 2..(group + 1) * entries * 2]);
        }
        Ok((
            Self::from_bytes(codes, self.elements, dtype)?,
            Self::from_bytes(palette, groups * (1 << bits), WeightDataType::Float16)?,
        ))
    }
    pub fn transpose_blocks(
        &self,
        rows: usize,
        columns: usize,
        block: usize,
    ) -> Result<Self, IrError> {
        let bits = self.dtype.bit_width();
        if block == 0
            || rows.checked_mul(columns).and_then(|n| n.checked_mul(block)) != Some(self.elements)
        {
            return Err(IrError::InvalidProgram("invalid packed matrix transpose"));
        }
        let mut bytes = vec![0; self.data.len()];
        for row in 0..rows {
            for column in 0..columns {
                let source = (row * columns + column) * block;
                let target = (column * rows + row) * block;
                if bits >= 8 {
                    let size = bits / 8;
                    bytes[target * size..(target + block) * size]
                        .copy_from_slice(&self.data[source * size..(source + block) * size]);
                } else {
                    for index in 0..block {
                        Self::put_code(&mut bytes, target + index, bits, self.code(source + index));
                    }
                }
            }
        }
        Self::from_bytes(bytes, self.elements, self.dtype)
    }
    pub fn slice_matrix(
        &self,
        shape: [usize; 2],
        origin: [usize; 2],
        size: [usize; 2],
    ) -> Result<Self, IrError> {
        if shape[0].checked_mul(shape[1]) != Some(self.elements)
            || !(0..2).all(|axis| {
                size[axis] > 0
                    && origin[axis]
                        .checked_add(size[axis])
                        .is_some_and(|v| v <= shape[axis])
            })
        {
            return Err(IrError::InvalidProgram("invalid packed matrix slice"));
        }
        let bits = self.dtype.bit_width();
        let count = size[0] * size[1];
        let mut bytes = vec![0; (count * bits).div_ceil(8)];
        for row in 0..size[0] {
            let source = (origin[0] + row) * shape[1] + origin[1];
            let target = row * size[1];
            if bits >= 8 {
                let width = bits / 8;
                bytes[target * width..(target + size[1]) * width]
                    .copy_from_slice(&self.data[source * width..(source + size[1]) * width]);
            } else {
                for column in 0..size[1] {
                    Self::put_code(
                        &mut bytes,
                        target + column,
                        bits,
                        self.code(source + column),
                    );
                }
            }
        }
        Self::from_bytes(bytes, count, self.dtype)
    }
    pub fn from_bytes(
        data: impl Into<Arc<[u8]>>,
        elements: usize,
        dtype: WeightDataType,
    ) -> Result<Self, IrError> {
        let data = data.into();
        let expected = elements
            .checked_mul(dtype.bit_width())
            .ok_or(IrError::WeightSizeOverflow {
                elements,
                data_type: dtype,
            })?
            .div_ceil(8);
        if data.len() != expected {
            return Err(IrError::WeightLength {
                elements,
                data_type: dtype,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            data,
            dtype,
            elements,
        })
    }
    pub fn from_f32(values: &[f32]) -> Result<Self, IrError> {
        if values
            .iter()
            .any(|&v| v.is_finite() && !half::f16::from_f32(v).is_finite())
        {
            return Err(IrError::InvalidValue("fp16"));
        }
        let values: Box<[_]> = values.iter().copied().map(half::f16::from_f32).collect();
        Ok(Self {
            data: bytemuck::cast_slice(&values).into(),
            dtype: WeightDataType::Float16,
            elements: values.len(),
        })
    }
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }
    pub fn data_type(&self) -> WeightDataType {
        self.dtype
    }
    pub fn element_count(&self) -> usize {
        self.elements
    }
    pub fn decoded(&self) -> Vec<f32> {
        use WeightDataType as D;
        let read = |bytes| match self.dtype {
            D::Float16 => half::f16::from_bits(bytemuck::pod_read_unaligned(bytes)).to_f32(),
            D::Float32 => bytemuck::pod_read_unaligned(bytes),
            D::Int16 => bytemuck::pod_read_unaligned::<i16>(bytes) as f32,
            D::UInt16 => bytemuck::pod_read_unaligned::<u16>(bytes) as f32,
            D::Int32 => bytemuck::pod_read_unaligned::<i32>(bytes) as f32,
            D::UInt32 => bytemuck::pod_read_unaligned::<u32>(bytes) as f32,
            _ => unreachable!(),
        };
        let bits = self.dtype.bit_width();
        if bits >= 16 {
            return self.data.chunks_exact(bits / 8).map(read).collect();
        }
        (0..self.elements)
            .map(|i| {
                let code = self.code(i);
                if matches!(self.dtype, D::Int4 | D::Int8) && code & (1 << (bits - 1)) != 0 {
                    (i32::from(code) - (1 << bits)) as f32
                } else {
                    code as f32
                }
            })
            .collect()
    }
    pub fn padding_bits(&self) -> usize {
        self.data.len() * 8 - self.elements * self.dtype.bit_width()
    }
}
