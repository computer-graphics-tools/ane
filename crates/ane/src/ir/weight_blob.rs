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
}
