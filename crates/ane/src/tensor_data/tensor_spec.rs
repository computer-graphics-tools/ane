use objc2_io_surface::IOSurface;

use crate::io_surface::IOSurfaceExt;
use crate::tensor_data::{TensorData, TensorDataError};
use crate::{DataType, padded_shape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    name: String,
    shape: Box<[usize]>,
    dtype: DataType,
    strides: [usize; 4],
    bytes: usize,
}

impl TensorSpec {
    pub fn new(
        name: impl Into<String>,
        shape: &[usize],
        dtype: DataType,
    ) -> Result<Self, TensorDataError> {
        let physical = checked_shape(shape)?;
        let mut strides = [dtype.byte_width(); 4];
        for axis in (0..3).rev() {
            strides[axis] = strides[axis + 1] * physical[axis + 1];
        }
        let bytes = strides[0] * physical[0];
        Ok(Self {
            name: name.into(),
            shape: shape.into(),
            dtype,
            strides,
            bytes,
        })
    }

    pub fn with_layout(
        mut self,
        strides: [usize; 4],
        bytes: usize,
    ) -> Result<Self, TensorDataError> {
        let shape = checked_shape(&self.shape)?;
        if strides[3] != self.dtype.byte_width() {
            return Err(TensorDataError::InvalidStrides(strides));
        }
        let mut extent = self.dtype.byte_width();
        for axis in (0..4).rev() {
            if shape[axis] > 1 && strides[axis] < extent {
                return Err(TensorDataError::InvalidStrides(strides));
            }
            extent = (shape[axis] - 1)
                .checked_mul(strides[axis])
                .and_then(|n| n.checked_add(extent))
                .ok_or(TensorDataError::InvalidStrides(strides))?;
        }
        if bytes < extent || bytes > isize::MAX as usize {
            return Err(TensorDataError::AllocationTooSmall {
                required: extent,
                available: bytes,
            });
        }
        self.strides = strides;
        self.bytes = bytes;
        Ok(self)
    }

    pub fn with_shape(mut self, shape: &[usize]) -> Result<Self, TensorDataError> {
        if checked_shape(shape)? != checked_shape(&self.shape)? {
            return Err(TensorDataError::ShapeMismatch {
                expected: self.shape,
                actual: shape.into(),
            });
        }
        self.shape = shape.into();
        Ok(self)
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data_type(&self) -> DataType {
        self.dtype
    }
    pub fn strides(&self) -> &[usize; 4] {
        &self.strides
    }
    pub fn allocation_size(&self) -> usize {
        self.bytes
    }
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_contiguous(&self) -> bool {
        let Ok(shape) = checked_shape(&self.shape) else {
            return false;
        };
        let mut stride = self.dtype.byte_width();
        for axis in (0..4).rev() {
            if shape[axis] > 1 && self.strides[axis] != stride {
                return false;
            }
            stride *= shape[axis];
        }
        true
    }
    pub fn byte_offset(&self, index: usize) -> Result<usize, TensorDataError> {
        let count = self.element_count();
        if index >= count {
            return Err(TensorDataError::IndexOutOfRange { index, count });
        }
        let shape = checked_shape(&self.shape)?;
        let mut index = index;
        let mut offset = 0;
        for axis in (0..4).rev() {
            offset += (index % shape[axis]) * self.strides[axis];
            index /= shape[axis];
        }
        Ok(offset)
    }
    pub fn validate(&self, data: &TensorData) -> Result<(), TensorDataError> {
        let shape = checked_shape(self.shape())?;
        if checked_shape(data.shape())? != shape
            || data.data_type() != self.data_type()
            || (0..4)
                .any(|axis| shape[axis] > 1 && data.spec().strides()[axis] != self.strides()[axis])
            || (data.surface().allocationSize() as usize) < self.allocation_size()
        {
            return Err(TensorDataError::LayoutMismatch {
                expected: Box::new(self.clone()),
                actual: Box::new(data.spec().clone()),
                allocation: data.surface().allocationSize() as usize,
            });
        }
        Ok(())
    }
    pub fn allocate(&self) -> Result<TensorData, TensorDataError> {
        unsafe {
            TensorData::from_layout(
                IOSurface::with_byte_count(self.bytes.max(16384))?,
                self.clone(),
            )
        }
    }
}

fn checked_shape(shape: &[usize]) -> Result<[usize; 4], TensorDataError> {
    padded_shape(shape).ok_or_else(|| TensorDataError::InvalidShape(shape.into()))
}
