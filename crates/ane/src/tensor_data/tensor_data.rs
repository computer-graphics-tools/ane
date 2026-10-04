use std::ptr;

use half::f16;
use objc2::rc::Retained;
use objc2_io_surface::IOSurface;

use crate::io_surface::{IOSurfaceExt, SurfaceLock};
use crate::tensor_data::{LockedSlice, LockedSliceMut, TensorDataError, TensorElement, TensorSpec};
use crate::{DataType, logical_shape};

#[derive(Clone)]
pub struct TensorData {
    surface: Retained<IOSurface>,
    spec: TensorSpec,
}

impl TensorData {
    pub fn from_slice<T: TensorElement, const RANK: usize>(
        data: &[T],
        shape: [usize; RANK],
    ) -> Result<Self, TensorDataError> {
        let tensor = Self::with_type(shape, T::DATA_TYPE)?;
        tensor.write(data)?;
        Ok(tensor)
    }
    pub fn write<T: TensorElement>(&self, data: &[T]) -> Result<(), TensorDataError> {
        self.expect_data_type(T::DATA_TYPE)?;
        self.expect_length(data.len())?;
        self.write_bytes(bytemuck::cast_slice(data))
    }
    pub fn read<T: TensorElement>(&self) -> Result<Box<[T]>, TensorDataError> {
        self.expect_data_type(T::DATA_TYPE)?;
        self.read_bytes()?
            .chunks(T::DATA_TYPE.byte_width())
            .map(bytemuck::checked::try_pod_read_unaligned)
            .collect::<Result<_, _>>()
            .map_err(|_| TensorDataError::InvalidBitPattern)
    }

    pub fn with_type<const RANK: usize>(
        shape: [usize; RANK],
        dtype: DataType,
    ) -> Result<Self, TensorDataError> {
        TensorSpec::new("", logical_shape(&shape), dtype)?.allocate()
    }
    /// # Safety
    ///
    /// External CPU, GPU and device users of the surface must be synchronized with this crate,
    /// the surface allocation must stay at least as large as the layout, and every element
    /// that is read must be initialized.
    pub unsafe fn from_surface<const RANK: usize>(
        surface: Retained<IOSurface>,
        shape: [usize; RANK],
        dtype: DataType,
    ) -> Result<Self, TensorDataError> {
        unsafe { Self::from_layout(surface, TensorSpec::new("", logical_shape(&shape), dtype)?) }
    }
    /// # Safety
    ///
    /// External CPU, GPU and device users of the surface must be synchronized with this crate,
    /// the surface allocation must stay at least as large as the layout, and every element
    /// that is read must be initialized.
    pub unsafe fn from_layout(
        surface: Retained<IOSurface>,
        spec: TensorSpec,
    ) -> Result<Self, TensorDataError> {
        let available = surface.allocationSize() as usize;
        if spec.allocation_size() > available {
            return Err(TensorDataError::AllocationTooSmall {
                required: spec.allocation_size(),
                available,
            });
        }
        if !(surface.baseAddress().as_ptr() as usize).is_multiple_of(spec.data_type().byte_width())
        {
            return Err(TensorDataError::Misaligned(spec.data_type()));
        }
        Ok(Self { surface, spec })
    }
    pub fn copy_from_f32(&self, data: &[f32]) -> Result<(), TensorDataError> {
        self.expect_length(data.len())?;
        match self.data_type() {
            DataType::Float32 if self.spec.is_contiguous() => {
                self.as_f32_slice_mut()?.copy_from_slice(data);
                Ok(())
            }
            DataType::Float32 => self.write(data),
            DataType::Float16 => {
                let data: Box<[_]> = data.iter().copied().map(f16::from_f32).collect();
                self.write(&data)
            }
            dtype => Err(TensorDataError::NotFloatingPoint(dtype)),
        }
    }
    pub fn write_bytes(&self, data: &[u8]) -> Result<(), TensorDataError> {
        let size = self.data_type().byte_width();
        self.expect_byte_length(data.len())?;
        if self.spec.is_contiguous() {
            return Ok(unsafe { self.surface.write_bytes(data) }?);
        }
        let lock = SurfaceLock::new(&self.surface, true)?;
        let base = lock.base_address();
        for (index, value) in data.chunks_exact(size).enumerate() {
            let offset = self.spec.byte_offset(index)?;
            unsafe { ptr::copy_nonoverlapping(value.as_ptr(), base.add(offset), size) };
        }
        Ok(lock.unlock()?)
    }
    pub fn read_bytes(&self) -> Result<Box<[u8]>, TensorDataError> {
        let size = self.data_type().byte_width();
        let mut bytes = vec![0; self.spec.element_count() * size];
        if self.spec.is_contiguous() {
            unsafe { self.surface.read_bytes(&mut bytes) }?;
            return Ok(bytes.into());
        }
        let lock = SurfaceLock::new(&self.surface, false)?;
        let base = lock.base_address();
        for (index, value) in bytes.chunks_exact_mut(size).enumerate() {
            let offset = self.spec.byte_offset(index)?;
            unsafe { ptr::copy_nonoverlapping(base.add(offset), value.as_mut_ptr(), size) };
        }
        lock.unlock()?;
        Ok(bytes.into())
    }
    pub fn as_f32_slice(&self) -> Result<LockedSlice<'_>, TensorDataError> {
        self.try_into()
    }
    pub fn as_f32_slice_mut(&self) -> Result<LockedSliceMut<'_>, TensorDataError> {
        self.try_into()
    }
    pub fn read_f32(&self) -> Result<Box<[f32]>, TensorDataError> {
        match self.data_type() {
            DataType::Float32 if self.spec.is_contiguous() => {
                Ok(self.as_f32_slice()?.to_vec().into())
            }
            DataType::Float32 => self.read(),
            DataType::Float16 => Ok(self
                .read::<f16>()?
                .iter()
                .map(|value| value.to_f32())
                .collect()),
            dtype => Err(TensorDataError::NotFloatingPoint(dtype)),
        }
    }
    pub fn shape(&self) -> &[usize] {
        self.spec.shape()
    }
    pub fn data_type(&self) -> DataType {
        self.spec.data_type()
    }
    pub fn spec(&self) -> &TensorSpec {
        &self.spec
    }
    pub fn surface(&self) -> &IOSurface {
        &self.surface
    }

    fn expect_data_type(&self, actual: DataType) -> Result<(), TensorDataError> {
        if actual == self.data_type() {
            Ok(())
        } else {
            Err(TensorDataError::DataTypeMismatch {
                expected: self.data_type(),
                actual,
            })
        }
    }
    fn expect_length(&self, actual: usize) -> Result<(), TensorDataError> {
        if actual == self.spec.element_count() {
            Ok(())
        } else {
            Err(TensorDataError::LengthMismatch {
                expected: self.spec.element_count(),
                actual,
            })
        }
    }
    fn expect_byte_length(&self, actual: usize) -> Result<(), TensorDataError> {
        let expected = self.spec.element_count() * self.data_type().byte_width();
        if actual == expected {
            Ok(())
        } else {
            Err(TensorDataError::ByteLengthMismatch { expected, actual })
        }
    }
}
