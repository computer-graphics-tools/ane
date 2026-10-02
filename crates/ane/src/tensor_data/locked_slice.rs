use std::ops::Deref;

use crate::tensor_data::{TensorData, TensorDataError};
use crate::{DataType, io_surface::SurfaceLock};

pub struct LockedSlice<'a> {
    lock: SurfaceLock<'a>,
    len: usize,
}

impl Deref for LockedSlice<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.lock.base_address().cast(), self.len) }
    }
}

impl<'a> TryFrom<&'a TensorData> for LockedSlice<'a> {
    type Error = TensorDataError;
    fn try_from(data: &'a TensorData) -> Result<Self, TensorDataError> {
        if data.data_type() != DataType::Float32 || !data.spec().is_contiguous() {
            return Err(TensorDataError::NotContiguousFloat32);
        }
        Ok(Self {
            lock: SurfaceLock::new(data.surface(), false)?,
            len: data.spec().element_count(),
        })
    }
}
