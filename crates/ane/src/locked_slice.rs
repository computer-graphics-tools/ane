use std::ops::Deref;
use std::ptr;

use super::TensorData;
use objc2_io_surface::{IOSurface, IOSurfaceLockOptions};

pub struct LockedSlice<'a> {
    surface: &'a IOSurface,
    pointer: *const f32,
    element_count: usize,
}

impl Deref for LockedSlice<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.pointer, self.element_count) }
    }
}

impl Drop for LockedSlice<'_> {
    fn drop(&mut self) {
        self.surface
            .unlockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
    }
}

impl<'a> From<&'a TensorData> for LockedSlice<'a> {
    fn from(data: &'a TensorData) -> Self {
        data.surface
            .lockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
        Self {
            surface: &data.surface,
            pointer: data.surface.baseAddress().as_ptr().cast::<f32>(),
            element_count: data.shape.iter().product(),
        }
    }
}
