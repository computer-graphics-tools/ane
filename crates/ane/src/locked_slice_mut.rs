use std::ops::{Deref, DerefMut};
use std::ptr;

use super::TensorData;
use objc2_io_surface::{IOSurface, IOSurfaceLockOptions};

pub struct LockedSliceMut<'a> {
    surface: &'a IOSurface,
    pointer: *mut f32,
    element_count: usize,
}

impl Deref for LockedSliceMut<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.pointer, self.element_count) }
    }
}

impl DerefMut for LockedSliceMut<'_> {
    fn deref_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.element_count) }
    }
}

impl Drop for LockedSliceMut<'_> {
    fn drop(&mut self) {
        self.surface
            .unlockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
    }
}

impl<'a> From<&'a TensorData> for LockedSliceMut<'a> {
    fn from(data: &'a TensorData) -> Self {
        data.surface
            .lockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
        Self {
            surface: &data.surface,
            pointer: data.surface.baseAddress().as_ptr().cast::<f32>(),
            element_count: data.shape.iter().product(),
        }
    }
}
