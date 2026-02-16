use std::ops::{Deref, DerefMut};
use std::ptr;

use objc2::rc::Retained;
use objc2_io_surface::{IOSurface, IOSurfaceLockOptions};

use crate::io_surface::IOSurfaceExt;
use crate::Shape;

/// IOSurface-backed tensor storage for ANE I/O.
///
/// The underlying IOSurface is sized for fp32 I/O (4 bytes per element)
/// because MIL function signatures declare inputs/outputs as `tensor<fp32, ...>`;
/// the ANE casts to fp16 internally.
///
/// For zero-copy access, use [`as_f32_slice`](Self::as_f32_slice) and
/// [`as_f32_slice_mut`](Self::as_f32_slice_mut) which return RAII guards
/// that lock/unlock the surface automatically.
pub struct TensorData {
    surface: Retained<IOSurface>,
    shape: Shape,
}

unsafe impl Send for TensorData {}
unsafe impl Sync for TensorData {}

impl TensorData {
    /// Allocate an empty IOSurface sized for the given shape (fp32 = 4 bytes/element).
    pub fn new(shape: Shape) -> Self {
        let byte_count = shape.total_elements() * 4;
        let surface = IOSurface::with_byte_count(byte_count);
        Self { surface, shape }
    }

    /// Allocate an IOSurface and write fp32 data into it.
    pub fn with_f32(data: &[f32], shape: Shape) -> Self {
        let tensor_data = Self::new(shape);
        tensor_data.copy_from_f32(data);
        tensor_data
    }

    /// Wrap an existing IOSurface.
    pub fn from_surface(surface: Retained<IOSurface>, shape: Shape) -> Self {
        Self { surface, shape }
    }

    /// Write fp32 data into the surface, reusing the existing allocation.
    pub fn copy_from_f32(&self, data: &[f32]) {
        let mut surface = self.as_f32_slice_mut();
        surface[..data.len()].copy_from_slice(data);
    }

    /// Lock the surface read-only and return an RAII guard exposing `&[f32]`.
    ///
    /// The surface is unlocked when the guard is dropped.
    pub fn as_f32_slice(&self) -> LockedSlice<'_> {
        let element_count = self.shape.total_elements();
        self.surface.lockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
        let pointer = self.surface.baseAddress().as_ptr().cast::<f32>();
        LockedSlice {
            surface: &self.surface,
            pointer,
            element_count,
        }
    }

    /// Lock the surface read-write and return an RAII guard exposing `&mut [f32]`.
    ///
    /// The surface is unlocked when the guard is dropped.
    pub fn as_f32_slice_mut(&self) -> LockedSliceMut<'_> {
        let element_count = self.shape.total_elements();
        self.surface.lockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
        let pointer = self.surface.baseAddress().as_ptr().cast::<f32>();
        LockedSliceMut {
            surface: &self.surface,
            pointer,
            element_count,
        }
    }

    /// Read the surface contents back as fp32 values (allocating).
    ///
    /// Prefer [`as_f32_slice`](Self::as_f32_slice) for zero-copy access.
    pub fn read_f32(&self) -> Box<[f32]> {
        let slice = self.as_f32_slice();
        slice.to_vec().into_boxed_slice()
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn surface(&self) -> &IOSurface {
        &self.surface
    }
}

/// RAII guard that holds a read-only lock on an IOSurface and derefs to `&[f32]`.
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
        self.surface.unlockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
    }
}

/// RAII guard that holds a read-write lock on an IOSurface and derefs to `&mut [f32]`.
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
        self.surface.unlockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
    }
}
