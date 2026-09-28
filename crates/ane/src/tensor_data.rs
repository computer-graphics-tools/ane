use objc2::rc::Retained;
use objc2_io_surface::IOSurface;

use crate::io_surface::IOSurfaceExt;

#[path = "locked_slice.rs"]
mod locked_slice;
#[path = "locked_slice_mut.rs"]
mod locked_slice_mut;
pub use locked_slice::LockedSlice;
pub use locked_slice_mut::LockedSliceMut;

pub struct TensorData {
    surface: Retained<IOSurface>,
    shape: [usize; 4],
}

unsafe impl Send for TensorData {}
unsafe impl Sync for TensorData {}

impl TensorData {
    pub fn new(shape: &[usize]) -> Self {
        let shape = crate::dimensions(shape);
        let byte_count = shape.iter().product::<usize>() * 4;
        let surface = IOSurface::with_byte_count(byte_count);
        Self { surface, shape }
    }

    pub fn with_f32(data: &[f32], shape: &[usize]) -> Self {
        let tensor_data = Self::new(shape);
        tensor_data.copy_from_f32(data);
        tensor_data
    }

    pub fn from_surface(surface: Retained<IOSurface>, shape: &[usize]) -> Self {
        let shape = crate::dimensions(shape);
        Self { surface, shape }
    }

    pub fn copy_from_f32(&self, data: &[f32]) {
        let mut surface = self.as_f32_slice_mut();
        surface[..data.len()].copy_from_slice(data);
    }

    pub fn as_f32_slice(&self) -> LockedSlice<'_> {
        self.into()
    }

    pub fn as_f32_slice_mut(&self) -> LockedSliceMut<'_> {
        self.into()
    }

    pub fn read_f32(&self) -> Box<[f32]> {
        let slice = self.as_f32_slice();
        slice.to_vec().into_boxed_slice()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn surface(&self) -> &IOSurface {
        &self.surface
    }
}
