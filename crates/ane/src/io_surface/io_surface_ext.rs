use std::ptr;

use objc2::{AnyThread, rc::Retained, runtime::AnyObject};
use objc2_foundation::{NSDictionary, NSNumber, NSString};
use objc2_io_surface::{
    IOSurface, IOSurfacePropertyKeyBytesPerElement, IOSurfacePropertyKeyBytesPerRow,
    IOSurfacePropertyKeyHeight, IOSurfacePropertyKeyWidth,
};

use crate::io_surface::{IOSurfaceError, SurfaceLock};

pub trait IOSurfaceExt {
    fn with_byte_count(byte_count: usize) -> Result<Retained<IOSurface>, IOSurfaceError>;

    /// # Safety
    ///
    /// No other CPU, GPU or ANE user may access the surface during the copy.
    unsafe fn write_bytes(&self, data: &[u8]) -> Result<(), IOSurfaceError>;

    /// # Safety
    ///
    /// No other CPU, GPU or ANE user may write the surface during the copy.
    unsafe fn read_bytes(&self, buffer: &mut [u8]) -> Result<(), IOSurfaceError>;
}

impl IOSurfaceExt for IOSurface {
    fn with_byte_count(byte_count: usize) -> Result<Retained<IOSurface>, IOSurfaceError> {
        if byte_count == 0 || byte_count > isize::MAX as usize {
            return Err(IOSurfaceError::InvalidSize(byte_count));
        }
        let (count, one) = (NSNumber::new_usize(byte_count), NSNumber::new_usize(1));
        let keys = unsafe {
            [
                IOSurfacePropertyKeyWidth,
                IOSurfacePropertyKeyHeight,
                IOSurfacePropertyKeyBytesPerElement,
                IOSurfacePropertyKeyBytesPerRow,
            ]
        };
        let properties: Retained<NSDictionary<NSString, AnyObject>> = NSDictionary::from_slices(
            &keys,
            &[count.as_ref(), one.as_ref(), one.as_ref(), count.as_ref()],
        );
        let surface = IOSurface::initWithProperties(IOSurface::alloc(), &properties)
            .ok_or(IOSurfaceError::Allocation)?;
        let lock = SurfaceLock::new(&surface, true)?;
        unsafe { ptr::write_bytes(lock.base_address(), 0, byte_count) };
        lock.unlock()?;
        Ok(surface)
    }

    unsafe fn write_bytes(&self, data: &[u8]) -> Result<(), IOSurfaceError> {
        let lock = lock_range(self, data.len(), true)?;
        unsafe { ptr::copy_nonoverlapping(data.as_ptr(), lock.base_address(), data.len()) };
        lock.unlock()
    }

    unsafe fn read_bytes(&self, buffer: &mut [u8]) -> Result<(), IOSurfaceError> {
        let lock = lock_range(self, buffer.len(), false)?;
        unsafe { ptr::copy_nonoverlapping(lock.base_address(), buffer.as_mut_ptr(), buffer.len()) };
        lock.unlock()
    }
}

fn lock_range(
    surface: &IOSurface,
    length: usize,
    write: bool,
) -> Result<SurfaceLock<'_>, IOSurfaceError> {
    let allocation = surface.allocationSize() as usize;
    if length > allocation {
        return Err(IOSurfaceError::OutOfBounds { length, allocation });
    }
    SurfaceLock::new(surface, write)
}
