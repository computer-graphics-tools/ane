use std::ptr;

use objc2_io_surface::{IOSurface, IOSurfaceLockOptions};

use crate::io_surface::{IOSurfaceError, SurfaceLease};

pub struct SurfaceLock<'a> {
    surface: &'a IOSurface,
    options: IOSurfaceLockOptions,
    locked: bool,
    _lease: SurfaceLease,
}

impl<'a> SurfaceLock<'a> {
    pub fn new(surface: &'a IOSurface, write: bool) -> Result<Self, IOSurfaceError> {
        let lease = SurfaceLease::acquire(surface, write)?;
        let options = if write {
            IOSurfaceLockOptions::empty()
        } else {
            IOSurfaceLockOptions::ReadOnly
        };
        match surface.lockWithOptions_seed(options, ptr::null_mut()) {
            0 => Ok(Self {
                surface,
                options,
                locked: true,
                _lease: lease,
            }),
            code => Err(IOSurfaceError::Lock(code)),
        }
    }

    pub fn base_address(&self) -> *mut u8 {
        self.surface.baseAddress().as_ptr().cast()
    }

    pub fn unlock(mut self) -> Result<(), IOSurfaceError> {
        self.release()
    }

    fn release(&mut self) -> Result<(), IOSurfaceError> {
        if !std::mem::take(&mut self.locked) {
            return Ok(());
        }
        match self
            .surface
            .unlockWithOptions_seed(self.options, ptr::null_mut())
        {
            0 => Ok(()),
            code => Err(IOSurfaceError::Unlock(code)),
        }
    }
}

impl Drop for SurfaceLock<'_> {
    fn drop(&mut self) {
        let _ = self.release();
    }
}
