use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, OnceLock, Weak,
    atomic::{AtomicUsize, Ordering},
};

use objc2::{Message, rc::Retained};
use objc2_io_surface::IOSurface;

use crate::io_surface::IOSurfaceError;

pub struct SurfaceLease {
    _surface: Retained<IOSurface>,
    access: Arc<AtomicUsize>,
    write: bool,
}

impl SurfaceLease {
    pub fn acquire(surface: &IOSurface, write: bool) -> Result<Self, IOSurfaceError> {
        static ACCESS: OnceLock<Mutex<HashMap<u32, Weak<AtomicUsize>>>> = OnceLock::new();
        let mut registry = ACCESS
            .get_or_init(Mutex::default)
            .lock()
            .map_err(|_| IOSurfaceError::Synchronization)?;
        if registry.len() > 4096 {
            registry.retain(|_, value| value.strong_count() != 0);
        }
        let entry = registry.entry(surface.surfaceID()).or_default();
        let access = entry.upgrade().unwrap_or_else(|| {
            let access = Arc::new(AtomicUsize::new(0));
            *entry = Arc::downgrade(&access);
            access
        });
        drop(registry);
        if write {
            access.compare_exchange(0, usize::MAX, Ordering::AcqRel, Ordering::Acquire)
        } else {
            access.try_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                (value < usize::MAX - 1).then(|| value + 1)
            })
        }
        .map_err(|_| IOSurfaceError::Busy)?;
        Ok(Self {
            _surface: surface.retain(),
            access,
            write,
        })
    }
}

impl Drop for SurfaceLease {
    fn drop(&mut self) {
        if self.write {
            self.access.store(0, Ordering::Release);
        } else {
            self.access.fetch_sub(1, Ordering::Release);
        }
    }
}
