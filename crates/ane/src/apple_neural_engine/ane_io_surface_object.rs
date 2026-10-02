use objc2::rc::Retained;
use objc2_io_surface::{IOSurface, IOSurfaceRef};

use crate::apple_neural_engine::AneError;

raw_class!(ANEIOSurfaceObject, c"_ANEIOSurfaceObject");

impl ANEIOSurfaceObject {
    pub fn new(surface: &IOSurface) -> Result<Retained<Self>, AneError> {
        let surface = unsafe { &*(surface as *const IOSurface).cast::<IOSurfaceRef>() };
        unsafe {
            Retained::retain_autoreleased(raw_message!(Self::class(), c"objectWithIOSurface:",
                surface => &IOSurfaceRef; *mut Self))
        }
        .ok_or(AneError::ObjectCreation(
            Self::class()
                .name()
                .to_str()
                .expect("invalid runtime class name"),
        ))
    }
}
