use objc2::rc::Retained;
use objc2::runtime::NSObject;
use objc2::{ClassType, extern_class, extern_conformance, msg_send};
use objc2_foundation::NSObjectProtocol;
use objc2_io_surface::IOSurface;

#[path = "io_surface_cf_ref.rs"]
mod io_surface_cf_ref;
use io_surface_cf_ref::IOSurfaceCFRef;

extern_class!(
    #[unsafe(super(NSObject))]
    #[name = "_ANEIOSurfaceObject"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct AneIoSurfaceObject;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for AneIoSurfaceObject {}
);

impl AneIoSurfaceObject {
    pub fn with_io_surface(surface: &IOSurface) -> Option<Retained<AneIoSurfaceObject>> {
        let cf_ref = IOSurfaceCFRef(surface as *const IOSurface);
        unsafe { msg_send![Self::class(), objectWithIOSurface: cf_ref] }
    }
}
