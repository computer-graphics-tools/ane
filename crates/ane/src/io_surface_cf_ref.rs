use objc2::encode::{Encode, Encoding};
use objc2_io_surface::IOSurface;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct IOSurfaceCFRef(pub *const IOSurface);

unsafe impl Encode for IOSurfaceCFRef {
    const ENCODING: Encoding = Encoding::Pointer(&Encoding::Struct("__IOSurface", &[]));
}
