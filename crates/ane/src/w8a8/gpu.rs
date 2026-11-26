//! Metal buffers for program surfaces and uploads, shared events, and host residency callbacks.

use std::{ffi::c_void, ptr::NonNull, sync::Arc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLSharedEvent};

use super::kernels::Region;
use crate::{Error, SharedEvent, TensorData};

/// Callbacks that add each created `MTLBuffer` to host residency and remove it before release.
/// Either callback may run on any thread.
pub struct Residency {
    pub add: Box<dyn Fn(*mut c_void) + Send + Sync>,
    pub remove: Box<dyn Fn(*mut c_void) + Send + Sync>,
}

/// A resident shared-storage Metal buffer.
pub struct Buffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    residency: Arc<Residency>,
}

// SAFETY: Metal buffers are thread safe; the program's buffers are written only by the host's queue
// and the ANE, ordered through events.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Aliases `data`'s storage without retaining its IOSurface.
    ///
    /// # Safety
    ///
    /// `data` must outlive this buffer and all GPU work using it.
    pub unsafe fn alias(
        device: &ProtocolObject<dyn MTLDevice>,
        residency: &Arc<Residency>,
        data: &TensorData,
    ) -> Result<Self, Error> {
        let shared = MTLResourceOptions::StorageModeShared;
        // SAFETY: the surface's memory is valid for its allocation size while `data` lives.
        let buffer = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                data.base_address(),
                data.allocation_size(),
                shared,
                None,
            )
        };
        Self::resident(buffer, residency, "cannot alias an IOSurface")
    }

    /// Copies `bytes` into shared storage at a 16-byte-aligned address.
    pub fn upload(
        device: &ProtocolObject<dyn MTLDevice>,
        residency: &Arc<Residency>,
        bytes: &[u8],
    ) -> Result<Self, Error> {
        let shared = MTLResourceOptions::StorageModeShared;
        let pointer = NonNull::from(bytes).cast();
        // SAFETY: `bytes` is valid for its length.
        let buffer =
            unsafe { device.newBufferWithBytes_length_options(pointer, bytes.len(), shared) };
        Self::resident(buffer, residency, "cannot allocate a buffer")
    }

    fn resident(
        buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        residency: &Arc<Residency>,
        failure: &'static str,
    ) -> Result<Self, Error> {
        let buffer = buffer.ok_or_else(|| Error::Metal(failure.to_string()))?;
        (residency.add)(Retained::as_ptr(&buffer).cast_mut().cast());
        Ok(Self {
            buffer,
            residency: residency.clone(),
        })
    }

    pub fn region(&self) -> Region {
        Region {
            address: self.buffer.gpuAddress(),
            len: self.buffer.length(),
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        (self.residency.remove)(Retained::as_ptr(&self.buffer).cast_mut().cast());
    }
}

/// A shared event adopted by the ANE and signaled or waited on by the host queue.
pub struct Event(Retained<ProtocolObject<dyn MTLSharedEvent>>);

// SAFETY: Metal shared events are thread safe.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    /// A new event and its adoption by the ANE.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<(Self, SharedEvent), Error> {
        let event = device
            .newSharedEvent()
            .ok_or_else(|| Error::Metal("cannot create a shared event".to_string()))?;
        // SAFETY: `event` is an `MTLSharedEvent`; `from_object` fails if a validation layer wraps it.
        let shared = unsafe { SharedEvent::from_object(event.clone().into()) }?;
        Ok((Self(event), shared))
    }

    /// The `id<MTLSharedEvent>`, valid while `self` lives.
    pub fn as_ptr(&self) -> *mut c_void {
        Retained::as_ptr(&self.0).cast_mut().cast()
    }
}
