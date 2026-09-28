use std::ffi::CStr;

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject};
use objc2_foundation::NSArray;
use objc2_io_surface::IOSurface;

use crate::Error;
use crate::ane_io_surface_object::AneIoSurfaceObject;
use crate::ane_request::ANERequest;

pub struct Request {
    inner: Retained<ANERequest>,
}

unsafe impl Send for Request {}
unsafe impl Sync for Request {}

impl Request {
    pub fn inner(&self) -> &ANERequest {
        &self.inner
    }
    pub fn new(
        inputs: &[&IOSurface],
        outputs: &[&IOSurface],
        input_indices: &[u32],
        output_indices: &[u32],
        wait: &[(&AnyObject, u64)],
        signal: &[(&AnyObject, u64)],
    ) -> Result<Self, Error> {
        let wrap = |surfaces: &[&IOSurface]| {
            surfaces
                .iter()
                .map(|surface| AneIoSurfaceObject::with_io_surface(surface))
                .collect::<Option<Vec<_>>>()
                .ok_or(Error::SurfaceWrap)
        };
        let input_objs = wrap(inputs)?;
        let output_objs = wrap(outputs)?;

        let input_refs: Vec<&AneIoSurfaceObject> =
            input_objs.iter().map(|object| &**object).collect();
        let output_refs: Vec<&AneIoSurfaceObject> =
            output_objs.iter().map(|object| &**object).collect();

        let events = if wait.is_empty() && signal.is_empty() {
            None
        } else {
            Some(shared_events(wait, signal)?)
        };
        let inner = ANERequest::with_multiple_io(
            &input_refs,
            &output_refs,
            input_indices,
            output_indices,
            events.as_deref(),
        )
        .ok_or(Error::RequestCreation)?;

        Ok(Self { inner })
    }
}

fn shared_events(
    wait: &[(&AnyObject, u64)],
    signal: &[(&AnyObject, u64)],
) -> Result<Retained<AnyObject>, Error> {
    let class = |name: &CStr| AnyClass::get(name).ok_or(Error::RequestCreation);
    let shared_event = class(c"IOSurfaceSharedEvent")?;
    if wait.iter().chain(signal).any(|(event, _)| {
        let kind: bool = unsafe { msg_send![*event, isKindOfClass: shared_event] };
        !kind
    }) {
        return Err(Error::Binding(
            "ANE events must be IOSurfaceSharedEvent objects".into(),
        ));
    }
    let (wait_class, signal_class) = (
        class(c"_ANESharedWaitEvent")?,
        class(c"_ANESharedSignalEvent")?,
    );
    let waits = wait
        .iter()
        .map(|(event, value)| -> Option<Retained<AnyObject>> {
            unsafe { msg_send![wait_class, waitEventWithValue: *value, sharedEvent: *event] }
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::RequestCreation)?;
    let signals = signal
        .iter()
        .map(|(event, value)| -> Option<Retained<AnyObject>> {
            unsafe {
                msg_send![signal_class,
                    signalEventWithValue: *value,
                    symbolIndex: 0u32,
                    eventType: 0i64,
                    sharedEvent: *event]
            }
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(Error::RequestCreation)?;
    let events: Option<Retained<AnyObject>> = unsafe {
        msg_send![class(c"_ANESharedEvents")?,
            sharedEventsWithSignalEvents: &*NSArray::from_retained_slice(&signals),
            waitEvents: &*NSArray::from_retained_slice(&waits)]
    };
    events.ok_or(Error::RequestCreation)
}
