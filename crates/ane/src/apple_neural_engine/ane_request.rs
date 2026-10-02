use block2::{Block, RcBlock};
use objc2::{
    rc::Retained,
    runtime::{AnyObject, Bool},
};
use objc2_foundation::{NSArray, NSError, NSNumber};
use objc2_io_surface::IOSurface;

use crate::io_surface::SurfaceEvent;

use crate::apple_neural_engine::{
    ANEIOSurfaceObject, ANESharedEvents, ANESharedSignalEvent, ANESharedWaitEvent, AneError,
    ensure_event_interfaces, ensure_model_interfaces,
};

raw_class!(ANERequest, c"_ANERequest");

impl ANERequest {
    pub fn new(
        inputs: &[&IOSurface],
        input_indices: &[u32],
        outputs: &[&IOSurface],
        output_indices: &[u32],
    ) -> Result<Retained<Self>, AneError> {
        ensure_model_interfaces()?;
        let surfaces = |surfaces: &[&IOSurface]| {
            surfaces
                .iter()
                .map(|surface| ANEIOSurfaceObject::new(surface))
                .collect::<Result<Vec<_>, _>>()
                .map(|objects| NSArray::from_retained_slice(&objects))
        };
        let indices = |indices: &[u32]| {
            NSArray::from_retained_slice(
                &indices
                    .iter()
                    .copied()
                    .map(NSNumber::new_u32)
                    .collect::<Vec<_>>(),
            )
        };
        let inputs = surfaces(inputs)?;
        let outputs = surfaces(outputs)?;
        let input_indices = indices(input_indices);
        let output_indices = indices(output_indices);
        let procedure = NSNumber::new_u32(0);
        unsafe {
            Retained::retain_autoreleased(raw_message!(Self::class(),
                c"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:",
                &*inputs => &NSArray<ANEIOSurfaceObject>, &*input_indices => &NSArray<NSNumber>,
                &*outputs => &NSArray<ANEIOSurfaceObject>, &*output_indices => &NSArray<NSNumber>,
                None => Option<&AnyObject>, None => Option<&AnyObject>,
                &*procedure => &NSNumber, None => Option<&ANESharedEvents>; *mut Self))
        }
        .ok_or(AneError::ObjectCreation(Self::class().name().to_str().expect("invalid runtime class name")))
    }

    pub fn set_completion_handler<F>(&self, handler: F) -> Result<(), AneError>
    where
        F: Fn(Result<(), AneError>) + Send + Sync + 'static,
    {
        ensure_event_interfaces()?;
        let handler = RcBlock::new(move |success: Bool, error: *mut NSError| {
            handler(if success.as_bool() {
                Ok(())
            } else {
                Err(unsafe { Retained::retain(error) }
                    .map(AneError::from)
                    .unwrap_or(AneError::EvaluationFailed))
            });
        });
        unsafe {
            raw_message!(self, c"setCompletionHandler:",
                &*handler => &Block<dyn Fn(Bool, *mut NSError)>; ())
        }
        Ok(())
    }

    pub fn set_shared_events(
        &self,
        wait: &[(&SurfaceEvent, u64)],
        signal: &[(&SurfaceEvent, u64)],
    ) -> Result<(), AneError> {
        ensure_event_interfaces()?;
        let wait = wait
            .iter()
            .map(|&(event, value)| ANESharedWaitEvent::new(value, event))
            .collect::<Result<Vec<_>, _>>()?;
        let signal = signal
            .iter()
            .map(|&(event, value)| ANESharedSignalEvent::new(value, event))
            .collect::<Result<Vec<_>, _>>()?;
        let events = ANESharedEvents::new(&signal, &wait)?;
        unsafe { raw_message!(self, c"setSharedEvents:", &*events => &ANESharedEvents; ()) }
        Ok(())
    }
}
