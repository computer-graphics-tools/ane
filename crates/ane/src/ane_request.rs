use block2::Block;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, NSObject};
use objc2::{ClassType, Message, extern_class, extern_conformance, msg_send};
use objc2_foundation::{NSArray, NSError, NSNumber, NSObjectProtocol};

use crate::ane_io_surface_object::AneIoSurfaceObject;

extern_class!(
    #[unsafe(super(NSObject))]
    #[name = "_ANERequest"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct ANERequest;
);

extern_conformance!(
    unsafe impl NSObjectProtocol for ANERequest {}
);

impl ANERequest {
    pub fn set_completion_handler(&self, handler: &Block<dyn Fn(Bool, *mut NSError)>) {
        unsafe { msg_send![self, setCompletionHandler: handler] }
    }

    pub fn with_multiple_io(
        input_surfaces: &[&AneIoSurfaceObject],
        output_surfaces: &[&AneIoSurfaceObject],
        input_indices: &[u32],
        output_indices: &[u32],
        shared_events: Option<&AnyObject>,
    ) -> Option<Retained<ANERequest>> {
        let zero = NSNumber::new_u32(0);

        let inputs = NSArray::from_retained_slice(
            &input_surfaces
                .iter()
                .map(|s| (*s).retain())
                .collect::<Vec<_>>(),
        );
        let outputs = NSArray::from_retained_slice(
            &output_surfaces
                .iter()
                .map(|s| (*s).retain())
                .collect::<Vec<_>>(),
        );
        let in_indices = NSArray::from_retained_slice(
            &input_indices
                .iter()
                .copied()
                .map(NSNumber::new_u32)
                .collect::<Vec<_>>(),
        );
        let out_indices = NSArray::from_retained_slice(
            &output_indices
                .iter()
                .copied()
                .map(NSNumber::new_u32)
                .collect::<Vec<_>>(),
        );

        unsafe {
            msg_send![Self::class(),
                requestWithInputs: &*inputs,
                inputIndices: &*in_indices,
                outputs: &*outputs,
                outputIndices: &*out_indices,
                weightsBuffer: Option::<&AnyObject>::None,
                perfStats: Option::<&AnyObject>::None,
                procedureIndex: &*zero,
                sharedEvents: shared_events]
        }
    }
}
