use obfstr::{obfcstr, obfstr};
use std::sync::OnceLock;

use objc2_foundation::{NSBundle, NSString};

use crate::UnavailableInterface;
use crate::apple_neural_engine::AneError;
use crate::unavailable_interface::ensure_interfaces;

pub fn ensure_model_interfaces() -> Result<(), AneError> {
    static AVAILABILITY: OnceLock<Result<(), AneError>> = OnceLock::new();
    ensure(&AVAILABILITY, || {
        ensure_interfaces(&[
            (
                obfcstr!(c"_ANEModel"),
                &[obfcstr!(c"modelAtURL:key:")],
                &[obfcstr!(c"modelAttributes")],
            ),
            (
                obfcstr!(c"_ANEClient"),
                &[obfcstr!(c"sharedConnection")],
                &[
                    obfcstr!(c"loadModel:options:qos:error:"),
                    obfcstr!(c"evaluateWithModel:options:request:qos:error:"),
                    obfcstr!(c"unloadModel:options:qos:error:"),
                    obfcstr!(c"mapIOSurfacesWithModel:request:cacheInference:error:"),
                    obfcstr!(c"unmapIOSurfacesWithModel:request:"),
                ],
            ),
            (
                obfcstr!(c"_ANEIOSurfaceObject"),
                &[obfcstr!(c"objectWithIOSurface:")],
                &[],
            ),
            (
                obfcstr!(c"_ANERequest"),
                &[obfcstr!(
                    c"requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:"
                )],
                &[],
            ),
        ])
    })
}

pub fn ensure_device_interfaces() -> Result<(), AneError> {
    static AVAILABILITY: OnceLock<Result<(), AneError>> = OnceLock::new();
    ensure(&AVAILABILITY, || {
        ensure_interfaces(&[(
            obfcstr!(c"_ANEDeviceInfo"),
            &[
                obfcstr!(c"hasANE"),
                obfcstr!(c"numANEs"),
                obfcstr!(c"numANECores"),
            ],
            &[],
        )])
    })
}

pub fn ensure_event_interfaces() -> Result<(), AneError> {
    static AVAILABILITY: OnceLock<Result<(), AneError>> = OnceLock::new();
    ensure(&AVAILABILITY, || {
        ensure_interfaces(&[
            (
                obfcstr!(c"_ANESharedWaitEvent"),
                &[obfcstr!(c"waitEventWithValue:sharedEvent:")],
                &[],
            ),
            (
                obfcstr!(c"_ANESharedSignalEvent"),
                &[obfcstr!(
                    c"signalEventWithValue:symbolIndex:eventType:sharedEvent:"
                )],
                &[],
            ),
            (
                obfcstr!(c"_ANESharedEvents"),
                &[obfcstr!(c"sharedEventsWithSignalEvents:waitEvents:")],
                &[],
            ),
            (
                obfcstr!(c"_ANERequest"),
                &[],
                &[
                    obfcstr!(c"setCompletionHandler:"),
                    obfcstr!(c"setSharedEvents:"),
                ],
            ),
        ])
    })
}

fn ensure(
    availability: &OnceLock<Result<(), AneError>>,
    interfaces: impl FnOnce() -> Result<(), UnavailableInterface>,
) -> Result<(), AneError> {
    availability
        .get_or_init(|| {
            let bundle = NSBundle::bundleWithPath(&NSString::from_str(obfstr!(
                "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"
            )))
            .ok_or(AneError::FrameworkNotFound)?;
            unsafe { bundle.loadAndReturnError() }?;
            Ok(interfaces()?)
        })
        .clone()
}
