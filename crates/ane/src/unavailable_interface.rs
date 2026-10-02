use std::{ffi::CStr, sync::Arc};

use objc2::runtime::{AnyClass, Sel};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "unavailable Objective-C interface {}{}",
    .class.to_string_lossy(),
    .selector.as_ref().map(|selector| format!(" -{}", selector.to_string_lossy())).unwrap_or_default()
)]
pub struct UnavailableInterface {
    class: Arc<CStr>,
    selector: Option<Arc<CStr>>,
}

impl UnavailableInterface {
    pub fn class(&self) -> &CStr {
        &self.class
    }
    pub fn selector(&self) -> Option<&CStr> {
        self.selector.as_deref()
    }
}

pub fn ensure_interfaces(
    interfaces: &[(&CStr, &[&CStr], &[&CStr])],
) -> Result<(), UnavailableInterface> {
    for &(class, class_methods, instance_methods) in interfaces {
        let receiver = AnyClass::get(class).ok_or_else(|| UnavailableInterface {
            class: class.into(),
            selector: None,
        })?;
        let methods = class_methods
            .iter()
            .map(|selector| (receiver.metaclass(), selector))
            .chain(instance_methods.iter().map(|selector| (receiver, selector)));
        for (receiver, &selector) in methods {
            if !receiver.responds_to(Sel::register(selector)) {
                return Err(UnavailableInterface {
                    class: class.into(),
                    selector: Some(selector.into()),
                });
            }
        }
    }
    Ok(())
}
