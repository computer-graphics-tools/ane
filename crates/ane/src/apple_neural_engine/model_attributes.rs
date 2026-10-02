use std::sync::Arc;

use objc2_foundation::{NSDictionary, NSPropertyListFormat, NSPropertyListSerialization};
use serde::Deserialize;

use crate::apple_neural_engine::{AneError, ModelDescription, NetworkStatus};

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ModelAttributes {
    #[serde(rename = "ANEFModelDescription")]
    pub description: ModelDescription,
    #[serde(rename = "NetworkStatusList")]
    pub networks: Box<[NetworkStatus]>,
}

impl ModelAttributes {
    pub fn new(attributes: &NSDictionary) -> Result<Self, AneError> {
        let property_list = unsafe {
            NSPropertyListSerialization::dataWithPropertyList_format_options_error(
                attributes,
                NSPropertyListFormat::BinaryFormat_v1_0,
                0,
            )
        }?;
        plist::from_bytes(&property_list.to_vec())
            .map_err(|error| AneError::Metadata(Arc::new(error)))
    }
}
