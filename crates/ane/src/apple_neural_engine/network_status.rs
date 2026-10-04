use serde::Deserialize;

use crate::apple_neural_engine::SurfaceLayout;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct NetworkStatus {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "LiveInputList", default)]
    pub inputs: Box<[SurfaceLayout]>,
    #[serde(rename = "LiveStateList", default)]
    pub states: Box<[SurfaceLayout]>,
    #[serde(rename = "LiveInputParamList", default)]
    pub parameters: Box<[SurfaceLayout]>,
    #[serde(rename = "LiveOutputList")]
    pub outputs: Box<[SurfaceLayout]>,
}
