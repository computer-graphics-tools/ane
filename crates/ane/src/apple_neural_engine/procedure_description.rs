use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ProcedureDescription {
    #[serde(rename = "ANEFModelProcedureID")]
    pub id: u32,
    #[serde(rename = "ANEFModelInputSymbolIndexArray")]
    pub inputs: Box<[u32]>,
    #[serde(rename = "ANEFModelOutputSymbolIndexArray")]
    pub outputs: Box<[u32]>,
}
