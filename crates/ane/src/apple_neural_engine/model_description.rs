use std::collections::BTreeMap;

use serde::Deserialize;

use crate::apple_neural_engine::ProcedureDescription;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ModelDescription {
    #[serde(rename = "kANEFModelInputSymbolsArrayKey")]
    pub input_symbols: Box<[String]>,
    #[serde(rename = "kANEFModelOutputSymbolsArrayKey")]
    pub output_symbols: Box<[String]>,
    #[serde(rename = "ANEFModelProcedures")]
    pub procedures: Box<[ProcedureDescription]>,
    #[serde(rename = "kANEFModelProcedureNameToIDMapKey")]
    pub procedure_ids: BTreeMap<String, u32>,
}
