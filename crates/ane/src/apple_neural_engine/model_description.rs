use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ModelDescription {
    #[serde(rename = "kANEFModelInputSymbolsArrayKey")]
    pub input_symbols: Box<[String]>,
    #[serde(rename = "kANEFModelOutputSymbolsArrayKey")]
    pub output_symbols: Box<[String]>,
}
