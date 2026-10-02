use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct SurfaceLayout {
    pub name: String,
    #[serde(rename = "Type")]
    pub storage_type: String,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub channels: Option<usize>,
    pub batches: Option<usize>,
    pub depth: Option<usize>,
    pub batch_stride: Option<usize>,
    pub plane_stride: Option<usize>,
    pub row_stride: Option<usize>,
}
