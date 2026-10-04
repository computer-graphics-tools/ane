use crate::ops::Palette;

/// `2^bits` palette entries per group of `group_shape` weights. With `vector_axis`, every entry
/// is a vector of consecutive weights along that axis and each index selects a whole vector.
#[derive(Clone, Copy, Debug)]
pub struct Palettization<'a, const RANK: usize> {
    /// Index width: 1, 2, 3, 4, 6 or 8 bits.
    pub bits: usize,
    /// Number of palette groups along every axis; each axis length must divide by it.
    pub group_shape: [usize; RANK],
    /// Axis along which every palette entry spans consecutive weights.
    pub vector_axis: Option<usize>,
    /// Palette entries, group by group.
    pub palette: Palette<'a>,
}

impl<'a, const RANK: usize> Palettization<'a, RANK> {
    /// One float palette for the whole tensor.
    pub fn new(bits: usize, palette: &'a [f32]) -> Self {
        Self {
            bits,
            group_shape: [1; RANK],
            vector_axis: None,
            palette: Palette::Float(palette),
        }
    }
}
