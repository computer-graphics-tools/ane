use crate::WeightDataType;

/// Codebook values of a palettized weight, ordered group by group, entry by entry.
#[derive(Clone, Copy, Debug)]
pub enum Palette<'a> {
    /// Float16 palette values.
    Float(&'a [f32]),
    /// Int8 or UInt8 codes dequantized per group as `scale * (code - zero_point)`.
    Quantized {
        /// Int8 or UInt8.
        data_type: WeightDataType,
        /// Palette codes packed in `data_type`.
        codes: &'a [u8],
        /// One scale per palette group.
        scales: &'a [f32],
        /// One zero point per palette group, packed in `data_type`.
        zero_points: Option<&'a [u8]>,
    },
}
