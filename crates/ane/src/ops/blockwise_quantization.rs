use crate::WeightDataType;

/// Weights dequantized per block as `scale * (data - offset)`. The offset is either a float
/// (`offsets`) or an integer zero point packed in `data_type` (`zero_points`).
#[derive(Clone, Copy, Debug)]
pub struct BlockwiseQuantization<'a, const RANK: usize> {
    /// Integer storage of the data: Int4, UInt4, Int8 or UInt8.
    pub data_type: WeightDataType,
    /// One scale per block, in row-major block order.
    pub scales: &'a [f32],
    /// Number of blocks along every axis; each axis length must divide by it.
    pub scale_shape: [usize; RANK],
    /// Float offsets, one per block.
    pub offsets: Option<&'a [f32]>,
    /// Integer zero points, one per block, packed in `data_type`.
    pub zero_points: Option<&'a [u8]>,
}

impl<'a, const RANK: usize> BlockwiseQuantization<'a, RANK> {
    /// Scales only, without offsets or zero points.
    pub fn new(data_type: WeightDataType, scales: &'a [f32], scale_shape: [usize; RANK]) -> Self {
        Self {
            data_type,
            scales,
            scale_shape,
            offsets: None,
            zero_points: None,
        }
    }
}
