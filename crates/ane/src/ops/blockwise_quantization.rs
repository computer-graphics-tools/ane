use crate::WeightDataType;

#[derive(Clone, Copy, Debug)]
pub struct BlockwiseQuantization<'a, const RANK: usize> {
    pub data_type: WeightDataType,
    pub scales: &'a [f32],
    pub scale_shape: [usize; RANK],
    pub offsets: Option<&'a [f32]>,
}

impl<'a, const RANK: usize> BlockwiseQuantization<'a, RANK> {
    pub fn new(data_type: WeightDataType, scales: &'a [f32], scale_shape: [usize; RANK]) -> Self {
        Self {
            data_type,
            scales,
            scale_shape,
            offsets: None,
        }
    }
}
