pub struct LayerWeights {
    pub ln1_weight: Box<[f32]>,
    pub ln1_bias: Box<[f32]>,
    pub qkv_weight: Box<[f32]>,
    pub qkv_bias: Box<[f32]>,
    pub attn_proj_weight: Box<[f32]>,
    pub attn_proj_bias: Box<[f32]>,
    pub ln2_weight: Box<[f32]>,
    pub ln2_bias: Box<[f32]>,
    pub fc_weight: Box<[f32]>,
    pub fc_bias: Box<[f32]>,
    pub fc_proj_weight: Box<[f32]>,
    pub fc_proj_bias: Box<[f32]>,
}
