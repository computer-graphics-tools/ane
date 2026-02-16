use safetensors::SafeTensors;

use crate::config::Gpt2Config;

use super::layer_weights::LayerWeights;
use super::safetensors_ext::{tensor_to_f32, tensor_to_f32_transposed};

pub struct ModelWeights {
    pub wte: Box<[f32]>,
    pub wpe: Box<[f32]>,
    pub layers: Box<[LayerWeights]>,
    pub ln_f_weight: Box<[f32]>,
    pub ln_f_bias: Box<[f32]>,
}

pub fn load_weights(safetensors: &SafeTensors, config: &Gpt2Config) -> ModelWeights {
    let embedding_dim = config.n_embd;

    let wte = tensor_to_f32(safetensors, "wte.weight");
    let wpe = tensor_to_f32(safetensors, "wpe.weight");

    let layers: Box<[LayerWeights]> = (0..config.n_layer)
        .map(|layer_index| {
            let layer_prefix = format!("h.{layer_index}");
            LayerWeights {
                ln1_weight: tensor_to_f32(safetensors, &format!("{layer_prefix}.ln_1.weight")),
                ln1_bias: tensor_to_f32(safetensors, &format!("{layer_prefix}.ln_1.bias")),
                qkv_weight: tensor_to_f32_transposed(
                    safetensors,
                    &format!("{layer_prefix}.attn.c_attn.weight"),
                    embedding_dim,
                    3 * embedding_dim,
                ),
                qkv_bias: tensor_to_f32(safetensors, &format!("{layer_prefix}.attn.c_attn.bias")),
                attn_proj_weight: tensor_to_f32_transposed(
                    safetensors,
                    &format!("{layer_prefix}.attn.c_proj.weight"),
                    embedding_dim,
                    embedding_dim,
                ),
                attn_proj_bias: tensor_to_f32(safetensors, &format!("{layer_prefix}.attn.c_proj.bias")),
                ln2_weight: tensor_to_f32(safetensors, &format!("{layer_prefix}.ln_2.weight")),
                ln2_bias: tensor_to_f32(safetensors, &format!("{layer_prefix}.ln_2.bias")),
                fc_weight: tensor_to_f32_transposed(
                    safetensors,
                    &format!("{layer_prefix}.mlp.c_fc.weight"),
                    embedding_dim,
                    4 * embedding_dim,
                ),
                fc_bias: tensor_to_f32(safetensors, &format!("{layer_prefix}.mlp.c_fc.bias")),
                fc_proj_weight: tensor_to_f32_transposed(
                    safetensors,
                    &format!("{layer_prefix}.mlp.c_proj.weight"),
                    4 * embedding_dim,
                    embedding_dim,
                ),
                fc_proj_bias: tensor_to_f32(safetensors, &format!("{layer_prefix}.mlp.c_proj.bias")),
            }
        })
        .collect();

    let ln_f_weight = tensor_to_f32(safetensors, "ln_f.weight");
    let ln_f_bias = tensor_to_f32(safetensors, "ln_f.bias");

    ModelWeights { wte, wpe, layers, ln_f_weight, ln_f_bias }
}
