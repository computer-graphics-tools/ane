use safetensors::SafeTensors;

use crate::config::Gpt2Config;
use crate::executables::{
    self, CompiledExecutables, DecodeLayer, PrefillLayer,
};
use crate::spinner::Spinner;
use crate::weights::{self, ModelWeights};

pub struct CompiledModel {
    pub config: Gpt2Config,
    pub weights: ModelWeights,
    pub executables: CompiledExecutables,
    pub max_sequence_length: usize,
}

impl CompiledModel {
    pub fn from_safetensors(
        config: Gpt2Config,
        safetensors: &SafeTensors,
        padded_prompt_length: usize,
        max_sequence_length: usize,
    ) -> Result<Self, ane::Error> {
        let mut spinner = Spinner::new("Loading weights");
        let model_weights = weights::load_weights(safetensors, &config);
        let num_layers = config.n_layer;

        let prefill: Box<[PrefillLayer]> = model_weights.layers.iter().enumerate()
            .map(|(layer_index, layer_weights)| {
                spinner.update(&format!("Compiling prefill layer {}/{num_layers}", layer_index + 1));
                Ok(PrefillLayer {
                    attention: executables::build_prefill_attention(layer_weights, &config, padded_prompt_length)?,
                    feed_forward: executables::build_prefill_feed_forward(layer_weights, &config, padded_prompt_length)?,
                })
            })
            .collect::<Result<_, ane::Error>>()?;

        let decode: Box<[DecodeLayer]> = model_weights.layers.iter().enumerate()
            .map(|(layer_index, layer_weights)| {
                spinner.update(&format!("Compiling decode layer {}/{num_layers}", layer_index + 1));
                Ok(DecodeLayer {
                    attention: executables::build_decode_attention(layer_weights, &config, max_sequence_length)?,
                    feed_forward: executables::build_decode_feed_forward(layer_weights, &config)?,
                })
            })
            .collect::<Result<_, ane::Error>>()?;

        spinner.update("Compiling LM head");
        let lm_head = executables::build_lm_head(
            &model_weights.ln_f_weight, &model_weights.ln_f_bias,
            &model_weights.wte, &config,
        )?;

        spinner.finish("Compiled ANE model");

        Ok(Self {
            config,
            weights: model_weights,
            executables: CompiledExecutables { prefill, decode, lm_head },
            max_sequence_length,
        })
    }
}
