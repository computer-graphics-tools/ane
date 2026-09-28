use safetensors::SafeTensors;

use crate::compiled_executables::{self, CompiledExecutables};
use crate::config::Gpt2Config;
use crate::error::Error;
use crate::spinner::Spinner;
use crate::weights::{self, ModelWeights};

pub struct CompiledModel {
    pub config: Gpt2Config,
    pub token_embeddings: Box<[f32]>,
    pub position_embeddings: Box<[f32]>,
    pub executables: CompiledExecutables,
    pub max_sequence_length: usize,
    pub padded_prompt_length: usize,
}

impl CompiledModel {
    pub fn from_safetensors(
        config: Gpt2Config,
        bytes: Vec<u8>,
        padded_prompt_length: usize,
        max_sequence_length: usize,
    ) -> Result<Self, Error> {
        let spinner = Spinner::new("Loading weights");
        let weights = weights::load_weights(&SafeTensors::deserialize(&bytes)?, &config);
        drop(bytes);
        spinner.finish("Loaded weights");
        Self::from_weights(config, weights, padded_prompt_length, max_sequence_length)
    }

    pub fn from_weights(
        config: Gpt2Config,
        weights: ModelWeights,
        padded_prompt_length: usize,
        max_sequence_length: usize,
    ) -> Result<Self, Error> {
        if config.n_layer == 0
            || config.n_head == 0
            || !config.n_embd.is_multiple_of(config.n_head)
            || config.head_size() < 64
            || !config.head_size().is_multiple_of(32)
            || padded_prompt_length < 64
            || !padded_prompt_length.is_multiple_of(64)
            || max_sequence_length < padded_prompt_length
            || !max_sequence_length.is_multiple_of(64)
            || max_sequence_length > config.n_positions
            || config.vocab_size < 64
            || weights.layers.len() != config.n_layer
        {
            return Err(Error::Input(
                "unsupported model dimensions or context length",
            ));
        }
        let mut spinner = Spinner::new("Compiling prefill");
        let prefill = compiled_executables::build(
            &weights,
            &config,
            padded_prompt_length,
            max_sequence_length,
        )?;
        spinner.update("Compiling decode");
        let decode = compiled_executables::build(&weights, &config, 1, max_sequence_length)?;
        spinner.finish("Compiled ANE model");
        Ok(Self {
            config,
            token_embeddings: weights.wte,
            position_embeddings: weights.wpe,
            executables: CompiledExecutables { prefill, decode },
            max_sequence_length,
            padded_prompt_length,
        })
    }
}
