use ane::{Shape, TensorData};

use crate::compiled_model::CompiledModel;
use crate::executables::DECODE_SPATIAL_WIDTH;
use crate::kv_cache::KvCache;

pub struct Session<'model> {
    model: &'model CompiledModel,
    kv_cache: KvCache,
    prefill_hidden: TensorData,
    prefill_attn_delta: TensorData,
    prefill_ffn_delta: TensorData,
    decode_hidden: TensorData,
    decode_attn_delta: TensorData,
    decode_ffn_delta: TensorData,
    decode_mask: TensorData,
    lm_head_output: TensorData,
    logits: Vec<f32>,
    position: usize,
}

impl<'model> Session<'model> {
    pub fn new(model: &'model CompiledModel, padded_prompt_length: usize) -> Self {
        let embedding_dim = model.config.n_embd;
        let max_sequence_length = model.max_sequence_length;

        let prefill_hidden_shape = Shape::spatial(embedding_dim, 1, padded_prompt_length);
        let prefill_attn_shape = Shape::spatial(3 * embedding_dim, 1, padded_prompt_length);

        let decode_hidden_shape = Shape::spatial(embedding_dim, 1, DECODE_SPATIAL_WIDTH);
        let decode_attn_shape = Shape::spatial(3 * embedding_dim, 1, DECODE_SPATIAL_WIDTH);
        let decode_mask_shape = Shape { batch: 1, channels: 1, height: DECODE_SPATIAL_WIDTH, width: max_sequence_length };
        let lm_head_output_shape = Shape::spatial(model.config.vocab_size, 1, DECODE_SPATIAL_WIDTH);

        Self {
            model,
            kv_cache: KvCache::new(model.config.n_layer, embedding_dim, max_sequence_length),
            prefill_hidden: TensorData::new(prefill_hidden_shape),
            prefill_attn_delta: TensorData::new(prefill_attn_shape),
            prefill_ffn_delta: TensorData::new(prefill_hidden_shape),
            decode_hidden: TensorData::new(decode_hidden_shape),
            decode_attn_delta: TensorData::new(decode_attn_shape),
            decode_ffn_delta: TensorData::new(decode_hidden_shape),
            decode_mask: TensorData::new(decode_mask_shape),
            lm_head_output: TensorData::new(lm_head_output_shape),
            logits: vec![0.0; model.config.vocab_size],
            position: 0,
        }
    }

    pub fn prefill(&mut self, token_ids: &[u32], real_prompt_length: usize) -> &[f32] {
        let embedding_dim = self.model.config.n_embd;
        let sequence_length = token_ids.len();

        {
            let mut surface = self.prefill_hidden.as_f32_slice_mut();
            embedding_lookup_into(
                &mut surface, token_ids,
                &self.model.weights.wte, &self.model.weights.wpe, embedding_dim,
            );
        }

        for (layer_index, layer) in self.model.executables.prefill.iter().enumerate() {
            layer
                .attention
                .run(&[&self.prefill_hidden], &[&self.prefill_attn_delta])
                .unwrap_or_else(|error| panic!("prefill layer {layer_index} attention: {error}"));

            {
                let attn_slice = self.prefill_attn_delta.as_f32_slice();
                let o_proj_size = embedding_dim * sequence_length;

                let key_data = &attn_slice[o_proj_size..2 * o_proj_size];
                let value_data = &attn_slice[2 * o_proj_size..3 * o_proj_size];
                self.kv_cache.write_kv_sequence(layer_index, key_data, value_data, real_prompt_length, sequence_length);

                let mut hidden_surface = self.prefill_hidden.as_f32_slice_mut();
                hidden_surface[..o_proj_size].copy_from_slice(&attn_slice[..o_proj_size]);
            }

            layer
                .feed_forward
                .run(&[&self.prefill_hidden], &[&self.prefill_ffn_delta])
                .unwrap_or_else(|error| panic!("prefill layer {layer_index} ffn: {error}"));
            std::mem::swap(&mut self.prefill_hidden, &mut self.prefill_ffn_delta);
        }

        self.position = real_prompt_length;
        self.kv_cache.position = real_prompt_length;

        {
            let hidden_slice = self.prefill_hidden.as_f32_slice();
            let mut lm_input = self.decode_hidden.as_f32_slice_mut();
            for dim_index in 0..embedding_dim {
                lm_input[dim_index * DECODE_SPATIAL_WIDTH] =
                    hidden_slice[dim_index * sequence_length + (real_prompt_length - 1)];
            }
        }

        {
            let mut mask_surface = self.decode_mask.as_f32_slice_mut();
            mask_surface.fill(-65504.0);
            for col in 0..self.position {
                mask_surface[col] = 0.0;
            }
        }

        self.run_lm_head()
    }

    pub fn decode_step(&mut self, token: u32) -> &[f32] {
        let embedding_dim = self.model.config.n_embd;

        {
            let mut hidden_surface = self.decode_hidden.as_f32_slice_mut();
            let token_index = token as usize;
            for dim_index in 0..embedding_dim {
                hidden_surface[dim_index * DECODE_SPATIAL_WIDTH] =
                    self.model.weights.wte[token_index * embedding_dim + dim_index]
                        + self.model.weights.wpe[self.position * embedding_dim + dim_index];
            }
        }

        {
            let mut mask_surface = self.decode_mask.as_f32_slice_mut();
            mask_surface[self.position] = 0.0;
        }

        for (layer_index, layer) in self.model.executables.decode.iter().enumerate() {
            layer
                .attention
                .run(
                    &[&self.decode_hidden, &self.kv_cache.keys[layer_index], &self.kv_cache.values[layer_index], &self.decode_mask],
                    &[&self.decode_attn_delta],
                )
                .unwrap_or_else(|error| panic!("decode layer {layer_index} attention: {error}"));

            {
                let attn_slice = self.decode_attn_delta.as_f32_slice();
                self.kv_cache.write_kv_from_attn(layer_index, &attn_slice, DECODE_SPATIAL_WIDTH, self.position);

                let mut hidden_surface = self.decode_hidden.as_f32_slice_mut();
                hidden_surface.copy_from_slice(&attn_slice[..embedding_dim * DECODE_SPATIAL_WIDTH]);
            }

            layer
                .feed_forward
                .run(&[&self.decode_hidden], &[&self.decode_ffn_delta])
                .unwrap_or_else(|error| panic!("decode layer {layer_index} ffn: {error}"));

            std::mem::swap(&mut self.decode_hidden, &mut self.decode_ffn_delta);
        }

        self.position += 1;
        self.kv_cache.position = self.position;

        self.run_lm_head()
    }

    fn run_lm_head(&mut self) -> &[f32] {
        self.model.executables.lm_head
            .run(&[&self.decode_hidden], &[&self.lm_head_output])
            .expect("lm_head");

        let vocab_size = self.model.config.vocab_size;
        let output_slice = self.lm_head_output.as_f32_slice();
        for v in 0..vocab_size {
            self.logits[v] = output_slice[v * DECODE_SPATIAL_WIDTH];
        }
        &self.logits
    }
}

fn embedding_lookup_into(
    destination: &mut [f32],
    token_ids: &[u32],
    token_embeddings: &[f32],
    position_embeddings: &[f32],
    embedding_dim: usize,
) {
    let sequence_length = token_ids.len();
    for seq_index in 0..sequence_length {
        let token = token_ids[seq_index] as usize;
        for dim_index in 0..embedding_dim {
            destination[dim_index * sequence_length + seq_index] =
                token_embeddings[token * embedding_dim + dim_index]
                    + position_embeddings[seq_index * embedding_dim + dim_index];
        }
    }
}
