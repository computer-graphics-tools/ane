use ane::{Executable, Graph, NSQualityOfService, Shape};

use crate::config::Gpt2Config;
use crate::weights::LayerWeights;

pub struct PrefillLayer {
    pub attention: Executable,
    pub feed_forward: Executable,
}

pub struct DecodeLayer {
    pub attention: Executable,
    pub feed_forward: Executable,
}

pub struct CompiledExecutables {
    pub prefill: Box<[PrefillLayer]>,
    pub decode: Box<[DecodeLayer]>,
    pub lm_head: Executable,
}

pub const DECODE_SPATIAL_WIDTH: usize = 64;

fn scalar_shape() -> Shape {
    Shape { batch: 1, channels: 1, height: 1, width: 1 }
}

fn layer_norm(
    graph: &mut Graph,
    input: ane::Tensor,
    gamma: &[f32],
    beta: &[f32],
    embedding_dim: usize,
    epsilon: f64,
) -> ane::Tensor {
    let broadcast_scalar = scalar_shape();
    let inverse_dim = graph.constant_with_scalar(1.0 / embedding_dim as f32, broadcast_scalar);
    let epsilon_constant = graph.constant_with_scalar(epsilon as f32, broadcast_scalar);
    let neg_half = graph.constant_with_scalar(-0.5, broadcast_scalar);
    let neg_one = graph.constant_with_scalar(-1.0, broadcast_scalar);
    let gamma_constant = graph.constant(gamma, Shape { batch: 1, channels: embedding_dim, height: 1, width: 1 });
    let beta_constant = graph.constant(beta, Shape { batch: 1, channels: embedding_dim, height: 1, width: 1 });

    let channel_sum = graph.reduce_sum(input, 1);
    let mean = graph.multiplication(channel_sum, inverse_dim);
    let negative_mean = graph.multiplication(mean, neg_one);
    let centered = graph.addition(input, negative_mean);
    let squared = graph.multiplication(centered, centered);
    let variance_sum = graph.reduce_sum(squared, 1);
    let variance = graph.multiplication(variance_sum, inverse_dim);
    let variance_plus_eps = graph.addition(variance, epsilon_constant);
    let rstd = graph.power(variance_plus_eps, neg_half);
    let normalized = graph.multiplication(centered, rstd);
    let scaled = graph.multiplication(normalized, gamma_constant);
    graph.addition(scaled, beta_constant)
}

fn gelu(graph: &mut Graph, input: ane::Tensor) -> ane::Tensor {
    let broadcast_scalar = scalar_shape();
    let half_constant = graph.constant_with_scalar(0.5, broadcast_scalar);
    let one_constant = graph.constant_with_scalar(1.0, broadcast_scalar);
    let gelu_coefficient = graph.constant_with_scalar(0.044715, broadcast_scalar);
    let sqrt_2_over_pi = graph.constant_with_scalar(0.7978845608028654, broadcast_scalar);

    let input_squared = graph.multiplication(input, input);
    let input_cubed = graph.multiplication(input_squared, input);
    let scaled_cube = graph.multiplication(gelu_coefficient, input_cubed);
    let inner_sum = graph.addition(input, scaled_cube);
    let tanh_argument = graph.multiplication(sqrt_2_over_pi, inner_sum);
    let tanh_result = graph.tanh(tanh_argument);
    let one_plus_tanh = graph.addition(one_constant, tanh_result);
    let half_input = graph.multiplication(half_constant, input);
    graph.multiplication(half_input, one_plus_tanh)
}

fn causal_mask(sequence_length: usize) -> Box<[f32]> {
    (0..sequence_length * sequence_length)
        .map(|flat_index| {
            let column = flat_index % sequence_length;
            let row = flat_index / sequence_length;
            if column <= row { 0.0 } else { -65504.0 }
        })
        .collect()
}

fn attention_body(
    graph: &mut Graph,
    normalized: ane::Tensor,
    layer_weights: &LayerWeights,
    config: &Gpt2Config,
    query_seq: usize,
    key_seq: usize,
    key_source: Option<ane::Tensor>,
    value_source: Option<ane::Tensor>,
    mask_tensor: Option<ane::Tensor>,
) -> (ane::Tensor, ane::Tensor, ane::Tensor) {
    let embedding_dim = config.n_embd;
    let num_heads = config.n_head;
    let head_size = config.head_size();

    let qkv_weight = graph.constant(&layer_weights.qkv_weight, Shape::spatial(3 * embedding_dim, 1, 1));
    let qkv = graph.convolution_2d_1x1(normalized, qkv_weight, None);
    let qkv_bias = graph.constant(
        &layer_weights.qkv_bias,
        Shape { batch: 1, channels: 3 * embedding_dim, height: 1, width: 1 },
    );
    let qkv = graph.addition(qkv, qkv_bias);

    let query_flat = graph.slice(qkv, [0, 0, 0, 0], [1, embedding_dim, 1, query_seq]);
    let key_new_flat = graph.slice(qkv, [0, embedding_dim, 0, 0], [1, embedding_dim, 1, query_seq]);
    let value_new_flat = graph.slice(qkv, [0, 2 * embedding_dim, 0, 0], [1, embedding_dim, 1, query_seq]);

    let query = graph.reshape(query_flat, Shape { batch: 1, channels: num_heads, height: head_size, width: query_seq });
    let transpose_hw = [0, 1, 3, 2];
    let query = graph.transpose(query, transpose_hw);

    let (key, value, attn_key_seq) = match (key_source, value_source) {
        (Some(k_cache), Some(v_cache)) => {
            let key = graph.reshape(k_cache, Shape { batch: 1, channels: num_heads, height: head_size, width: key_seq });
            let key = graph.transpose(key, transpose_hw);
            let value = graph.reshape(v_cache, Shape { batch: 1, channels: num_heads, height: head_size, width: key_seq });
            let value = graph.transpose(value, transpose_hw);
            (key, value, key_seq)
        }
        _ => {
            let key = graph.reshape(key_new_flat, Shape { batch: 1, channels: num_heads, height: head_size, width: query_seq });
            let key = graph.transpose(key, transpose_hw);
            let value = graph.reshape(value_new_flat, Shape { batch: 1, channels: num_heads, height: head_size, width: query_seq });
            let value = graph.transpose(value, transpose_hw);
            (key, value, query_seq)
        }
    };

    let scale = graph.constant_with_scalar(1.0 / (head_size as f32).sqrt(), scalar_shape());

    let scores = graph.matrix_multiplication(query, key, false, true);
    let scores = graph.multiplication(scores, scale);

    let scores = match mask_tensor {
        Some(mask) => graph.addition(scores, mask),
        None => {
            let mask = graph.constant(
                &causal_mask(query_seq),
                Shape { batch: 1, channels: 1, height: query_seq, width: attn_key_seq },
            );
            graph.addition(scores, mask)
        }
    };

    let probabilities = graph.soft_max(scores, -1);
    let attention = graph.matrix_multiplication(probabilities, value, false, false);

    let attention = graph.transpose(attention, transpose_hw);
    let attention = graph.reshape(attention, Shape::spatial(embedding_dim, 1, query_seq));

    let projection_weight = graph.constant(&layer_weights.attn_proj_weight, Shape::spatial(embedding_dim, 1, 1));
    let projection = graph.convolution_2d_1x1(attention, projection_weight, None);
    let projection_bias = graph.constant(
        &layer_weights.attn_proj_bias,
        Shape { batch: 1, channels: embedding_dim, height: 1, width: 1 },
    );
    let o_proj = graph.addition(projection, projection_bias);

    (o_proj, key_new_flat, value_new_flat)
}

fn ffn_body(
    graph: &mut Graph,
    input: ane::Tensor,
    layer_weights: &LayerWeights,
    config: &Gpt2Config,
) -> ane::Tensor {
    let embedding_dim = config.n_embd;

    let normalized = layer_norm(
        graph, input,
        &layer_weights.ln2_weight, &layer_weights.ln2_bias,
        embedding_dim, config.layer_norm_epsilon,
    );

    let fc_weight = graph.constant(&layer_weights.fc_weight, Shape::spatial(4 * embedding_dim, 1, 1));
    let hidden = graph.convolution_2d_1x1(normalized, fc_weight, None);
    let fc_bias = graph.constant(
        &layer_weights.fc_bias,
        Shape { batch: 1, channels: 4 * embedding_dim, height: 1, width: 1 },
    );
    let hidden = graph.addition(hidden, fc_bias);
    let hidden = gelu(graph, hidden);

    let projection_weight = graph.constant(&layer_weights.fc_proj_weight, Shape::spatial(embedding_dim, 1, 1));
    let projection = graph.convolution_2d_1x1(hidden, projection_weight, None);
    let projection_bias = graph.constant(
        &layer_weights.fc_proj_bias,
        Shape { batch: 1, channels: embedding_dim, height: 1, width: 1 },
    );
    let ffn_output = graph.addition(projection, projection_bias);
    graph.addition(ffn_output, input)
}

pub fn build_prefill_attention(
    layer_weights: &LayerWeights,
    config: &Gpt2Config,
    sequence_length: usize,
) -> Result<Executable, ane::Error> {
    let embedding_dim = config.n_embd;
    let mut graph = Graph::new();
    let input = graph.placeholder(Shape::spatial(embedding_dim, 1, sequence_length));

    let normalized = layer_norm(
        &mut graph, input,
        &layer_weights.ln1_weight, &layer_weights.ln1_bias,
        embedding_dim, config.layer_norm_epsilon,
    );

    let (o_proj, key_flat, value_flat) = attention_body(
        &mut graph, normalized, layer_weights, config,
        sequence_length, sequence_length, None, None, None,
    );

    let o_proj_residual = graph.addition(o_proj, input);
    let _output = graph.concat(&[o_proj_residual, key_flat, value_flat], 1);
    graph.compile(NSQualityOfService::Default)
}

pub fn build_prefill_feed_forward(
    layer_weights: &LayerWeights,
    config: &Gpt2Config,
    sequence_length: usize,
) -> Result<Executable, ane::Error> {
    let embedding_dim = config.n_embd;
    let mut graph = Graph::new();
    let input = graph.placeholder(Shape::spatial(embedding_dim, 1, sequence_length));
    let _output = ffn_body(&mut graph, input, layer_weights, config);
    graph.compile(NSQualityOfService::Default)
}

pub fn build_decode_attention(
    layer_weights: &LayerWeights,
    config: &Gpt2Config,
    max_sequence_length: usize,
) -> Result<Executable, ane::Error> {
    let embedding_dim = config.n_embd;
    let mut graph = Graph::new();

    let x_padded = graph.placeholder(Shape::spatial(embedding_dim, 1, DECODE_SPATIAL_WIDTH));
    let k_cache = graph.placeholder(Shape::spatial(embedding_dim, 1, max_sequence_length));
    let v_cache = graph.placeholder(Shape::spatial(embedding_dim, 1, max_sequence_length));
    let mask = graph.placeholder(Shape { batch: 1, channels: 1, height: DECODE_SPATIAL_WIDTH, width: max_sequence_length });

    let normalized = layer_norm(
        &mut graph, x_padded,
        &layer_weights.ln1_weight, &layer_weights.ln1_bias,
        embedding_dim, config.layer_norm_epsilon,
    );

    let (o_proj, key_new_flat, value_new_flat) = attention_body(
        &mut graph, normalized, layer_weights, config,
        DECODE_SPATIAL_WIDTH, max_sequence_length,
        Some(k_cache), Some(v_cache), Some(mask),
    );

    let o_proj_residual = graph.addition(o_proj, x_padded);
    let _output = graph.concat(&[o_proj_residual, key_new_flat, value_new_flat], 1);
    graph.compile(NSQualityOfService::Default)
}

pub fn build_decode_feed_forward(
    layer_weights: &LayerWeights,
    config: &Gpt2Config,
) -> Result<Executable, ane::Error> {
    let embedding_dim = config.n_embd;
    let mut graph = Graph::new();
    let input = graph.placeholder(Shape::spatial(embedding_dim, 1, DECODE_SPATIAL_WIDTH));
    let _output = ffn_body(&mut graph, input, layer_weights, config);
    graph.compile(NSQualityOfService::Default)
}

pub fn build_lm_head(
    ln_f_weight: &[f32],
    ln_f_bias: &[f32],
    wte: &[f32],
    config: &Gpt2Config,
) -> Result<Executable, ane::Error> {
    let embedding_dim = config.n_embd;
    let vocab_size = config.vocab_size;
    let mut graph = Graph::new();

    let input = graph.placeholder(Shape::spatial(embedding_dim, 1, DECODE_SPATIAL_WIDTH));
    let normalized = layer_norm(
        &mut graph, input,
        ln_f_weight, ln_f_bias,
        embedding_dim, config.layer_norm_epsilon,
    );
    let wte_weight = graph.constant(wte, Shape::spatial(vocab_size, 1, 1));
    let _output = graph.convolution_2d_1x1(normalized, wte_weight, None);
    graph.compile(NSQualityOfService::Default)
}
