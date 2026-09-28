use crate::config::Gpt2Config;
use crate::weights::{LayerWeights, ModelWeights};
use ane::{Executable, Graph, NSQualityOfService, State, Tensor};

pub struct CompiledExecutables {
    pub prefill: Executable,
    pub decode: Executable,
}

fn layer_norm(
    graph: &mut Graph,
    input: ane::Tensor,
    gamma: &[f32],
    beta: &[f32],
    embedding_dim: usize,
    epsilon: f64,
) -> ane::Tensor {
    let broadcast_scalar = [1; 4];
    let inverse_dim = graph.constant_with_scalar(1.0 / embedding_dim as f32, &broadcast_scalar);
    let epsilon_constant = graph.constant_with_scalar(epsilon as f32, &broadcast_scalar);
    let neg_half = graph.constant_with_scalar(-0.5, &broadcast_scalar);
    let neg_one = graph.constant_with_scalar(-1.0, &broadcast_scalar);
    let gamma_constant = graph.constant(gamma, &[1, embedding_dim, 1, 1]);
    let beta_constant = graph.constant(beta, &[1, embedding_dim, 1, 1]);

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
    let broadcast_scalar = [1; 4];
    let half_constant = graph.constant_with_scalar(0.5, &broadcast_scalar);
    let one_constant = graph.constant_with_scalar(1.0, &broadcast_scalar);
    let gelu_coefficient = graph.constant_with_scalar(0.044715, &broadcast_scalar);
    let sqrt_2_over_pi = graph.constant_with_scalar(0.797_884_6, &broadcast_scalar);

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

fn linear(
    graph: &mut Graph,
    input: Tensor,
    weights: &[f32],
    bias: Option<&[f32]>,
    channels: usize,
) -> Tensor {
    let weight = graph.constant(weights, &[1, channels, 1, 1]);
    let bias = bias.map(|b| graph.constant(b, &[1, channels, 1, 1]));
    graph.convolution_2d_1x1(input, weight, bias)
}

fn split_heads(graph: &mut Graph, input: Tensor, config: &Gpt2Config, sequence: usize) -> Tensor {
    let shaped = graph.reshape(input, &[1, config.n_head, config.head_size(), sequence]);
    graph.transpose(shaped, [0, 1, 3, 2])
}

fn layer(
    graph: &mut Graph,
    input: Tensor,
    mask: Tensor,
    position: Tensor,
    current: (usize, &LayerWeights),
    states: (&State, &State),
    config: &Gpt2Config,
) -> Tensor {
    let (index, weights) = current;
    let sequence = input.shape[3];
    let context = mask.shape[3];
    let e = config.n_embd;
    let normalized = layer_norm(
        graph,
        input,
        &weights.ln1_weight,
        &weights.ln1_bias,
        e,
        config.layer_norm_epsilon,
    );
    let qkv = linear(
        graph,
        normalized,
        &weights.qkv_weight,
        Some(&weights.qkv_bias),
        3 * e,
    );
    let q = graph.slice(qkv, [0, 0, 0, 0], [1, e, 1, sequence]);
    let k = graph.slice(qkv, [0, e, 0, 0], [1, e, 1, sequence]);
    let v = graph.slice(qkv, [0, 2 * e, 0, 0], [1, e, 1, sequence]);
    let q = split_heads(graph, q, config, sequence);
    let k = split_heads(graph, k, config, sequence);
    let v = split_heads(graph, v, config, sequence);
    let channel = index * config.n_head;
    let k = states.0.update_rows_at_channel(graph, k, position, channel);
    let k = graph.slice(
        k,
        [0, channel, 0, 0],
        [1, config.n_head, context, config.head_size()],
    );
    let v = states.1.update_rows_at_channel(graph, v, position, channel);
    let v = graph.slice(
        v,
        [0, channel, 0, 0],
        [1, config.n_head, context, config.head_size()],
    );
    let scores = graph.matrix_multiplication(q, k, false, true);
    let scores = graph.multiply_scalar(scores, 1.0 / (config.head_size() as f32).sqrt());
    let scores = graph.addition(scores, mask);
    let probabilities = graph.soft_max(scores, -1);
    let attention = graph.matrix_multiplication(probabilities, v, false, false);
    let attention = graph.transpose(attention, [0, 1, 3, 2]);
    let attention = graph.reshape(attention, &[1, e, 1, sequence]);
    let projection = linear(
        graph,
        attention,
        &weights.attn_proj_weight,
        Some(&weights.attn_proj_bias),
        e,
    );
    let residual = graph.addition(input, projection);
    let normalized = layer_norm(
        graph,
        residual,
        &weights.ln2_weight,
        &weights.ln2_bias,
        e,
        config.layer_norm_epsilon,
    );
    let hidden = linear(
        graph,
        normalized,
        &weights.fc_weight,
        Some(&weights.fc_bias),
        4 * e,
    );
    let hidden = gelu(graph, hidden);
    let projection = linear(
        graph,
        hidden,
        &weights.fc_proj_weight,
        Some(&weights.fc_proj_bias),
        e,
    );
    graph.addition(residual, projection)
}

pub fn build(
    weights: &ModelWeights,
    config: &Gpt2Config,
    sequence: usize,
    context: usize,
) -> Result<Executable, ane::Error> {
    let mut graph = Graph::new();
    let prefill = sequence > 1;
    let input = graph.placeholder(&if prefill {
        [1, config.n_embd, 1, sequence]
    } else {
        [1, 1, 1, config.n_embd]
    });
    let mask = graph.placeholder(&[1, 1, sequence, context]);
    let position = graph.integer_parameter();
    let selector = prefill.then(|| graph.placeholder(&[1, 1, 1, sequence]));
    let mut hidden = if prefill {
        input
    } else {
        graph.reshape(input, &[1, config.n_embd, 1, 1])
    };
    let state_shape = [
        1,
        weights.layers.len() * config.n_head,
        context,
        config.head_size(),
    ];
    let keys = State::new(&mut graph, &state_shape);
    let values = State::new(&mut graph, &state_shape);
    for current in weights.layers.iter().enumerate() {
        hidden = layer(
            &mut graph,
            hidden,
            mask,
            position,
            current,
            (&keys, &values),
            config,
        );
    }
    if let Some(selector) = selector {
        hidden = graph.multiplication(hidden, selector);
        hidden = graph.reduce_sum(hidden, 3);
    }
    let normalized = layer_norm(
        &mut graph,
        hidden,
        &weights.ln_f_weight,
        &weights.ln_f_bias,
        config.n_embd,
        config.layer_norm_epsilon,
    );
    let logits = linear(
        &mut graph,
        normalized,
        &weights.wte,
        None,
        config.vocab_size,
    );
    graph.reshape(logits, &[1, 1, 1, config.vocab_size]);
    graph.compile(NSQualityOfService::Default)
}
