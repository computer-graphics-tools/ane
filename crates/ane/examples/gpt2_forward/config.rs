use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Gpt2Config {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    #[allow(dead_code)]
    pub n_positions: usize,
    #[serde(default = "default_layer_norm_epsilon")]
    pub layer_norm_epsilon: f64,
}

fn default_layer_norm_epsilon() -> f64 {
    1e-5
}

impl Gpt2Config {
    pub fn head_size(&self) -> usize {
        self.n_embd / self.n_head
    }
}
