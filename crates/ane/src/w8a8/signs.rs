//! Deterministic program and layer rotation signs.

use super::kernels::WeightLayer;

const SEED: u64 = 0x5eed_a11e_2026_1103;

fn random_signs(count: usize, salt: u64) -> Vec<f32> {
    let mut state = SEED ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// Rows of the orthonormal signed Hadamard matrix `D H_block`, in grouped-convolution order.
pub fn signed_hadamard_blocks(signs: &[f32], block: usize) -> Vec<f32> {
    let norm = 1.0 / (block as f32).sqrt();
    let mut out = Vec::with_capacity(signs.len() * block);
    for (channel, sign) in signs.iter().enumerate() {
        let row = channel % block;
        for column in 0..block {
            let hadamard = if (row & column).count_ones() % 2 == 0 {
                1.0
            } else {
                -1.0
            };
            out.push(sign * hadamard * norm);
        }
    }
    out
}

pub fn intermediate_signs(intermediate: usize) -> Vec<f32> {
    random_signs(intermediate, 2)
}

pub fn pack_signs(width: usize) -> Vec<f32> {
    random_signs(width, 1)
}

/// Pack 16 signs per word; bit `i` is set when sign `i` is negative.
pub fn sign_bits(signs: &[f32]) -> Vec<u16> {
    signs
        .chunks(16)
        .map(|group| {
            group.iter().enumerate().fold(0, |bits, (index, sign)| {
                bits | (u16::from(*sign < 0.0) << index)
            })
        })
        .collect()
}

/// Signs used to stage the up and (for an FFN) down weights in the program's basis.
pub struct LayerSigns {
    /// Input signs times the program's pack signs.
    pub pre: Vec<f32>,
    /// Output signs applied when unpacking.
    pub output: Vec<f32>,
    /// Gate signs for the FFN's `zg`; empty for a linear layer.
    pub gate: Vec<f32>,
    /// Down-input signs times value-output signs, packed as bits; empty for a linear layer.
    pub pre_bits: Vec<u16>,
    /// The same down-input signs in float form for compact-source staging.
    pub down_pre: Vec<f32>,
}

impl LayerSigns {
    pub fn new(up: &WeightLayer<'_>, down: Option<&WeightLayer<'_>>) -> Self {
        let to_f32 =
            |signs: &[i32]| -> Vec<f32> { signs.iter().map(|sign| *sign as f32).collect() };
        let identity = |count: usize| vec![1.0; count];
        let up_output = up.output_rotation.map_or_else(
            || identity(up.rows as usize),
            |rotation| to_f32(rotation.signs),
        );
        let output = match down {
            Some(down) => down.output_rotation.map_or_else(
                || identity(down.rows as usize),
                |rotation| to_f32(rotation.signs),
            ),
            None => up.output_rotation.map_or_else(
                || identity(up.rows as usize),
                |rotation| to_f32(rotation.signs),
            ),
        };
        let mut signs = Self {
            pre: up
                .input_rotation
                .signs
                .iter()
                .zip(pack_signs(up.input_rotation.signs.len()))
                .map(|(sign, pack)| *sign as f32 * pack)
                .collect(),
            output,
            gate: Vec::new(),
            pre_bits: Vec::new(),
            down_pre: Vec::new(),
        };
        if let Some(down) = down {
            let (value, gate) = up_output.split_at(down.input_rotation.signs.len());
            let pre_signs: Vec<f32> = down
                .input_rotation
                .signs
                .iter()
                .zip(value)
                .map(|(down, value)| *down as f32 * value)
                .collect();
            signs.gate = gate.to_vec();
            signs.pre_bits = sign_bits(&pre_signs);
            signs.down_pre = pre_signs;
        }
        signs
    }
}
