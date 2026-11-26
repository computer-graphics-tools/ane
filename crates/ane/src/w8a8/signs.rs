//! Deterministic program and layer rotation signs.

use super::{INPUT_CHANNELS, kernels::W4};

const SEED: u64 = 0x5eed_a11e_2026_1103;

/// An RHT int4 matrix and the signs for `R_out W R_in^T` (signed block-32 Hadamard rotations).
/// `w4.layout` describes the source code order; staging preserves its logical rows and columns.
pub struct RhtW4<'a> {
    pub w4: W4,
    /// Input signs, one per column.
    pub input_signs: &'a [i32],
    /// Output signs, one per row.
    pub output_signs: &'a [i32],
}

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

pub fn pack_signs() -> Vec<f32> {
    random_signs(INPUT_CHANNELS, 1)
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
}

impl LayerSigns {
    pub fn new(up: &RhtW4, down: Option<&RhtW4>) -> Self {
        let to_f32 =
            |signs: &[i32]| -> Vec<f32> { signs.iter().map(|sign| *sign as f32).collect() };
        let mut signs = Self {
            pre: up
                .input_signs
                .iter()
                .zip(pack_signs())
                .map(|(sign, pack)| *sign as f32 * pack)
                .collect(),
            output: to_f32(down.unwrap_or(up).output_signs),
            gate: Vec::new(),
            pre_bits: Vec::new(),
        };
        if let Some(down) = down {
            let (value, gate) = up.output_signs.split_at(down.input_signs.len());
            let pre_signs: Vec<f32> = down
                .input_signs
                .iter()
                .zip(value)
                .map(|(down, value)| (down * value) as f32)
                .collect();
            signs.gate = to_f32(gate);
            signs.pre_bits = sign_bits(&pre_signs);
        }
        signs
    }
}
