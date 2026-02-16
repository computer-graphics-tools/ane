use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};

pub fn tensor_to_f32(safetensors: &SafeTensors, name: &str) -> Box<[f32]> {
    let tensor = safetensors
        .tensor(name)
        .unwrap_or_else(|_| panic!("tensor not found: {name}"));
    let bytes = tensor.data();
    match tensor.dtype() {
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        other => panic!("unsupported dtype: {other:?}"),
    }
}

pub fn tensor_to_f32_transposed(
    safetensors: &SafeTensors,
    name: &str,
    rows: usize,
    cols: usize,
) -> Box<[f32]> {
    let raw = tensor_to_f32(safetensors, name);
    assert_eq!(raw.len(), rows * cols, "shape mismatch for {name}");
    let mut transposed = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = raw[row * cols + col];
        }
    }
    transposed.into_boxed_slice()
}
