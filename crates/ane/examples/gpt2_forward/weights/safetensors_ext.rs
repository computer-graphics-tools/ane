use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};

pub fn tensor_to_f32(safetensors: &SafeTensors, name: &str) -> Box<[f32]> {
    let tensor = safetensors
        .tensor(name)
        .unwrap_or_else(|_| panic!("tensor not found: {name}"));
    let bytes = tensor.data();
    match tensor.dtype() {
        Dtype::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
            .collect(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| f16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
            .collect(),
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
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
