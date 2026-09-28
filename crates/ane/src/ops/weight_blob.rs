use std::sync::Arc;

#[derive(Clone)]
pub struct WeightBlob {
    pub data: Arc<[u8]>,
}

impl WeightBlob {
    pub fn zeros(count: usize) -> Self {
        Self {
            data: vec![0u8; count * 2].into(),
        }
    }

    pub fn from_f32(values: &[f32]) -> Self {
        let mut bytes = vec![0u8; values.len() * 2];
        for (index, &value) in values.iter().enumerate() {
            let f16 = super::weights::f32_to_f16(value);
            bytes[index * 2] = (f16 & 0xFF) as u8;
            bytes[index * 2 + 1] = (f16 >> 8) as u8;
        }
        Self { data: bytes.into() }
    }

    pub fn from_f16_bytes(bytes: Box<[u8]>) -> Self {
        Self { data: bytes.into() }
    }
}
