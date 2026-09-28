pub use super::WeightBlob;

pub fn build_mil_weight_blob(blobs: &[&WeightBlob]) -> Box<[u8]> {
    let total_size = mil_blob_total_size(blobs);
    let mut out = vec![0u8; total_size];

    out[0] = 0x01;
    out[4] = 0x02;

    let mut cursor = 64usize;
    for blob in blobs {
        let byte_count = blob.data.len();
        let data_offset = cursor + 64;

        out[cursor] = 0xEF;
        out[cursor + 1] = 0xBE;
        out[cursor + 2] = 0xAD;
        out[cursor + 3] = 0xDE;
        out[cursor + 4] = 0x01;

        let size_bytes = (byte_count as u32).to_le_bytes();
        out[cursor + 8..cursor + 12].copy_from_slice(&size_bytes);

        let offset_bytes = (data_offset as u32).to_le_bytes();
        out[cursor + 16..cursor + 20].copy_from_slice(&offset_bytes);

        out[data_offset..data_offset + byte_count].copy_from_slice(&blob.data);

        cursor += 64 + byte_count;
    }

    out.into_boxed_slice()
}

pub fn mil_blob_chunk_offset(blobs: &[&WeightBlob], index: usize) -> u64 {
    let mut offset = 64u64;
    for blob in &blobs[..index] {
        offset += 64 + blob.data.len() as u64;
    }
    offset
}

fn mil_blob_total_size(blobs: &[&WeightBlob]) -> usize {
    64 + blobs.iter().map(|blob| 64 + blob.data.len()).sum::<usize>()
}

pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x007F_FFFF;

    if exponent <= 0 {
        if exponent < -10 {
            return sign as u16;
        }
        let shifted_mantissa = (mantissa | 0x0080_0000) >> (14 - exponent);
        return (sign | shifted_mantissa) as u16;
    }
    if exponent >= 31 {
        return (sign | 0x7C00) as u16;
    }
    (sign | ((exponent as u32) << 10) | (mantissa >> 13)) as u16
}
