/// Weight data stored as raw IEEE 754 fp16 bytes (2 bytes per element).
///
/// All three constructors produce the same in-memory layout so that
/// `build_mil_weight_blob` can copy bytes directly without any conversion.
#[derive(Clone)]
pub struct WeightBlob {
    pub data: Box<[u8]>,
}

impl WeightBlob {
    /// Allocate `count` fp16 zero elements (`count * 2` zero bytes).
    pub fn zeros(count: usize) -> Self {
        Self {
            data: vec![0u8; count * 2].into_boxed_slice(),
        }
    }

    /// Encode a slice of f32 values into fp16 bytes at construction time.
    pub fn from_f32(values: &[f32]) -> Self {
        let mut bytes = vec![0u8; values.len() * 2];
        for (index, &value) in values.iter().enumerate() {
            let f16 = f32_to_f16(value);
            bytes[index * 2]     = (f16 & 0xFF) as u8;
            bytes[index * 2 + 1] = (f16 >> 8) as u8;
        }
        Self {
            data: bytes.into_boxed_slice(),
        }
    }

    /// Take pre-encoded fp16 bytes directly (zero-copy for callers that already
    /// have fp16 weights, e.g. loaded from a checkpoint).
    pub fn from_f16_bytes(bytes: Box<[u8]>) -> Self {
        Self { data: bytes }
    }

    pub(crate) fn fp16_byte_count(&self) -> usize {
        self.data.len()
    }
}

/// Build the MIL weight blob binary for a sequence of blobs.
///
/// Binary layout:
/// - Bytes   0-63: global header (`0x01` at byte 0, `0x02` at byte 4, rest zero)
/// - For each blob i:
///   - chunk header (64 bytes): magic `0xDEADBEEF` LE, version `0x01`,
///     `data_size` (uint32 LE at +8), `data_offset` (uint32 LE at +16)
///   - fp16 weight bytes (direct copy from blob.data)
///
/// The BLOBFILE offset for blob i in MIL text = byte offset of chunk header i.
pub(crate) fn build_mil_weight_blob(blobs: &[&WeightBlob]) -> Box<[u8]> {
    let total_size = mil_blob_total_size(blobs);
    let mut out = vec![0u8; total_size];

    out[0] = 0x01;
    out[4] = 0x02;

    let mut cursor = 64usize;
    for blob in blobs {
        let fp16_size = blob.fp16_byte_count();
        let data_offset = cursor + 64;

        out[cursor]     = 0xEF;
        out[cursor + 1] = 0xBE;
        out[cursor + 2] = 0xAD;
        out[cursor + 3] = 0xDE;
        out[cursor + 4] = 0x01;

        let size_bytes = (fp16_size as u32).to_le_bytes();
        out[cursor + 8..cursor + 12].copy_from_slice(&size_bytes);

        let offset_bytes = (data_offset as u32).to_le_bytes();
        out[cursor + 16..cursor + 20].copy_from_slice(&offset_bytes);

        out[data_offset..data_offset + fp16_size].copy_from_slice(&blob.data);

        cursor += 64 + fp16_size;
    }

    out.into_boxed_slice()
}

/// Returns the BLOBFILE byte offset for blob `index` (points to chunk header).
pub(crate) fn mil_blob_chunk_offset(blobs: &[&WeightBlob], index: usize) -> u64 {
    let mut offset = 64u64;
    for blob in &blobs[..index] {
        offset += 64 + blob.fp16_byte_count() as u64;
    }
    offset
}

pub(crate) fn mil_blob_total_size(blobs: &[&WeightBlob]) -> usize {
    64 + blobs.iter().map(|blob| 64 + blob.fp16_byte_count()).sum::<usize>()
}

pub(crate) fn f32_to_f16(value: f32) -> u16 {
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
