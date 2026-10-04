use crate::ir::WeightBlob;

const ALIGNMENT: usize = 64;
const SENTINEL: u32 = 0xDEAD_BEEF;
const VERSION: u32 = 2;

pub struct WeightFile {
    bytes: Vec<u8>,
    count: u32,
    records: Vec<(WeightBlob, u64)>,
}

impl Default for WeightFile {
    fn default() -> Self {
        let mut bytes = vec![0; ALIGNMENT];
        bytes[4..8].copy_from_slice(&VERSION.to_le_bytes());
        Self {
            bytes,
            count: 0,
            records: Vec::new(),
        }
    }
}

impl WeightFile {
    pub fn add(&mut self, blob: &WeightBlob) -> u64 {
        if let Some((_, record)) = self.records.iter().find(|(stored, _)| stored == blob) {
            return *record;
        }
        let record = self.bytes.len();
        let data = record + ALIGNMENT;
        let mut header = [0; ALIGNMENT];
        header[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
        header[4..8].copy_from_slice(&(blob.data_type() as u32).to_le_bytes());
        header[8..16].copy_from_slice(&(blob.bytes().len() as u64).to_le_bytes());
        header[16..24].copy_from_slice(&(data as u64).to_le_bytes());
        self.bytes.extend_from_slice(&header);
        self.bytes.extend_from_slice(blob.bytes());
        self.bytes
            .resize(self.bytes.len().next_multiple_of(ALIGNMENT), 0);
        self.count += 1;
        self.bytes[0..4].copy_from_slice(&self.count.to_le_bytes());
        self.records.push((blob.clone(), record as u64));
        record as u64
    }

    pub fn into_bytes(self) -> Vec<u8> {
        if self.count == 0 {
            Vec::new()
        } else {
            self.bytes
        }
    }
}
