#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum WeightDataType {
    Float16 = 1,
    Float32 = 2,
    UInt8 = 3,
    Int8 = 4,
    Int16 = 6,
    UInt16 = 7,
    Int4 = 8,
    UInt1 = 9,
    UInt2 = 10,
    UInt4 = 11,
    UInt3 = 12,
    UInt6 = 13,
    Int32 = 14,
    UInt32 = 15,
}

impl WeightDataType {
    pub fn bit_width(self) -> usize {
        match self {
            Self::UInt1 => 1,
            Self::UInt2 => 2,
            Self::UInt3 => 3,
            Self::Int4 | Self::UInt4 => 4,
            Self::UInt6 => 6,
            Self::Int8 | Self::UInt8 => 8,
            Self::Float16 | Self::Int16 | Self::UInt16 => 16,
            Self::Float32 | Self::Int32 | Self::UInt32 => 32,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "fp16",
            Self::Float32 => "fp32",
            Self::UInt8 => "uint8",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::UInt16 => "uint16",
            Self::Int4 => "int4",
            Self::UInt1 => "uint1",
            Self::UInt2 => "uint2",
            Self::UInt4 => "uint4",
            Self::UInt3 => "uint3",
            Self::UInt6 => "uint6",
            Self::Int32 => "int32",
            Self::UInt32 => "uint32",
        }
    }
}
