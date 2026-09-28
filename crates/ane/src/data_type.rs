#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float16,
    Int8,
    UInt8,
    Int32,
}

impl DataType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 => 2,
            Self::Int8 | Self::UInt8 => 1,
        }
    }

    pub const fn mil_name(self) -> &'static str {
        match self {
            Self::Float32 => "fp32",
            Self::Float16 => "fp16",
            Self::Int8 => "int8",
            Self::UInt8 => "uint8",
            Self::Int32 => "int32",
        }
    }
}
