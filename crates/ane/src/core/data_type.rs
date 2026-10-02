#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    Float32,
    Float16,
    Int8,
    UInt8,
    Int32,
    Int16,
    UInt16,
    Bool,
}

impl DataType {
    pub const fn storage_name(self) -> &'static str {
        match self {
            Self::Float16 => "Float16",
            Self::Float32 => "Float32",
            Self::Int8 => "Int8",
            Self::UInt8 | Self::Bool => "UInt8",
            Self::Int16 => "Int16",
            Self::UInt16 => "UInt16",
            Self::Int32 => "Int32",
        }
    }
    pub const fn byte_width(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::Int16 | Self::UInt16 => 2,
            Self::Int8 | Self::UInt8 | Self::Bool => 1,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float32 => "fp32",
            Self::Float16 => "fp16",
            Self::Int8 => "int8",
            Self::UInt8 => "uint8",
            Self::Int32 => "int32",
            Self::Int16 => "int16",
            Self::UInt16 => "uint16",
            Self::Bool => "bool",
        }
    }
}
