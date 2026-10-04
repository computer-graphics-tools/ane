use crate::DataType;
use crate::ir::IrError;

#[derive(Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Int32(usize),
    Fp16(f32),
    String(&'static str),
    Integer(DataType, i32),
    Int32List(Box<[usize]>),
    Int32Matrix([[usize; 2]; 2]),
    BoolList(Box<[bool]>),
    Fp16List(Box<[f32]>),
    IntegerList(DataType, Box<[i32]>),
}

impl Value {
    pub fn int32_list(values: &[usize]) -> Self {
        Self::Int32List(values.into())
    }
    pub fn validate(&self) -> Result<(), IrError> {
        let int = |v: &usize| i32::try_from(*v).is_ok();
        let fp16 = |v: &f32| v.is_finite() && half::f16::from_f32(*v).is_finite();
        let integer = |dtype, v| match dtype {
            DataType::Int8 => i8::try_from(v).is_ok(),
            DataType::UInt8 => u8::try_from(v).is_ok(),
            DataType::Int16 => i16::try_from(v).is_ok(),
            DataType::UInt16 => u16::try_from(v).is_ok(),
            DataType::Int32 => true,
            _ => false,
        };
        let (valid, kind) = match self {
            Self::Bool(_) | Self::BoolList(_) => (true, "bool"),
            Self::Int32(v) => (int(v), "int32"),
            Self::Int32List(v) => (v.iter().all(int), "int32"),
            Self::Int32Matrix(v) => (v.iter().flatten().all(int), "int32"),
            Self::Fp16(v) => (fp16(v), "fp16"),
            Self::Fp16List(v) => (v.iter().all(fp16), "fp16"),
            Self::Integer(dtype, v) => (integer(*dtype, *v), dtype.as_str()),
            Self::IntegerList(dtype, v) => (v.iter().all(|v| integer(*dtype, *v)), dtype.as_str()),
            Self::String(v) => (
                !v.chars().any(|c| c.is_control() || matches!(c, '"' | '\\')),
                "string",
            ),
        };
        if valid {
            Ok(())
        } else {
            Err(IrError::InvalidValue(kind))
        }
    }
}
