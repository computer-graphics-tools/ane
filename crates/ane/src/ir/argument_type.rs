use crate::ir::Value;
use crate::{DataType, WeightDataType};

#[derive(Clone, Copy)]
pub enum ArgumentType {
    Tensor,
    FloatTensor,
    BoolTensor,
    IndexTensor,
    Int32Tensor,
    Scalar,
    Bool,
    Int32,
    Int32List,
    Int32Matrix,
    BoolList,
    Fp16,
    Fp32,
    String,
    Scale,
    ZeroPoint,
    Blob,
    FloatBlob,
    MaskBlob,
}

impl ArgumentType {
    pub fn accepts_tensor(self, dtype: DataType) -> bool {
        match self {
            Self::Tensor => true,
            Self::Scalar | Self::FloatTensor => dtype == DataType::Float16,
            Self::BoolTensor => dtype == DataType::Bool,
            Self::Int32Tensor => dtype == DataType::Int32,
            Self::IndexTensor => dtype == DataType::UInt16,
            _ => false,
        }
    }
    pub fn accepts_value(self, value: &Value) -> bool {
        matches!(
            (self, value),
            (Self::Scalar | Self::Fp16, Value::Fp16(_))
                | (Self::Bool, Value::Bool(_))
                | (Self::Int32, Value::Int32(_))
                | (Self::Int32List, Value::Int32List(_))
                | (Self::Int32Matrix, Value::Int32Matrix(_))
                | (Self::BoolList, Value::BoolList(_))
                | (Self::Fp32, Value::Fp32(_))
                | (Self::String, Value::String(_))
                | (Self::Scale, Value::Fp16(_) | Value::Fp16List(_))
                | (
                    Self::ZeroPoint,
                    Value::Integer(_, _) | Value::IntegerList(_, _)
                )
        )
    }
    pub fn accepts_blob(self, dtype: WeightDataType) -> bool {
        match self {
            Self::Blob => true,
            Self::FloatBlob => dtype == WeightDataType::Float16,
            Self::MaskBlob => dtype == WeightDataType::UInt1,
            _ => false,
        }
    }
}
