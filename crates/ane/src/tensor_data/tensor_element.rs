use crate::DataType;

pub trait TensorElement: bytemuck::NoUninit + bytemuck::CheckedBitPattern {
    const DATA_TYPE: DataType;
}

macro_rules! element {
    ($type:ty, $dtype:ident) => {
        impl TensorElement for $type {
            const DATA_TYPE: DataType = DataType::$dtype;
        }
    };
}
element!(f32, Float32);
element!(half::f16, Float16);
element!(u8, UInt8);
element!(i8, Int8);
element!(u16, UInt16);
element!(i16, Int16);
element!(i32, Int32);
element!(bool, Bool);
