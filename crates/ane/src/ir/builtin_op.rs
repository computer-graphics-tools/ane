use crate::DataType;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use std::collections::HashSet;

use crate::ir::{
    ArgumentType, ConstantInput, IrError, Operator, Operator as O, Parameter, Parameter as P, Value,
};

#[derive(Clone, PartialEq)]
pub struct BuiltinOp {
    pub operation: Operator,
    pub inputs: Box<[(Parameter, Tensor)]>,
    pub attributes: Box<[(Parameter, Value)]>,
    pub outputs: Box<[Tensor]>,
    pub blobs: Box<[ConstantInput]>,
}

impl BuiltinOp {
    pub fn matmul_shape(
        x: Tensor,
        y: Tensor,
        transpose_x: bool,
        transpose_y: bool,
    ) -> Result<[usize; 4], IrError> {
        let x = x.physical_shape();
        let y = y.physical_shape();
        if x[if transpose_x { 2 } else { 3 }] != y[if transpose_y { 3 } else { 2 }] {
            return Err(IrError::InvalidProgram(
                "matmul contraction dimensions differ",
            ));
        }
        if !(0..2).all(|a| x[a] == y[a] || x[a] == 1 || y[a] == 1) {
            return Err(IrError::InvalidProgram("matmul batch dimensions differ"));
        }
        Ok([
            x[0].max(y[0]),
            x[1].max(y[1]),
            x[if transpose_x { 3 } else { 2 }],
            y[if transpose_y { 2 } else { 3 }],
        ])
    }

    pub fn validate(&self) -> Result<(), IrError> {
        let schema = self.operation.schema();
        let error = |parameter: Parameter, reason| IrError::Argument {
            operation: self.operation.as_str(),
            parameter: parameter.as_str(),
            reason,
        };
        let mut seen = HashSet::new();
        let mut check = |key, accepts: &dyn Fn(ArgumentType) -> bool| {
            let (_, kind, _) = schema
                .iter()
                .find(|(name, _, _)| *name == key)
                .ok_or_else(|| error(key, "not defined by the operation schema"))?;
            if !seen.insert(key)
                && !(self.operation == Operator::Concat && key == Parameter::Values)
            {
                return Err(error(key, "duplicate binding"));
            }
            if !accepts(*kind) {
                return Err(error(key, "incorrect argument type"));
            }
            Ok(())
        };
        for &(key, tensor) in &self.inputs {
            check(key, &|kind| kind.accepts_tensor(tensor.data_type()))?;
        }
        for (key, value) in &self.attributes {
            value.validate()?;
            check(*key, &|kind| kind.accepts_value(value))?;
            if let Value::String(value) = value {
                let valid = match key {
                    P::Mode if self.operation == O::Gelu => {
                        matches!(
                            *value,
                            "EXACT" | "TANH_APPROXIMATION" | "SIGMOID_APPROXIMATION"
                        )
                    }
                    P::Mode => matches!(*value, "constant" | "reflect" | "symmetric" | "replicate"),
                    P::PadType => matches!(*value, "valid" | "custom" | "same" | "same_lower"),
                    P::SamplingMode if self.operation == O::Resample => {
                        matches!(*value, "nearest" | "bilinear")
                    }
                    P::SamplingMode => matches!(
                        *value,
                        "STRICT_ALIGN_CORNERS"
                            | "ALIGN_CORNERS"
                            | "DEFAULT"
                            | "OFFSET_CORNERS"
                            | "UNALIGN_CORNERS"
                    ),
                    P::PaddingMode => {
                        matches!(*value, "constant" | "reflection" | "symmetric" | "border")
                    }
                    P::CoordinatesMode => matches!(
                        *value,
                        "unnormalized" | "normalized_minus_one_to_one" | "normalized_zero_to_one"
                    ),
                    P::OutputIndicesDtype => *value == "uint16",
                    P::OutputDtype
                        if matches!(self.operation, O::ReduceArgmax | O::ReduceArgmin) =>
                    {
                        *value == "uint16"
                    }
                    P::OutputDtype => matches!(*value, "int8" | "uint8"),
                    P::Dtype => matches!(
                        *value,
                        "fp16" | "bool" | "int8" | "uint8" | "int16" | "uint16"
                    ),
                    _ => false,
                };
                if !valid {
                    return Err(error(*key, "unsupported value"));
                }
            }
        }
        for blob in &self.blobs {
            check(blob.name, &|kind| kind.accepts_blob(blob.data.data_type()))?;
            let elements = blob
                .shape
                .iter()
                .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) });
            if elements != Some(blob.data.element_count()) {
                return Err(error(blob.name, "blob shape differs from its data"));
            }
        }
        for &(key, _, required) in schema {
            if required && !seen.contains(&key) {
                return Err(error(key, "required argument is missing"));
            }
        }
        let sizes = self
            .attributes
            .iter()
            .find(|(key, _)| *key == P::SplitSizes)
            .map(|(_, value)| value);
        let expected = match (self.operation, sizes) {
            (Operator::Topk, _) => 2,
            (Operator::Split, Some(Value::Int32List(sizes))) => sizes.len(),
            _ => 1,
        };
        if self.outputs.len() != expected {
            return Err(IrError::InvalidProgram(
                "operation result count differs from schema",
            ));
        }
        self.validate_results()
    }

    fn validate_results(&self) -> Result<(), IrError> {
        let input = |key| {
            self.inputs
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, t)| *t)
        };
        let attr = |key| {
            self.attributes
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, v)| v)
        };
        let output = self.outputs[0];
        let require = |valid| {
            if valid {
                Ok(())
            } else {
                Err(IrError::InvalidProgram(
                    "operation result shape or type differs from its operands",
                ))
            }
        };
        let expected_dtype = match self.operation {
            O::Equal
            | O::NotEqual
            | O::Less
            | O::LessEqual
            | O::Greater
            | O::GreaterEqual
            | O::LogicalAnd => DataType::Bool,
            O::ReduceArgmax | O::ReduceArgmin => DataType::UInt16,
            O::Select => input(P::A).unwrap().data_type(),
            O::Cast | O::Quantize => output.data_type(),
            O::Reshape
            | O::Transpose
            | O::SliceBySize
            | O::SliceByIndex
            | O::DynamicSlice
            | O::Tile
            | O::Reverse
            | O::Gather
            | O::GatherAlongAxis
            | O::SliceUpdate
            | O::Split => input(P::X).unwrap().data_type(),
            O::Concat | O::Stack => input(P::Values).unwrap().data_type(),
            _ => DataType::Float16,
        };
        require(output.data_type() == expected_dtype)?;
        match self.operation {
            O::DynamicSlice => {
                let axis = match attr(P::Axis) {
                    Some(Value::Int32(v)) => *v,
                    _ => unreachable!(),
                };
                let length = match attr(P::Size) {
                    Some(Value::Int32(v)) => *v,
                    _ => unreachable!(),
                };
                let mut shape = input(P::X).unwrap().physical_shape();
                require(axis < 4 && length > 0 && length <= shape[axis])?;
                shape[axis] = length;
                require(
                    input(P::Begin)
                        .unwrap()
                        .physical_shape()
                        .iter()
                        .product::<usize>()
                        == 1
                        && output.physical_shape() == shape,
                )
            }
            O::Matmul => {
                let x = input(P::X).unwrap();
                let y = input(P::Y).unwrap();
                let tx = attr(P::TransposeX) == Some(&Value::Bool(true));
                let ty = attr(P::TransposeY) == Some(&Value::Bool(true));
                require(output.physical_shape() == Self::matmul_shape(x, y, tx, ty)?)
            }
            O::Reshape => {
                let x = input(P::X).unwrap();
                let Some(Value::Int32List(shape)) = attr(P::Shape) else {
                    unreachable!()
                };
                require(shape.as_ref() == output.physical_shape())?;
                require(
                    x.physical_shape().iter().product::<usize>()
                        == output.physical_shape().iter().product::<usize>()
                        && x.data_type() == output.data_type(),
                )
            }
            O::Equal
            | O::NotEqual
            | O::Less
            | O::LessEqual
            | O::Greater
            | O::GreaterEqual
            | O::LogicalAnd => require(
                output.data_type() == DataType::Bool
                    && input(P::X).unwrap().data_type() == input(P::Y).unwrap().data_type(),
            ),
            O::Select => require(
                output.data_type() == input(P::A).unwrap().data_type()
                    && output.data_type() == input(P::B).unwrap().data_type(),
            ),
            O::Cast | O::Quantize => {
                let key = if self.operation == O::Cast {
                    P::Dtype
                } else {
                    P::OutputDtype
                };
                require(attr(key) == Some(&Value::String(output.data_type().as_str())))?;
                require(
                    output.shape()
                        == input(if self.operation == O::Cast {
                            P::X
                        } else {
                            P::Input
                        })
                        .unwrap()
                        .shape(),
                )
            }
            O::Dequantize => require(
                output.data_type() == DataType::Float16
                    && matches!(
                        input(P::Input).unwrap().data_type(),
                        DataType::Int8 | DataType::UInt8
                    ),
            ),
            O::Topk => require(
                output.data_type() == DataType::Float16
                    && self.outputs[1].data_type() == DataType::UInt16
                    && output.shape() == self.outputs[1].shape(),
            ),
            _ => Ok(()),
        }
    }
}
