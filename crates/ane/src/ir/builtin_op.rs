use crate::DataType;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::ir::{ArgumentType, ConstantInput, IrError, Operator, Parameter, Value, WeightBlob};

#[derive(Clone, PartialEq)]
pub struct BuiltinOp {
    pub operation: Operator,
    pub logical: bool,
    pub inputs: Box<[(Parameter, Tensor)]>,
    pub attributes: Box<[(Parameter, Value)]>,
    pub outputs: Box<[Tensor]>,
    pub blobs: Box<[ConstantInput]>,
}

impl BuiltinOp {
    pub fn constant_data(&self) -> Result<WeightBlob, IrError> {
        let blob = |key| self.blobs.iter().find(|b| b.name == key).unwrap();
        use Operator as O;
        use Parameter as P;
        let (data, mask) = match self.operation {
            O::ConstexprBlockwiseShiftScale => (P::Data, None),
            O::ConstexprLutToDense => (P::Indices, None),
            O::ConstexprSparseToDense => (P::NonzeroData, Some(P::Mask)),
            O::SparseBlockwiseWeights => (P::NonzeroData, Some(P::DataMask)),
            O::SparsePaletteWeights => (P::IndicesNonzeroData, Some(P::IndicesMask)),
            _ => unreachable!("constant operation required"),
        };
        let shape = self.outputs[0].physical_shape();
        let data = blob(data).data.decoded();
        let mask = mask.map(|key| blob(key).data.bytes());
        let palette = self
            .blobs
            .iter()
            .find(|b| b.name == P::Lut)
            .map(|b| b.data.decoded());
        let scale = self.blobs.iter().find(|b| b.name == P::Scale);
        let scales = scale.map(|b| b.data.decoded());
        let offset = self
            .blobs
            .iter()
            .find(|b| b.name == P::Offset)
            .map(|b| b.data.decoded());
        let mut source = 0;
        let values = (0..shape.iter().product())
            .map(|i| {
                if mask.is_some_and(|mask| mask[i / 8] & (1 << (i % 8)) == 0) {
                    return 0.;
                }
                let mut value = data[source];
                source += 1;
                if let Some(palette) = &palette {
                    let layout = &blob(P::Lut).shape;
                    let mut coordinate = i;
                    let mut group = 0;
                    let mut stride = 1;
                    for axis in (0..4).rev() {
                        group +=
                            ((coordinate % shape[axis]) / (shape[axis] / layout[axis])) * stride;
                        stride *= layout[axis];
                        coordinate /= shape[axis];
                    }
                    value = palette[group * layout[4] + value as usize];
                }
                if let Some(scale) = scale {
                    let mut coordinate = i;
                    let mut index = 0;
                    let mut stride = 1;
                    for axis in (0..4).rev() {
                        index += ((coordinate % shape[axis]) / (shape[axis] / scale.shape[axis]))
                            * stride;
                        stride *= scale.shape[axis];
                        coordinate /= shape[axis];
                    }
                    value = (value - offset.as_ref().map_or(0., |v| v[index]))
                        * scales.as_ref().unwrap()[index];
                }
                value
            })
            .collect::<Vec<_>>();
        WeightBlob::from_f32(&values)
    }
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
        let mut seen = std::collections::HashSet::new();
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
                use Operator as O;
                use Parameter as P;
                let valid = match key {
                    P::Mode if self.operation == O::Gelu => {
                        matches!(*value, "EXACT" | "TANH_APPROXIMATION")
                    }
                    P::Mode => matches!(*value, "constant" | "reflect" | "symmetric" | "replicate"),
                    P::PadType => matches!(*value, "valid" | "custom" | "same_lower"),
                    P::SamplingMode if matches!(self.operation, O::Resample | O::Affine) => {
                        matches!(*value, "nearest" | "bilinear")
                    }
                    P::SamplingMode => matches!(*value, "UNALIGN_CORNERS"),
                    P::PaddingMode => {
                        matches!(*value, "constant" | "reflection" | "symmetric" | "border")
                    }
                    P::CoordinatesMode => matches!(
                        *value,
                        "unnormalized" | "normalized_minus_one_to_one" | "normalized_zero_to_one"
                    ),
                    P::OutputIndicesDtype => *value == "uint16",
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
        let expected = if self.operation == Operator::Topk {
            2
        } else {
            1
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
        use Operator as O;
        use Parameter as P;
        let expected_dtype = match self.operation {
            O::Equal
            | O::NotEqual
            | O::Less
            | O::LessEqual
            | O::Greater
            | O::GreaterEqual
            | O::LogicalNot
            | O::LogicalAnd
            | O::LogicalOr => DataType::Bool,
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
            | O::GatherNd => input(P::X).unwrap().data_type(),
            O::Concat => input(P::Values).unwrap().data_type(),
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
            | O::LogicalAnd
            | O::LogicalOr => require(
                output.data_type() == DataType::Bool
                    && input(P::X).unwrap().data_type() == input(P::Y).unwrap().data_type(),
            ),
            O::LogicalNot => require(
                output.data_type() == DataType::Bool
                    && output.shape() == input(P::X).unwrap().shape(),
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
