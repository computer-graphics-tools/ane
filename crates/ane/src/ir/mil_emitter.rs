use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::graph::TensorHandle;
use crate::ir::{
    BuiltinOp, IrError, MilProgram, Op, Operator as O, Parameter as P, Program, StateUpdateOp,
    Value, WeightBlob, WeightFile,
};
use crate::{DataType, Tensor};

const HEADER: &str = concat!(
    "program(1.3)\n",
    "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3600.16.1\"}, ",
    "{\"coremlc-version\", \"3600.25.2\"}})]\n{\n"
);
const NO_MASK: &str = "tensor<bool, [4]>([false, false, false, false])";

pub struct MilEmitter {
    arguments: Vec<String>,
    body: String,
    names: HashMap<Tensor, String>,
    states: HashSet<Tensor>,
    constants: HashMap<Tensor, WeightBlob>,
    produced: HashSet<String>,
    weights: WeightFile,
    next: usize,
}

impl MilEmitter {
    pub fn emit(functions: &[Program]) -> Result<MilProgram, IrError> {
        let mut text = HEADER.to_string();
        let mut weights = WeightFile::default();
        let mut outputs = Vec::new();
        for (index, function) in functions.iter().enumerate() {
            if functions[..index]
                .iter()
                .any(|f| f.name() == function.name())
            {
                return Err(IrError::InvalidProgram("MIL function names must be unique"));
            }
            let (body, names, rest) = Self::function(function, weights)?;
            weights = rest;
            text.push_str(&body);
            outputs.push(names);
        }
        text.push_str("}\n");
        Ok(MilProgram {
            text,
            weights: weights.into_bytes(),
            outputs,
        })
    }

    fn function(
        program: &Program,
        weights: WeightFile,
    ) -> Result<(String, Vec<String>, WeightFile), IrError> {
        let states = program
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::StateWrite(op) => Some(op.state),
                Op::StateUpdate(op) => Some(op.state),
                _ => None,
            })
            .collect();
        let mut emitter = Self {
            arguments: Vec::new(),
            body: String::new(),
            names: HashMap::new(),
            states,
            constants: HashMap::new(),
            produced: HashSet::new(),
            weights,
            next: 0,
        };
        for (&tensor, (name, shape, storage)) in program.feed_tensors().iter().zip(program.inputs())
        {
            emitter.input(tensor, name, shape, *storage);
        }
        for op in program.ops() {
            emitter.op(op)?;
        }
        let mut outputs = Vec::new();
        for (&tensor, (_, _, storage)) in program.target_tensors().iter().zip(program.outputs()) {
            let output = emitter.output(tensor, *storage, &outputs)?;
            outputs.push(output);
        }
        let text = format!(
            "    func {}<ios18>({}) {{\n{}    }} -> ({});\n",
            program.name(),
            emitter.arguments.join(", "),
            emitter.body,
            outputs.join(", ")
        );
        Ok((text, outputs, emitter.weights))
    }

    fn input(&mut self, tensor: Tensor, name: &str, shape: &[usize; 4], storage: DataType) {
        let declared = if self.states.contains(&tensor) {
            format!("state<{}>", ty(tensor.data_type(), shape))
        } else if storage == DataType::Int32 {
            "tensor<int32, [1]>".into()
        } else {
            ty(storage, shape)
        };
        self.arguments.push(format!("{declared} {name}"));
        let state = self.states.contains(&tensor);
        let value = if !state && storage != tensor.data_type() && storage != DataType::Int32 {
            self.convert(name, storage, tensor.data_type(), shape)
        } else {
            name.into()
        };
        self.names.insert(tensor, value);
    }

    fn output(
        &mut self,
        tensor: Tensor,
        storage: DataType,
        outputs: &[String],
    ) -> Result<String, IrError> {
        let shape = tensor.physical_shape();
        let value = self.value(tensor)?;
        if storage != tensor.data_type() {
            return Ok(self.convert(&value, tensor.data_type(), storage, &shape));
        }
        if is_float(storage) {
            return Ok(self.assign(ty(storage, &shape), format!("identity(x = {value})")));
        }
        if self.produced.contains(&value) && !outputs.contains(&value) {
            Ok(value)
        } else {
            Err(IrError::InvalidProgram(
                "an integer output must be the unique result of an operation; MIL has no integer identity",
            ))
        }
    }

    fn op(&mut self, op: &Op) -> Result<(), IrError> {
        match op {
            Op::Constant(op) => {
                self.constants.insert(op.top, op.data.clone());
            }
            Op::StateRead(op) => {
                let value = self.value(op.state)?;
                self.names.insert(op.top, value);
            }
            Op::StateWrite(op) => {
                let data = self.value(op.bottom)?;
                self.write_state(op.state, op.top, &data)?;
            }
            Op::StateUpdate(op) => self.state_update(op)?,
            Op::Builtin(op) => self.builtin(op)?,
        }
        Ok(())
    }

    fn builtin(&mut self, op: &BuiltinOp) -> Result<(), IrError> {
        let output = op.outputs[0];
        let value = match op.operation {
            _ if op.outputs.len() > 1 => self.multiple(op)?,
            O::ConstexprLutToSparse | O::ConstexprSparseBlockwiseShiftScale => {
                self.sparse_constant(op)?
            }
            O::ConstexprLutToDense if op.blobs.iter().any(|blob| blob.name == P::LutScale) => {
                self.quantized_palette(op)?
            }
            O::Dequantize if self.constants.contains_key(&input(op, P::Input)) => {
                self.constant_dequantize(op)?
            }
            O::DynamicSlice => self.dynamic_slice(op)?,
            O::Gather => self.gather(op)?,
            _ => self.generic(op)?,
        };
        self.names.insert(output, value);
        Ok(())
    }

    fn arguments(&mut self, op: &BuiltinOp) -> Result<String, IrError> {
        let mut arguments = Vec::new();
        let values = op
            .inputs
            .iter()
            .filter(|(key, _)| *key == P::Values)
            .map(|&(_, tensor)| self.value(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        if !values.is_empty() {
            arguments.push(format!("values = ({})", values.join(", ")));
        }
        for &(key, tensor) in op.inputs.iter().filter(|(key, _)| *key != P::Values) {
            arguments.push(format!("{key} = {}", self.value(tensor)?));
        }
        for (key, value) in &op.attributes {
            arguments.push(format!("{key} = {}", literal(value)));
        }
        for blob in &op.blobs {
            arguments.push(format!(
                "{} = {}",
                blob.name,
                self.blob(&blob.data, &blob.shape)
            ));
        }
        Ok(arguments.join(", "))
    }

    fn multiple(&mut self, op: &BuiltinOp) -> Result<String, IrError> {
        let arguments = self.arguments(op)?;
        let names: Vec<_> = op.outputs.iter().map(|_| self.fresh()).collect();
        let declarations: Vec<_> = op
            .outputs
            .iter()
            .zip(&names)
            .map(|(&tensor, name)| format!("{} {name}", ty_of(tensor)))
            .collect();
        writeln!(
            self.body,
            "        {} = {}({arguments})[name = string(\"{}\")];",
            declarations.join(", "),
            op.operation,
            names[0]
        )
        .unwrap();
        self.produced.extend(names.iter().cloned());
        for (&tensor, name) in op.outputs.iter().zip(&names).skip(1) {
            self.names.insert(tensor, name.clone());
        }
        Ok(names[0].clone())
    }

    fn generic(&mut self, op: &BuiltinOp) -> Result<String, IrError> {
        let arguments = self.arguments(op)?;
        Ok(self.assign(
            ty_of(op.outputs[0]),
            format!("{}({arguments})", op.operation),
        ))
    }

    fn write_state(&mut self, state: Tensor, top: Tensor, data: &str) -> Result<(), IrError> {
        let name = self.name(state)?;
        self.statement(format!("write_state(data = {data}, input = {name})"));
        let value = self.assign(ty_of(state), format!("read_state(input = {name})"));
        self.names.insert(top, value);
        Ok(())
    }

    fn state_update(&mut self, op: &StateUpdateOp) -> Result<(), IrError> {
        let shape = op.state.physical_shape();
        let previous = self.value(op.state)?;
        let update = self.value(op.bottom)?;
        let position = self.name(op.position)?;
        let rows = self.integer(op.rows);
        let end = self.assign(
            "tensor<int32, [1]>".into(),
            format!("add(x = {position}, y = {rows})"),
        );
        let starts = [
            self.integer(0),
            self.integer(op.channel),
            position,
            self.integer(0),
        ];
        let ends = [
            self.integer(shape[0]),
            self.integer(op.channel + op.channels),
            end,
            self.integer(shape[3]),
        ];
        let begin = self.concat_integers(&starts);
        let end = self.concat_integers(&ends);
        let next = self.assign(
            ty_of(op.state),
            format!(
                "slice_update(begin = {begin}, begin_mask = {NO_MASK}, end = {end}, end_mask = {NO_MASK}, \
                 squeeze_mask = {NO_MASK}, stride = {}, update = {update}, x = {previous})",
                integers(&[1, 1, 1, 1])
            ),
        );
        self.write_state(op.state, op.top, &next)
    }

    fn constant_dequantize(&mut self, op: &BuiltinOp) -> Result<String, IrError> {
        let source = input(op, P::Input);
        let data = self.constants[&source].clone();
        let scales: Vec<f32> = match attribute(op, P::Scale) {
            Some(Value::Fp16(scale)) => vec![*scale],
            Some(Value::Fp16List(scales)) => scales.to_vec(),
            _ => return Err(IrError::InvalidProgram("dequantization scale is missing")),
        };
        let mut shape = [1; 4];
        if let (Some(Value::Int32(axis)), true) = (attribute(op, P::Axis), scales.len() > 1) {
            shape[*axis] = scales.len();
        }
        let scale = WeightBlob::from_f32(&scales)?;
        let mut arguments = vec![
            format!("data = {}", self.blob(&data, &source.physical_shape())),
            format!("scale = {}", self.blob(&scale, &shape)),
        ];
        if let Some(zero_point) = attribute(op, P::ZeroPoint) {
            let offsets: Vec<u8> = match zero_point {
                Value::Integer(_, value) => vec![*value as u8; scales.len()],
                Value::IntegerList(_, values) => values.iter().map(|&value| value as u8).collect(),
                _ => {
                    return Err(IrError::InvalidProgram(
                        "dequantization zero point is invalid",
                    ));
                }
            };
            let count = offsets.len();
            let offset = WeightBlob::from_bytes(offsets, count, data.data_type())?;
            arguments.push(format!("offset = {}", self.blob(&offset, &shape)));
        }
        Ok(self.assign(
            ty_of(op.outputs[0]),
            format!("constexpr_blockwise_shift_scale({})", arguments.join(", ")),
        ))
    }

    fn quantized_palette(&mut self, op: &BuiltinOp) -> Result<String, IrError> {
        let blobs = [P::Indices, P::Lut, P::LutScale, P::LutOffset]
            .map(|key| op.blobs.iter().find(|blob| blob.name == key));
        let [Some(indices), Some(lut), Some(scale), offset] = blobs else {
            return Err(IrError::InvalidProgram("quantized palette is incomplete"));
        };
        let mut arguments = vec![
            format!("data = {}", self.blob(&lut.data, &lut.shape)),
            format!("scale = {}", self.blob(&scale.data, &scale.shape)),
        ];
        if let Some(offset) = offset {
            arguments.push(format!(
                "offset = {}",
                self.blob(&offset.data, &offset.shape)
            ));
        }
        let palette = self.assign(
            ty(DataType::Float16, &lut.shape),
            format!("constexpr_blockwise_shift_scale({})", arguments.join(", ")),
        );
        let mut arguments = vec![
            format!("indices = {}", self.blob(&indices.data, &indices.shape)),
            format!("lut = {palette}"),
        ];
        if let Some(axis) = attribute(op, P::VectorAxis) {
            arguments.push(format!("vector_axis = {}", literal(axis)));
        }
        Ok(self.assign(
            ty_of(op.outputs[0]),
            format!("constexpr_lut_to_dense({})", arguments.join(", ")),
        ))
    }

    fn sparse_constant(&mut self, op: &BuiltinOp) -> Result<String, IrError> {
        let values = if op.operation == O::ConstexprLutToSparse {
            P::IndicesNonzeroData
        } else {
            P::NonzeroData
        };
        let count = op
            .blobs
            .iter()
            .find(|blob| blob.name == values)
            .map(|blob| blob.data.element_count())
            .ok_or(IrError::InvalidProgram(
                "sparse weights have no nonzero data",
            ))?;
        let arguments = self.arguments(op)?;
        let output = op.outputs[0];
        let [mask, nonzero] = [0, 1].map(|_| self.fresh());
        writeln!(
            self.body,
            "        tensor<uint1, {}> {mask}, tensor<fp16, [{count}]> {nonzero} = {}({arguments})[name = string(\"{mask}\")];",
            dimensions(&output.physical_shape()),
            op.operation
        )
        .unwrap();
        Ok(self.assign(
            ty_of(output),
            format!("constexpr_sparse_to_dense(mask = {mask}, nonzero_data = {nonzero})"),
        ))
    }

    fn dynamic_slice(&mut self, op: &BuiltinOp) -> Result<String, IrError> {
        let x = input(op, P::X);
        let axis = integer_attribute(op, P::Axis);
        let begin = self.name(input(op, P::Begin))?;
        let starts: Vec<_> = (0..4)
            .map(|i| {
                if i == axis {
                    begin.clone()
                } else {
                    self.integer(0)
                }
            })
            .collect();
        let starts = self.concat_integers(&starts);
        let mut sizes = x.physical_shape();
        sizes[axis] = integer_attribute(op, P::Size);
        let source = self.value(x)?;
        Ok(self.assign(
            ty_of(op.outputs[0]),
            format!(
                "slice_by_size(begin = {starts}, size = {}, x = {source})",
                integers(&sizes)
            ),
        ))
    }

    fn gather(&mut self, op: &BuiltinOp) -> Result<String, IrError> {
        let x = input(op, P::X);
        let indices = input(op, P::Indices);
        let axis = integer_attribute(op, P::Axis);
        let count = indices.shape().iter().product();
        let source = self.value(x)?;
        let index = self.value(indices)?;
        let mut shape = x.physical_shape();
        shape[axis] = count;
        let index = self.reshape(&index, DataType::UInt16, &[count]);
        let gathered = self.assign(
            ty(x.data_type(), &shape),
            format!(
                "gather(axis = int32({axis}), batch_dims = int32(0), indices = {index}, \
                     validate_indices = bool(false), x = {source})"
            ),
        );
        Ok(self.reshape(&gathered, x.data_type(), &op.outputs[0].physical_shape()))
    }

    fn convert(&mut self, value: &str, from: DataType, to: DataType, shape: &[usize]) -> String {
        if from == to {
            return value.into();
        }
        self.assign(
            ty(to, shape),
            format!("cast(dtype = string(\"{}\"), x = {value})", to.as_str()),
        )
    }

    fn reshape(&mut self, value: &str, dtype: DataType, shape: &[usize]) -> String {
        self.assign(
            ty(dtype, shape),
            format!("reshape(shape = {}, x = {value})", integers(shape)),
        )
    }

    fn concat_integers(&mut self, values: &[String]) -> String {
        self.assign(
            format!("tensor<int32, [{}]>", values.len()),
            format!(
                "concat(axis = int32(0), interleave = bool(false), values = ({}))",
                values.join(", ")
            ),
        )
    }

    fn integer(&mut self, value: usize) -> String {
        self.define("tensor<int32, [1]>".into(), integers(&[value]))
    }

    fn constant(&mut self, data: &WeightBlob, tensor: Tensor) -> String {
        let shape = tensor.physical_shape();
        let value = self.blob(data, &shape);
        self.define(ty_of(tensor), value)
    }

    fn blob(&mut self, data: &WeightBlob, shape: &[usize]) -> String {
        let offset = self.weights.add(data);
        format!(
            "tensor<{}, {}>(BLOBFILE(path = string(\"{}\"), offset = uint64({offset})))",
            data.data_type().as_str(),
            dimensions(shape),
            MilProgram::WEIGHT_PATH
        )
    }

    fn value(&mut self, tensor: Tensor) -> Result<String, IrError> {
        if !self.names.contains_key(&tensor)
            && let Some(data) = self.constants.get(&tensor).cloned()
        {
            let value = self.constant(&data, tensor);
            self.names.insert(tensor, value);
        }
        let name = self.name(tensor)?;
        Ok(if self.states.contains(&tensor) {
            self.assign(ty_of(tensor), format!("read_state(input = {name})"))
        } else {
            name
        })
    }

    fn name(&self, tensor: Tensor) -> Result<String, IrError> {
        self.names
            .get(&tensor)
            .cloned()
            .ok_or(IrError::InvalidProgram(
                "operand used before its definition",
            ))
    }

    fn define(&mut self, ty: String, value: String) -> String {
        let name = self.fresh();
        writeln!(
            self.body,
            "        {ty} {name} = const()[name = string(\"{name}\"), val = {value}];"
        )
        .unwrap();
        name
    }

    fn assign(&mut self, ty: String, expression: String) -> String {
        let name = self.fresh();
        self.produced.insert(name.clone());
        writeln!(
            self.body,
            "        {ty} {name} = {expression}[name = string(\"{name}\")];"
        )
        .unwrap();
        name
    }

    fn statement(&mut self, expression: String) {
        let name = self.fresh();
        writeln!(
            self.body,
            "        {expression}[name = string(\"{name}\")];"
        )
        .unwrap();
    }

    fn fresh(&mut self) -> String {
        self.next += 1;
        format!("v{}", self.next)
    }
}

fn input(op: &BuiltinOp, key: P) -> Tensor {
    op.inputs
        .iter()
        .find(|(name, _)| *name == key)
        .map(|&(_, tensor)| tensor)
        .expect("validated operation input")
}

fn attribute(op: &BuiltinOp, key: P) -> Option<&Value> {
    op.attributes
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, value)| value)
}

fn integer_attribute(op: &BuiltinOp, key: P) -> usize {
    match attribute(op, key) {
        Some(Value::Int32(value)) => *value,
        _ => unreachable!("validated integer attribute"),
    }
}

fn is_float(dtype: DataType) -> bool {
    matches!(dtype, DataType::Float16 | DataType::Float32)
}

fn ty_of(tensor: Tensor) -> String {
    ty(tensor.data_type(), &tensor.physical_shape())
}

fn ty(dtype: DataType, shape: &[usize]) -> String {
    format!("tensor<{}, {}>", dtype.as_str(), dimensions(shape))
}

fn dimensions(shape: &[usize]) -> String {
    let dimensions: Vec<_> = shape.iter().map(usize::to_string).collect();
    format!("[{}]", dimensions.join(", "))
}

fn integers(values: &[usize]) -> String {
    list("int32", values.iter().map(usize::to_string))
}

fn list(dtype: &str, items: impl Iterator<Item = String>) -> String {
    let items: Vec<_> = items.collect();
    format!("tensor<{dtype}, [{}]>([{}])", items.len(), items.join(", "))
}

fn literal(value: &Value) -> String {
    match value {
        Value::Bool(value) => format!("bool({value})"),
        Value::Int32(value) => format!("int32({value})"),
        Value::Fp16(value) => format!("fp16({})", hex_float(*value)),
        Value::String(value) => format!("string(\"{value}\")"),
        Value::Integer(dtype, value) => format!("{}({value})", dtype.as_str()),
        Value::Int32List(values) => integers(values),
        Value::Int32Matrix([[a, b], [c, d]]) => {
            format!("tensor<int32, [2, 2]>([[{a}, {b}], [{c}, {d}]])")
        }
        Value::BoolList(values) => list("bool", values.iter().map(bool::to_string)),
        Value::Fp16List(values) => list("fp16", values.iter().map(|&v| hex_float(v))),
        Value::IntegerList(dtype, values) => {
            list(dtype.as_str(), values.iter().map(i32::to_string))
        }
    }
}

fn hex_float(value: f32) -> String {
    let value = half::f16::from_f32(value).to_f64();
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if value == 0. {
        return format!("{sign}0x0p+0");
    }
    let bits = value.abs().to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let mantissa = format!("{:013x}", bits & ((1 << 52) - 1));
    let mantissa = mantissa.trim_end_matches('0');
    let fraction = if mantissa.is_empty() {
        String::new()
    } else {
        format!(".{mantissa}")
    };
    format!("{sign}0x1{fraction}p{exponent:+}")
}
