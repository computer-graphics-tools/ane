use crate::graph::TensorHandle;
use crate::ir::{
    BuiltinOp, CompilerException, IrError, Op, Operator as O, Parameter as P, Program, Value,
    WeightBlob,
};
use crate::{DataType, Tensor, WeightDataType};
use objc2::{
    AnyThread,
    rc::Retained,
    runtime::{AnyClass, AnyObject, Bool},
};
use objc2_foundation::{NSArray, NSData, NSNumber, NSString};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::*;
use std::{
    collections::{HashMap, HashSet},
    panic::AssertUnwindSafe,
};

pub struct GraphCompiler {
    graph: Retained<MPSGraph>,
    values: HashMap<Tensor, Retained<MPSGraphTensor>>,
    variables: HashMap<Tensor, Retained<MPSGraphTensor>>,
    palettized: HashMap<Tensor, BuiltinOp>,
    feeds: Vec<(Retained<MPSGraphTensor>, Retained<MPSGraphShapedType>)>,
    targets: Vec<Retained<MPSGraphTensor>>,
    effects: Vec<Retained<MPSGraphOperation>>,
    pub state_outputs: Vec<usize>,
    pub discarded_outputs: Vec<Tensor>,
}

fn numbers(values: &[usize]) -> Retained<NSArray<NSNumber>> {
    NSArray::from_retained_slice(
        &values
            .iter()
            .copied()
            .map(NSNumber::new_usize)
            .collect::<Vec<_>>(),
    )
}
fn data_type(dtype: DataType) -> MPSDataType {
    match dtype {
        DataType::Float16 => MPSDataType::Float16,
        DataType::Float32 => MPSDataType::Float32,
        DataType::Int8 => MPSDataType::Int8,
        DataType::UInt8 => MPSDataType::UInt8,
        DataType::Int16 => MPSDataType::Int16,
        DataType::UInt16 => MPSDataType::UInt16,
        DataType::Int32 => MPSDataType::Int32,
        DataType::Bool => MPSDataType::Bool,
    }
}
fn storage_type(dtype: DataType) -> MPSDataType {
    if dtype == DataType::Bool {
        MPSDataType::UInt8
    } else {
        data_type(dtype)
    }
}
fn padding_mode(mode: &str) -> MPSGraphPaddingMode {
    match mode {
        "reflect" | "reflection" => MPSGraphPaddingMode::Reflect,
        "symmetric" => MPSGraphPaddingMode::Symmetric,
        "replicate" | "border" => MPSGraphPaddingMode::ClampToEdge,
        _ => MPSGraphPaddingMode::Constant,
    }
}
pub fn catch_native<T>(f: impl FnOnce() -> T) -> Result<T, IrError> {
    objc2::exception::catch(AssertUnwindSafe(f)).map_err(|e| {
        e.map(|e| IrError::from(CompilerException::from(e)))
            .unwrap_or(IrError::CompilerFailed)
    })
}

impl GraphCompiler {
    pub fn new(program: &Program) -> Result<Self, IrError> {
        catch_native(|| unsafe { Self::lower(program) })?
    }
    unsafe fn lower(program: &Program) -> Result<Self, IrError> {
        let palettized = program.ops().iter().filter_map(|op| {
            let Op::Builtin(op) = op else { return None; };
            let tensor = op.outputs[0];
            if op.operation != O::ConstexprLutToDense || program.target_tensors().contains(&tensor) { return None; }
            let users = program.ops().iter().filter(|op| op.bottoms().contains(&tensor)).collect::<Vec<_>>();
            let only = |operation, parameter| users.iter().all(|op| matches!(op, Op::Builtin(op) if op.operation == operation && op.inputs.iter().filter(|(_,t)| *t == tensor).all(|(p,_)| *p == parameter)));
            let shared_table = op.blobs.iter().any(|b| b.name == P::Lut && b.shape[..4].iter().all(|&n| n == 1));
            let packed = (tensor.physical_shape()[..2] == [1, 1] && only(O::Matmul, P::Y))
                || (shared_table && only(O::Conv, P::Weight));
            (!users.is_empty() && packed).then(|| (tensor, op.clone()))
        }).collect();
        let mut compiler = Self {
            graph: unsafe { MPSGraph::new() },
            values: HashMap::new(),
            variables: HashMap::new(),
            palettized,
            feeds: Vec::new(),
            targets: Vec::new(),
            effects: Vec::new(),
            state_outputs: Vec::new(),
            discarded_outputs: Vec::new(),
        };
        let states: HashSet<_> = program
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::StateUpdate(op) => Some(op.state),
                _ => None,
            })
            .collect();
        for (&tensor, (_, shape, storage)) in program.feed_tensors().iter().zip(program.inputs()) {
            let shape = numbers(if *storage == DataType::Int32 {
                &[1]
            } else {
                shape
            });
            let feed = unsafe {
                compiler.graph.placeholderWithShape_dataType_name(
                    Some(&shape),
                    storage_type(*storage),
                    Some(&NSString::from_str(&tensor.symbol())),
                )
            };
            let shaped_type = unsafe {
                MPSGraphShapedType::initWithShape_dataType(
                    MPSGraphShapedType::alloc(),
                    Some(&shape),
                    storage_type(*storage),
                )
            };
            compiler.feeds.push((feed.clone(), shaped_type));
            let value = unsafe {
                compiler
                    .graph
                    .castTensor_toType_name(&feed, data_type(tensor.data_type()), None)
            };
            if states.contains(&tensor) {
                let variable = unsafe {
                    compiler.graph.variableFromTensorWithTensor_name(
                        &value,
                        Some(&NSString::from_str(&tensor.symbol())),
                    )
                };
                compiler.variables.insert(tensor, variable);
            }
            compiler.values.insert(tensor, value);
        }
        let mut written = HashMap::new();
        for op in program.ops() {
            match op {
                Op::Constant(op) => {
                    let tensor = compiler.constant(&op.data, &op.top.physical_shape());
                    compiler.values.insert(op.top, tensor);
                }
                Op::StateRead(op) => {
                    let tensor = if let Some(variable) = compiler.variables.get(&op.state) {
                        unsafe { compiler.graph.readVariable_name(variable, None) }
                    } else {
                        compiler.values[&op.state].clone()
                    };
                    compiler.values.insert(op.top, tensor);
                }
                Op::StateWrite(op) => {
                    let value = &compiler.values[&op.bottom];
                    let tensor = if let Some(variable) = compiler.variables.get(&op.state) {
                        let effect = unsafe {
                            compiler
                                .graph
                                .assignVariable_withValueOfTensor_name(variable, value, None)
                        };
                        compiler.effects.push(effect);
                        unsafe { compiler.graph.readVariable_name(variable, None) }
                    } else {
                        written.insert(op.state, value.clone());
                        value.clone()
                    };
                    compiler.values.insert(op.top, tensor);
                }
                Op::StateUpdate(op) => {
                    let variable = &compiler.variables[&op.state];
                    let previous = unsafe { compiler.graph.readVariable_name(variable, None) };
                    let shape = op.top.physical_shape();
                    let scalar = |value| compiler.integer(&[value]);
                    let zero = scalar(0);
                    let position = &compiler.values[&op.position];
                    let end = unsafe {
                        compiler
                            .graph
                            .additionWithPrimaryTensor_secondaryTensor_name(
                                position,
                                &scalar(op.rows as i32),
                                None,
                            )
                    };
                    let starts = unsafe {
                        compiler.graph.concatTensors_dimension_name(
                            &NSArray::from_retained_slice(&[
                                zero.clone(),
                                scalar(op.channel as i32),
                                position.clone(),
                                zero,
                            ]),
                            0,
                            None,
                        )
                    };
                    let ends = unsafe {
                        compiler.graph.concatTensors_dimension_name(
                            &NSArray::from_retained_slice(&[
                                scalar(shape[0] as i32),
                                scalar((op.channel + op.channels) as i32),
                                end,
                                scalar(shape[3] as i32),
                            ]),
                            0,
                            None,
                        )
                    };
                    let next = unsafe {
                        compiler.graph.sliceUpdateDataTensor_updateTensor_startsTensor_endsTensor_stridesTensor_startMask_endMask_squeezeMask_name(&previous, &compiler.values[&op.bottom], &starts, &ends, &compiler.integer(&[1,1,1,1]), 0,0,0,None)
                    };
                    let effect = unsafe {
                        compiler
                            .graph
                            .assignVariable_withValueOfTensor_name(variable, &next, None)
                    };
                    compiler.effects.push(effect);
                    let tensor = unsafe { compiler.graph.readVariable_name(variable, None) };
                    compiler.values.insert(op.top, tensor);
                }
                Op::Builtin(op) => compiler.builtin(op)?,
            }
        }
        for (&tensor, (_, _, dtype)) in program.target_tensors().iter().zip(program.outputs()) {
            let value = unsafe {
                compiler.graph.castTensor_toType_name(
                    &compiler.values[&tensor],
                    storage_type(*dtype),
                    Some(&NSString::from_str(&tensor.symbol())),
                )
            };
            compiler.targets.push(value);
        }
        for (index, tensor) in program.feed_tensors().iter().enumerate() {
            if let Some(value) = written.remove(tensor) {
                compiler.state_outputs.push(index);
                compiler.targets.push(value);
            }
        }
        for op in program.ops() {
            if let Op::Builtin(op) = op
                && op.operation == O::Topk
            {
                for &value in &op.outputs {
                    if !program.target_tensors().contains(&value)
                        && !program.ops().iter().any(|op| op.bottoms().contains(&value))
                    {
                        compiler.discarded_outputs.push(value);
                        let target = unsafe {
                            compiler.graph.castTensor_toType_name(
                                &compiler.values[&value],
                                data_type(value.data_type()),
                                None,
                            )
                        };
                        compiler.targets.push(target);
                    }
                }
            }
        }
        Ok(compiler)
    }
    pub fn executable(
        &self,
    ) -> Result<
        (
            Retained<MPSGraphExecutable>,
            Retained<MPSGraphCompilationDescriptor>,
        ),
        IrError,
    > {
        catch_native(|| unsafe {
            let class = AnyClass::get(c"NSMutableDictionary").unwrap();
            let feeds = Retained::from_raw(raw_message!(class, c"new"; *mut AnyObject)).unwrap();
            for (tensor, ty) in &self.feeds {
                raw_message!(&*feeds,c"setObject:forKey:",&**ty => &MPSGraphShapedType,&**tensor => &MPSGraphTensor; ());
            }
            let descriptor = MPSGraphCompilationDescriptor::new();

            raw_message!(&*descriptor,c"setPreferredDevice:",2u64 => u64; ());
            raw_message!(&*descriptor,c"setEnableANECValidationWorkflow:",Bool::YES => Bool; ());
            raw_message!(&*descriptor,c"setAneMaxRegions:",1u64 => u64; ());
            let executable = Retained::retain_autoreleased(raw_message!(&*self.graph,c"compileWithDevice:feeds:targetTensors:targetOperations:compilationDescriptor:",
                None => Option<&AnyObject>,&*feeds => &AnyObject,&*NSArray::from_retained_slice(&self.targets) => &NSArray<MPSGraphTensor>,
                &*NSArray::from_retained_slice(&self.effects) => &NSArray<MPSGraphOperation>,&*descriptor => &MPSGraphCompilationDescriptor;*mut MPSGraphExecutable)).unwrap();
            (executable, descriptor)
        })
    }
    pub fn feed_names(&self, executable: &MPSGraphExecutable) -> Result<Vec<String>, IrError> {
        catch_native(|| unsafe {
            let tensors = Retained::retain_autoreleased(
                raw_message!(executable,c"feedTensors"; *mut NSArray<MPSGraphTensor>),
            )
            .unwrap();
            tensors
                .iter()
                .map(|tensor| {
                    let name = Retained::retain_autoreleased(
                        raw_message!(&*tensor,c"name"; *mut NSString),
                    )
                    .unwrap();
                    name.to_string()
                })
                .collect()
        })
    }
    fn dequantize_lut(
        &self,
        indices: &WeightBlob,
        shape: &[usize],
        table: &WeightBlob,
    ) -> Retained<MPSGraphTensor> {
        let coefficients = self.constant(indices, shape);
        let lut = self.constant(table, &[1, 1, 1, 1, table.element_count()]);
        unsafe {
            self.graph
                .dequantizeTensor_LUTTensor_name(&coefficients, &lut, None)
        }
    }
    fn constant(&self, data: &WeightBlob, shape: &[usize]) -> Retained<MPSGraphTensor> {
        let dtype = match data.data_type() {
            WeightDataType::Float16 => MPSDataType::Float16,
            WeightDataType::Float32 => MPSDataType::Float32,
            WeightDataType::Int8 => MPSDataType::Int8,
            WeightDataType::Int16 => MPSDataType::Int16,
            WeightDataType::Int32 => MPSDataType::Int32,
            WeightDataType::Int4 => MPSDataType::Int4,
            dtype => MPSDataType(dtype.bit_width() as u32),
        };
        unsafe {
            self.graph.constantWithData_shape_dataType(
                &NSData::with_bytes(data.bytes()),
                &numbers(shape),
                dtype,
            )
        }
    }
    fn integer(&self, values: &[i32]) -> Retained<MPSGraphTensor> {
        unsafe {
            self.graph.constantWithData_shape_dataType(
                &NSData::with_bytes(bytemuck::cast_slice(values)),
                &numbers(&[values.len()]),
                MPSDataType::Int32,
            )
        }
    }
    fn scalar(&self, value: f64) -> Retained<MPSGraphTensor> {
        unsafe {
            self.graph
                .constantWithScalar_dataType(value, MPSDataType::Float16)
        }
    }
    fn value(&self, value: &Value) -> Retained<MPSGraphTensor> {
        let (bytes, shape, dtype) = match value {
            Value::Fp16(v) | Value::Fp32(v) => return self.scalar(*v as f64),
            Value::Bool(v) => (vec![u8::from(*v)], vec![], MPSDataType::Bool),
            Value::Integer(dtype, v) => {
                return unsafe {
                    self.graph
                        .constantWithScalar_dataType(*v as f64, data_type(*dtype))
                };
            }
            Value::Fp16List(v) => (
                bytemuck::cast_slice(
                    &v.iter()
                        .copied()
                        .map(half::f16::from_f32)
                        .collect::<Vec<_>>(),
                )
                .to_vec(),
                vec![v.len()],
                MPSDataType::Float16,
            ),
            Value::IntegerList(dtype, v) => {
                let bytes = match dtype {
                    DataType::Int8 => v.iter().map(|&v| v as i8 as u8).collect(),
                    DataType::UInt8 => v.iter().map(|&v| v as u8).collect(),
                    DataType::Int16 | DataType::UInt16 => {
                        v.iter().flat_map(|&v| (v as u16).to_le_bytes()).collect()
                    }
                    _ => bytemuck::cast_slice(v).to_vec(),
                };
                (bytes, vec![v.len()], data_type(*dtype))
            }
            _ => unreachable!("validated numeric attribute required"),
        };
        unsafe {
            self.graph.constantWithData_shape_dataType(
                &NSData::with_bytes(&bytes),
                &numbers(&shape),
                dtype,
            )
        }
    }
    fn builtin(&mut self, op: &BuiltinOp) -> Result<(), IrError> {
        if op.operation.is_constant() {
            if self.palettized.contains_key(&op.outputs[0]) {
                return Ok(());
            }
            let tensor = self.constant(&op.constant_data()?, &op.outputs[0].physical_shape());
            self.values.insert(op.outputs[0], tensor);
            return Ok(());
        }
        let attr = |key| {
            op.attributes
                .iter()
                .find(|(p, _)| *p == key)
                .map(|(_, v)| v)
        };
        let int = |key| {
            if let Some(Value::Int32(v)) = attr(key) {
                *v
            } else {
                unreachable!("validated integer required")
            }
        };
        let float = |key| match attr(key) {
            Some(Value::Fp16(v) | Value::Fp32(v)) => *v as f64,
            _ => unreachable!("validated float required"),
        };
        let flag = |key| attr(key) == Some(&Value::Bool(true));
        let list = |key| {
            if let Some(Value::Int32List(v)) = attr(key) {
                v.as_ref()
            } else {
                unreachable!("validated list required")
            }
        };
        let string = |key| {
            if let Some(Value::String(v)) = attr(key) {
                *v
            } else {
                unreachable!("validated string required")
            }
        };
        let mask = |key| {
            if let Some(Value::BoolList(v)) = attr(key) {
                v.iter()
                    .enumerate()
                    .fold(0u32, |m, (i, v)| m | (u32::from(*v) << i))
            } else {
                0
            }
        };
        let mut inputs = HashMap::new();
        for &(key, tensor) in &op.inputs {
            if self.palettized.contains_key(&tensor) {
                continue;
            }
            let value = self.values[&tensor].clone();
            let physical = tensor.physical_shape();
            let value = if op.logical {
                unsafe {
                    self.graph.reshapeTensor_withShape_name(
                        &value,
                        &numbers(if op.operation == O::Gather && key == P::X {
                            &physical
                        } else {
                            tensor.shape()
                        }),
                        None,
                    )
                }
            } else {
                value
            };
            inputs.insert(key, value);
        }
        for blob in &op.blobs {
            inputs.insert(blob.name, self.constant(&blob.data, &blob.shape));
        }
        let arg = |key| {
            inputs
                .get(&key)
                .cloned()
                .unwrap_or_else(|| self.value(attr(key).unwrap()))
        };
        let x = || arg(P::X);
        let g = &*self.graph;
        macro_rules! unary {
            ($method:ident) => {
                unsafe { g.$method(&x(), None) }
            };
        }
        macro_rules! binary {
            ($method:ident) => {
                unsafe { g.$method(&arg(P::X), &arg(P::Y), None) }
            };
        }
        let result = match op.operation {
            O::Abs => unary!(absoluteWithTensor_name),
            O::Atan => unary!(atanWithTensor_name),
            O::Ceil => unary!(ceilWithTensor_name),
            O::Cos => unary!(cosWithTensor_name),
            O::Erf => unary!(erfWithTensor_name),
            O::Exp => unary!(exponentWithTensor_name),
            O::Exp2 => unary!(exponentBase2WithTensor_name),
            O::Floor => unary!(floorWithTensor_name),
            O::Relu => unary!(reLUWithTensor_name),
            O::Round => unary!(roundWithTensor_name),
            O::Sigmoid => unary!(sigmoidWithTensor_name),
            O::Sign => unary!(signWithTensor_name),
            O::Sin => unary!(sinWithTensor_name),
            O::Sqrt => unary!(squareRootWithTensor_name),
            O::Tanh => unary!(tanhWithTensor_name),
            O::LogicalNot => unary!(notWithTensor_name),
            O::LogicalAnd => binary!(logicalANDWithPrimaryTensor_secondaryTensor_name),
            O::LogicalOr => binary!(logicalORWithPrimaryTensor_secondaryTensor_name),
            O::Add => binary!(additionWithPrimaryTensor_secondaryTensor_name),
            O::Sub => binary!(subtractionWithPrimaryTensor_secondaryTensor_name),
            O::Mul => binary!(multiplicationWithPrimaryTensor_secondaryTensor_name),
            O::RealDiv => binary!(divisionWithPrimaryTensor_secondaryTensor_name),
            O::Pow => binary!(powerWithPrimaryTensor_secondaryTensor_name),
            O::Maximum => binary!(maximumWithPrimaryTensor_secondaryTensor_name),
            O::Minimum => binary!(minimumWithPrimaryTensor_secondaryTensor_name),
            O::Equal => binary!(equalWithPrimaryTensor_secondaryTensor_name),
            O::NotEqual => binary!(notEqualWithPrimaryTensor_secondaryTensor_name),
            O::Less => binary!(lessThanWithPrimaryTensor_secondaryTensor_name),
            O::LessEqual => binary!(lessThanOrEqualToWithPrimaryTensor_secondaryTensor_name),
            O::Greater => binary!(greaterThanWithPrimaryTensor_secondaryTensor_name),
            O::GreaterEqual => binary!(greaterThanOrEqualToWithPrimaryTensor_secondaryTensor_name),
            O::Select => unsafe {
                g.selectWithPredicateTensor_truePredicateTensor_falsePredicateTensor_name(
                    &arg(P::Cond),
                    &arg(P::A),
                    &arg(P::B),
                    None,
                )
            },
            O::Silu => unsafe {
                g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    &x(),
                    &g.sigmoidWithTensor_name(&x(), None),
                    None,
                )
            },
            O::Softsign => unsafe {
                g.divisionWithPrimaryTensor_secondaryTensor_name(
                    &x(),
                    &g.additionWithPrimaryTensor_secondaryTensor_name(
                        &g.absoluteWithTensor_name(&x(), None),
                        &self.scalar(1.),
                        None,
                    ),
                    None,
                )
            },
            O::Softplus => unsafe {
                let negative =
                    g.negativeWithTensor_name(&g.absoluteWithTensor_name(&x(), None), None);
                let exp = g.exponentWithTensor_name(&negative, None);
                let log = g.logarithmWithTensor_name(
                    &g.additionWithPrimaryTensor_secondaryTensor_name(&exp, &self.scalar(1.), None),
                    None,
                );
                let positive =
                    g.maximumWithPrimaryTensor_secondaryTensor_name(&x(), &self.scalar(0.), None);
                g.additionWithPrimaryTensor_secondaryTensor_name(&log, &positive, None)
            },
            O::Log | O::Inverse | O::Rsqrt => unsafe {
                let epsilon = float(P::Epsilon);
                let source = if epsilon == 0.0 {
                    x()
                } else {
                    g.additionWithPrimaryTensor_secondaryTensor_name(
                        &x(),
                        &self.scalar(epsilon),
                        None,
                    )
                };
                match op.operation {
                    O::Log => g.logarithmWithTensor_name(&source, None),
                    O::Inverse => g.reciprocalWithTensor_name(&source, None),
                    _ => g.reciprocalSquareRootWithTensor_name(&source, None),
                }
            },
            O::LeakyRelu => unsafe {
                g.leakyReLUWithTensor_alpha_name(&x(), float(P::Alpha), None)
            },
            O::LinearActivation => unsafe {
                g.additionWithPrimaryTensor_secondaryTensor_name(
                    &g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                        &x(),
                        &self.scalar(float(P::Alpha)),
                        None,
                    ),
                    &self.scalar(float(P::Beta)),
                    None,
                )
            },
            O::Elu => unsafe {
                let negative =
                    g.minimumWithPrimaryTensor_secondaryTensor_name(&x(), &self.scalar(0.), None);
                let exp = g.subtractionWithPrimaryTensor_secondaryTensor_name(
                    &g.exponentWithTensor_name(&negative, None),
                    &self.scalar(1.),
                    None,
                );
                let exp = g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    &exp,
                    &self.scalar(float(P::Alpha)),
                    None,
                );
                g.additionWithPrimaryTensor_secondaryTensor_name(
                    &g.maximumWithPrimaryTensor_secondaryTensor_name(&x(), &self.scalar(0.), None),
                    &exp,
                    None,
                )
            },
            O::Threshold => unsafe {
                g.maximumWithPrimaryTensor_secondaryTensor_name(
                    &x(),
                    &self.scalar(float(P::Alpha)),
                    None,
                )
            },
            O::ThresholdedRelu => unsafe {
                let predicate = g.greaterThanOrEqualToWithPrimaryTensor_secondaryTensor_name(
                    &x(),
                    &self.scalar(float(P::Alpha)),
                    None,
                );
                g.selectWithPredicateTensor_truePredicateTensor_falsePredicateTensor_name(
                    &predicate,
                    &x(),
                    &self.scalar(0.),
                    None,
                )
            },
            O::Gelu => unsafe {
                let source = x();
                let cdf = if string(P::Mode) == "EXACT" {
                    g.erfWithTensor_name(
                        &g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                            &source,
                            &self.scalar(std::f64::consts::FRAC_1_SQRT_2),
                            None,
                        ),
                        None,
                    )
                } else {
                    let cube = g.powerWithPrimaryTensor_secondaryTensor_name(
                        &source,
                        &self.scalar(3.),
                        None,
                    );
                    let inner = g.additionWithPrimaryTensor_secondaryTensor_name(
                        &source,
                        &g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                            &cube,
                            &self.scalar(0.044715),
                            None,
                        ),
                        None,
                    );
                    g.tanhWithTensor_name(
                        &g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                            &inner,
                            &self.scalar((2. / std::f64::consts::PI).sqrt()),
                            None,
                        ),
                        None,
                    )
                };
                let cdf =
                    g.additionWithPrimaryTensor_secondaryTensor_name(&cdf, &self.scalar(1.), None);
                let cdf = g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    &cdf,
                    &self.scalar(0.5),
                    None,
                );
                g.multiplicationWithPrimaryTensor_secondaryTensor_name(&source, &cdf, None)
            },
            O::Softmax => unsafe {
                g.softMaxWithTensor_axis_name(&x(), int(P::Axis) as isize, None)
            },
            O::ReduceSum | O::ReduceMean | O::ReduceMin | O::ReduceMax => unsafe {
                let axes = numbers(list(P::Axes));
                match op.operation {
                    O::ReduceSum => g.reductionSumWithTensor_axes_name(&x(), Some(&axes), None),
                    O::ReduceMean => g.meanOfTensor_axes_name(&x(), &axes, None),
                    O::ReduceMin => g.reductionMinimumWithTensor_axes_name(&x(), Some(&axes), None),
                    _ => g.reductionMaximumWithTensor_axes_name(&x(), Some(&axes), None),
                }
            },
            O::Matmul => unsafe {
                let transpose = |key, flag| {
                    let source = arg(key);
                    if flag {
                        g.transposeTensor_dimension_withDimension_name(&source, 2, 3, None)
                    } else {
                        source
                    }
                };
                let left = transpose(P::X, flag(P::TransposeX));
                let weight = op.inputs.iter().find(|(p, _)| *p == P::Y).unwrap().1;
                if let Some(palette) = self
                    .palettized
                    .get(&weight)
                    .filter(|_| weight.physical_shape()[..2] == [1, 1])
                {
                    let shape = op.outputs[0].physical_shape();
                    let input = op
                        .inputs
                        .iter()
                        .find(|(p, _)| *p == P::X)
                        .unwrap()
                        .1
                        .physical_shape();
                    let k = input[if flag(P::TransposeX) { 2 } else { 3 }];
                    let left = g.reshapeTensor_withShape_name(
                        &g.transposeTensor_dimension_withDimension_name(&left, 2, 3, None),
                        &numbers(&[shape[0] * shape[1], k, 1, shape[2]]),
                        None,
                    );
                    let blob = |key| palette.blobs.iter().find(|b| b.name == key).unwrap();
                    let (data, mut table) =
                        blob(P::Indices).data.native_palette(&blob(P::Lut).data)?;
                    let mut groups = blob(P::Lut).shape[..4].to_vec();
                    let entries = 1usize << data.data_type().bit_width();
                    let indices = if !flag(P::TransposeY) {
                        table = table.transpose_blocks(groups[2], groups[3], entries)?;
                        groups.swap(2, 3);
                        data.transpose_blocks(k, shape[3], 1)?
                    } else {
                        data
                    };
                    let output_groups = groups[2];
                    let input_groups = groups[3];
                    let block_n = shape[3] / output_groups;
                    let block_k = k / input_groups;
                    let descriptor = MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(1,1,1,1,1,0,0,0,0,MPSGraphPaddingStyle::Explicit,MPSGraphTensorNamedDataLayout::NCHW,MPSGraphTensorNamedDataLayout::OIHW).unwrap();
                    let mut tiles = Vec::new();
                    for output_group in 0..output_groups {
                        let mut sum: Option<Retained<MPSGraphTensor>> = None;
                        for input_group in 0..input_groups {
                            let tile = if input_groups == 1 && output_groups == 1 {
                                indices.clone()
                            } else {
                                indices.slice_matrix(
                                    [shape[3], k],
                                    [output_group * block_n, input_group * block_k],
                                    [block_n, block_k],
                                )?
                            };
                            let group = output_group * input_groups + input_group;
                            let palette = WeightBlob::from_bytes(
                                &table.bytes()[group * entries * 2..(group + 1) * entries * 2],
                                entries,
                                WeightDataType::Float16,
                            )?;
                            let weights =
                                self.dequantize_lut(&tile, &[block_n, block_k, 1, 1], &palette);
                            let input = if input_groups == 1 {
                                left.clone()
                            } else {
                                g.sliceTensor_dimension_start_length_name(
                                    &left,
                                    1,
                                    (input_group * block_k) as isize,
                                    block_k as isize,
                                    None,
                                )
                            };
                            let result = g
                                .convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                                    &input,
                                    &weights,
                                    &descriptor,
                                    None,
                                );
                            sum = Some(if let Some(previous) = sum {
                                g.additionWithPrimaryTensor_secondaryTensor_name(
                                    &previous, &result, None,
                                )
                            } else {
                                result
                            });
                        }
                        tiles.push(sum.unwrap());
                    }
                    let result = if tiles.len() == 1 {
                        tiles.remove(0)
                    } else {
                        g.concatTensors_dimension_name(
                            &NSArray::from_retained_slice(&tiles),
                            1,
                            None,
                        )
                    };
                    let result = g.reshapeTensor_withShape_name(
                        &result,
                        &numbers(&[shape[0], shape[1], shape[3], shape[2]]),
                        None,
                    );
                    g.transposeTensor_dimension_withDimension_name(&result, 2, 3, None)
                } else {
                    let right = transpose(P::Y, flag(P::TransposeY));
                    g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(
                        &left, &right, None,
                    )
                }
            },
            O::ScaledDotProductAttention => unsafe {
                g.scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_name(&arg(P::Query),&arg(P::Key),&arg(P::Value),inputs.get(&P::AttnMask).map(|t|&**t),1.0/(op.inputs.iter().find(|(p,_)|*p==P::Query).unwrap().1.physical_shape()[3] as f32).sqrt(),None)
            },
            O::Reshape => unsafe {
                g.reshapeTensor_withShape_name(&x(), &numbers(list(P::Shape)), None)
            },
            O::Transpose => unsafe {
                g.transposeTensor_permutation_name(&x(), &numbers(list(P::Perm)), None)
            },
            O::Cast => unsafe {
                let input = x();
                let dtype = op.outputs[0].data_type();
                if input.dataType() == MPSDataType::Float16
                    && matches!(dtype, DataType::Int8 | DataType::UInt8)
                {
                    let value = g.truncateWithTensor_name(&input, None);
                    g.quantizeTensor_scale_zeroPoint_dataType_name(
                        &value,
                        1.0,
                        0.0,
                        data_type(dtype),
                        None,
                    )
                } else {
                    g.castTensor_toType_name(&input, data_type(dtype), None)
                }
            },
            O::SliceBySize => unsafe {
                let source = x();
                let mut value = source;
                for (axis, (&start, &size)) in list(P::Begin).iter().zip(list(P::Size)).enumerate()
                {
                    value = g.sliceTensor_dimension_start_length_name(
                        &value,
                        axis,
                        start as isize,
                        size as isize,
                        None,
                    );
                }
                value
            },
            O::DynamicSlice => unsafe {
                let axis = int(P::Axis);
                let starts = (0..4)
                    .map(|i| {
                        if i == axis {
                            arg(P::Begin)
                        } else {
                            self.integer(&[0])
                        }
                    })
                    .collect::<Vec<_>>();
                let starts =
                    g.concatTensors_dimension_name(&NSArray::from_retained_slice(&starts), 0, None);
                let mut size = op
                    .inputs
                    .iter()
                    .find(|(p, _)| *p == P::X)
                    .unwrap()
                    .1
                    .physical_shape();
                size[axis] = int(P::Size);
                let sizes = self.integer(&size.map(|v| v as i32));
                Retained::retain_autoreleased(raw_message!(g,
                    c"sliceTensor:startTensor:sizeTensor:squeezeMask:name:", &*x() => &MPSGraphTensor,
                    &*starts => &MPSGraphTensor, &*sizes => &MPSGraphTensor, 0u32 => u32,
                    None => Option<&NSString>; *mut MPSGraphTensor)).ok_or(IrError::CompilerFailed)?
            },
            O::SliceByIndex => unsafe {
                g.sliceTensor_starts_ends_strides_startMask_endMask_squeezeMask_name(
                    &x(),
                    &numbers(list(P::Begin)),
                    &numbers(list(P::End)),
                    &numbers(list(P::Stride)),
                    mask(P::BeginMask),
                    mask(P::EndMask),
                    mask(P::SqueezeMask),
                    None,
                )
            },
            O::Concat => unsafe {
                let values = op
                    .inputs
                    .iter()
                    .filter(|(p, _)| *p == P::Values)
                    .map(|(_, t)| self.values[t].clone())
                    .collect::<Vec<_>>();
                g.concatTensors_dimension_interleave_name(
                    &NSArray::from_retained_slice(&values),
                    int(P::Axis) as isize,
                    flag(P::Interleave),
                    None,
                )
            },
            O::Tile => unsafe {
                g.tileTensor_withMultiplier_name(&x(), &numbers(list(P::Reps)), None)
            },
            O::Reverse => unsafe { g.reverseTensor_axes_name(&x(), &numbers(list(P::Axes)), None) },
            O::Pad => unsafe {
                let pad = list(P::Pad);
                g.padTensor_withPaddingMode_leftPadding_rightPadding_constantValue_name(
                    &x(),
                    padding_mode(string(P::Mode)),
                    &numbers(&[0, 0, pad[0], pad[2]]),
                    &numbers(&[0, 0, pad[1], pad[3]]),
                    float(P::ConstantVal),
                    None,
                )
            },
            O::DepthToSpace | O::SpaceToDepth | O::PixelShuffle | O::PixelUnshuffle => unsafe {
                let key = match op.operation {
                    O::PixelShuffle => P::UpscaleFactor,
                    O::PixelUnshuffle => P::DownscaleFactor,
                    _ => P::BlockSize,
                };
                if matches!(op.operation, O::DepthToSpace | O::PixelShuffle) {
                    g.depthToSpace2DTensor_widthAxis_heightAxis_depthAxis_blockSize_usePixelShuffleOrder_name(&x(),3,2,1,int(key),op.operation==O::PixelShuffle,None)
                } else {
                    g.spaceToDepth2DTensor_widthAxis_heightAxis_depthAxis_blockSize_usePixelShuffleOrder_name(&x(),3,2,1,int(key),op.operation==O::PixelUnshuffle,None)
                }
            },
            O::BatchToSpace => unsafe {
                let Some(Value::Int32Matrix([[top, _], [left, _]])) = attr(P::Crops) else {
                    unreachable!()
                };
                let axes = numbers(&[2, 3]);
                let blocks = numbers(list(P::BlockShape));
                let expanded = g.batchToSpaceTensor_spatialAxes_batchAxis_blockDimensions_usePixelShuffleOrder_name(&x(),&axes,0,&blocks,false,None);
                let shape = op.outputs[0].physical_shape();
                let cropped = g.sliceTensor_dimension_start_length_name(
                    &expanded,
                    2,
                    *top as isize,
                    shape[2] as isize,
                    None,
                );
                g.sliceTensor_dimension_start_length_name(
                    &cropped,
                    3,
                    *left as isize,
                    shape[3] as isize,
                    None,
                )
            },
            O::AvgPool | O::MaxPool | O::L2Pool => unsafe {
                let kernel = list(P::KernelSizes);
                let strides = list(P::Strides);
                let pad = list(P::Pad);
                let style = if string(P::PadType) == "same_lower" {
                    MPSGraphPaddingStyle::ONNX_SAME_LOWER
                } else {
                    MPSGraphPaddingStyle::Explicit
                };
                let desc=MPSGraphPooling2DOpDescriptor::descriptorWithKernelWidth_kernelHeight_strideInX_strideInY_dilationRateInX_dilationRateInY_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout(kernel[1],kernel[0],strides[1],strides[0],1,1,pad[2],pad[3],pad[0],pad[1],style,MPSGraphTensorNamedDataLayout::NCHW).unwrap();
                desc.setCeilMode(flag(P::CeilMode));
                desc.setIncludeZeroPadToAverage(!flag(P::ExcludePaddingFromAverage));
                if op.operation == O::MaxPool {
                    g.maxPooling2DWithSourceTensor_descriptor_name(&x(), &desc, None)
                } else {
                    let source = if op.operation == O::L2Pool {
                        g.squareWithTensor_name(&x(), None)
                    } else {
                        x()
                    };
                    let result =
                        g.avgPooling2DWithSourceTensor_descriptor_name(&source, &desc, None);
                    if op.operation == O::L2Pool {
                        let sum = g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                            &result,
                            &self.scalar((kernel[0] * kernel[1]) as f64),
                            None,
                        );
                        g.squareRootWithTensor_name(&sum, None)
                    } else {
                        result
                    }
                }
            },
            O::LocalResponseNorm => unsafe {
                let source = x();
                let channels = op.outputs[0].physical_shape()[1];
                let size = int(P::Size);
                let square = g.squareWithTensor_name(&source, None);
                let padded = g
                    .padTensor_withPaddingMode_leftPadding_rightPadding_constantValue_name(
                        &square,
                        MPSGraphPaddingMode::Constant,
                        &numbers(&[0, (size - 1) / 2, 0, 0]),
                        &numbers(&[0, size / 2, 0, 0]),
                        0.,
                        None,
                    );
                let mut sum = g.sliceTensor_dimension_start_length_name(
                    &padded,
                    1,
                    0,
                    channels as isize,
                    None,
                );
                for offset in 1..size {
                    let part = g.sliceTensor_dimension_start_length_name(
                        &padded,
                        1,
                        offset as isize,
                        channels as isize,
                        None,
                    );
                    sum = g.additionWithPrimaryTensor_secondaryTensor_name(&sum, &part, None);
                }
                let sum = g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                    &sum,
                    &self.scalar(float(P::Alpha) / size as f64),
                    None,
                );
                let sum = g.additionWithPrimaryTensor_secondaryTensor_name(
                    &sum,
                    &self.scalar(float(P::K)),
                    None,
                );
                let denominator = g.powerWithPrimaryTensor_secondaryTensor_name(
                    &sum,
                    &self.scalar(float(P::Beta)),
                    None,
                );
                g.divisionWithPrimaryTensor_secondaryTensor_name(&source, &denominator, None)
            },
            O::Topk => unsafe {
                let results = if flag(P::Ascending) {
                    g.bottomKWithSourceTensor_axis_k_name(
                        &x(),
                        int(P::Axis) as isize,
                        int(P::K),
                        None,
                    )
                } else {
                    g.topKWithSourceTensor_axis_k_name(&x(), int(P::Axis) as isize, int(P::K), None)
                };
                let index =
                    g.castTensor_toType_name(&results.objectAtIndex(1), MPSDataType::UInt16, None);
                let index = g.reshapeTensor_withShape_name(
                    &index,
                    &numbers(&op.outputs[1].physical_shape()),
                    None,
                );
                self.values.insert(op.outputs[1], index);
                results.objectAtIndex(0)
            },
            O::Gather | O::GatherAlongAxis | O::GatherNd => unsafe {
                let indices = arg(P::Indices);
                match op.operation {
                    O::Gather => g.gatherWithUpdatesTensor_indicesTensor_axis_batchDimensions_name(
                        &x(),
                        &indices,
                        int(P::Axis),
                        int(P::BatchDims),
                        None,
                    ),
                    O::GatherAlongAxis => g.gatherAlongAxis_withUpdatesTensor_indicesTensor_name(
                        int(P::Axis) as isize,
                        &x(),
                        &indices,
                        None,
                    ),
                    _ => g.gatherNDWithUpdatesTensor_indicesTensor_batchDimensions_name(
                        &x(),
                        &indices,
                        int(P::BatchDims),
                        None,
                    ),
                }
            },
            O::ResizeNearestNeighbor | O::ResizeBilinear => unsafe {
                let size = self.integer(&[
                    int(P::TargetSizeHeight) as i32,
                    int(P::TargetSizeWidth) as i32,
                ]);
                if op.operation == O::ResizeBilinear {
                    g.resizeBilinearWithTensor_sizeTensor_centerResult_alignCorners_layout_name(
                        &x(),
                        &size,
                        true,
                        false,
                        MPSGraphTensorNamedDataLayout::NCHW,
                        None,
                    )
                } else {
                    g.resizeNearestWithTensor_sizeTensor_nearestRoundingMode_centerResult_alignCorners_layout_name(&x(),&size,MPSGraphResizeNearestRoundingMode::Floor,false,false,MPSGraphTensorNamedDataLayout::NCHW,None)
                }
            },
            O::Resample => unsafe {
                let mut coordinates = arg(P::Coordinates);
                let mode = string(P::CoordinatesMode);
                if mode == "normalized_zero_to_one" {
                    coordinates = g.multiplicationWithPrimaryTensor_secondaryTensor_name(
                        &coordinates,
                        &self.scalar(2.),
                        None,
                    );
                    coordinates = g.subtractionWithPrimaryTensor_secondaryTensor_name(
                        &coordinates,
                        &self.scalar(1.),
                        None,
                    );
                }
                let sampling = if string(P::SamplingMode) == "nearest" {
                    MPSGraphResizeMode::Nearest
                } else {
                    MPSGraphResizeMode::Bilinear
                };
                g.sampleGridWithSourceTensor_coordinateTensor_layout_normalizeCoordinates_relativeCoordinates_alignCorners_paddingMode_samplingMode_constantValue_name(&x(),&coordinates,MPSGraphTensorNamedDataLayout::NCHW,mode!="unnormalized",false,flag(P::AlignCorners),padding_mode(string(P::PaddingMode)),sampling,float(P::PaddingValue),None)
            },
            O::Affine => unsafe {
                let height = int(P::OutputHeight);
                let width = int(P::OutputWidth);
                let mut grid = Vec::with_capacity(3 * height * width);
                grid.extend((0..height * width).map(|i| {
                    if width == 1 {
                        0.
                    } else {
                        2. * (i % width) as f32 / (width - 1) as f32 - 1.
                    }
                }));
                grid.extend((0..height * width).map(|i| {
                    if height == 1 {
                        0.
                    } else {
                        2. * (i / width) as f32 / (height - 1) as f32 - 1.
                    }
                }));
                grid.resize(3 * height * width, 1.);
                let grid = self.constant(&WeightBlob::from_f32(&grid)?, &[1, 3, height * width]);
                let batch = op
                    .inputs
                    .iter()
                    .find(|(p, _)| *p == P::TransformMatrix)
                    .unwrap()
                    .1
                    .physical_shape()[2];
                let matrix = g.reshapeTensor_withShape_name(
                    &arg(P::TransformMatrix),
                    &numbers(&[batch, 2, 3]),
                    None,
                );
                let coordinates = g.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(
                    &matrix, &grid, None,
                );
                let coordinates =
                    g.transposeTensor_dimension_withDimension_name(&coordinates, 1, 2, None);
                let coordinates = g.reshapeTensor_withShape_name(
                    &coordinates,
                    &numbers(&[batch, height, width, 2]),
                    None,
                );
                let coordinates = g.broadcastTensor_toShape_name(
                    &coordinates,
                    &numbers(&[op.outputs[0].physical_shape()[0], height, width, 2]),
                    None,
                );
                g.sampleGridWithSourceTensor_coordinateTensor_layout_normalizeCoordinates_relativeCoordinates_alignCorners_paddingMode_samplingMode_constantValue_name(&x(),&coordinates,MPSGraphTensorNamedDataLayout::NCHW,true,false,true,MPSGraphPaddingMode::Constant,MPSGraphResizeMode::Bilinear,0.,None)
            },
            O::Quantize | O::Dequantize => unsafe {
                let input = arg(P::Input);
                let scale = arg(P::Scale);
                let zero = if attr(P::ZeroPoint).is_some() {
                    arg(P::ZeroPoint)
                } else {
                    g.constantWithScalar_dataType(
                        0.,
                        if op.operation == O::Quantize {
                            data_type(op.outputs[0].data_type())
                        } else {
                            input.dataType()
                        },
                    )
                };
                let axis = attr(P::Axis).map_or(0, |_| int(P::Axis)) as isize;
                if op.operation == O::Quantize {
                    g.quantizeTensor_scaleTensor_zeroPointTensor_dataType_axis_name(
                        &input,
                        &scale,
                        &zero,
                        data_type(op.outputs[0].data_type()),
                        axis,
                        None,
                    )
                } else {
                    g.dequantizeTensor_scaleTensor_zeroPointTensor_dataType_axis_name(
                        &input,
                        &scale,
                        &zero,
                        MPSDataType::Float16,
                        axis,
                        None,
                    )
                }
            },
            O::Conv | O::ConvTranspose => unsafe {
                let strides = list(P::Strides);
                let dilations = list(P::Dilations);
                let pad = list(P::Pad);
                let style = if string(P::PadType) == "same_lower" {
                    MPSGraphPaddingStyle::ONNX_SAME_LOWER
                } else {
                    MPSGraphPaddingStyle::Explicit
                };
                let desc=MPSGraphConvolution2DOpDescriptor::descriptorWithStrideInX_strideInY_dilationRateInX_dilationRateInY_groups_paddingLeft_paddingRight_paddingTop_paddingBottom_paddingStyle_dataLayout_weightsLayout(strides[1],strides[0],dilations[1],dilations[0],int(P::Groups),pad[2],pad[3],pad[0],pad[1],style,MPSGraphTensorNamedDataLayout::NCHW,MPSGraphTensorNamedDataLayout::OIHW).unwrap();
                let weight = op.inputs.iter().find(|(p, _)| *p == P::Weight).unwrap().1;
                let weights = match self.palettized.get(&weight) {
                    Some(palette) => {
                        let blob = |key| palette.blobs.iter().find(|b| b.name == key).unwrap();
                        let (data, table) =
                            blob(P::Indices).data.native_palette(&blob(P::Lut).data)?;
                        self.dequantize_lut(&data, &weight.physical_shape(), &table)
                    }
                    None => arg(P::Weight),
                };
                let conv = if op.operation == O::Conv {
                    g.convolution2DWithSourceTensor_weightsTensor_descriptor_name(
                        &x(),
                        &weights,
                        &desc,
                        None,
                    )
                } else {
                    g.convolutionTranspose2DWithSourceTensor_weightsTensor_outputShape_descriptor_name(&x(),&arg(P::Weight),&numbers(list(P::OutputShape)),&desc,None)
                };
                if let Some(bias) = inputs.get(&P::Bias) {
                    let bias = g.reshapeTensor_withShape_name(
                        bias,
                        &numbers(&[1, op.outputs[0].physical_shape()[1], 1, 1]),
                        None,
                    );
                    g.additionWithPrimaryTensor_secondaryTensor_name(&conv, &bias, None)
                } else {
                    conv
                }
            },
            O::ConstexprBlockwiseShiftScale
            | O::ConstexprLutToDense
            | O::ConstexprSparseToDense
            | O::SparseBlockwiseWeights
            | O::SparsePaletteWeights => unreachable!("constant operation was materialized"),
        };
        let result = unsafe {
            g.reshapeTensor_withShape_name(&result, &numbers(&op.outputs[0].physical_shape()), None)
        };
        self.values.insert(op.outputs[0], result);
        Ok(())
    }
}
