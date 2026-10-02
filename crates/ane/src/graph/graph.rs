use crate::graph::{GraphError, ensure};
use crate::ir::{ConstantOp, Op, Operator, Parameter, Program, Value};
use std::cell::{RefCell, RefMut};
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::graph::{
    GraphBuilder, GraphState, LoweredGraph, Operation, Tensor, TensorHandle, operation_tensor,
};
use crate::{CompilationCache, CompilationDescriptor, DataType, Error, Executable};

pub struct Graph {
    state: RefCell<GraphState>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            state: RefCell::new(GraphState::default()),
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder for Graph {
    fn state(&self) -> RefMut<'_, GraphState> {
        self.state.borrow_mut()
    }
}

impl Graph {
    pub fn compile(
        &self,
        targets: &[Tensor],
        target_operations: &[&Operation],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Result<Executable, Error> {
        let default = CompilationDescriptor::default();
        let descriptor = descriptor.unwrap_or(&default);
        let program = self.compiled_program(targets, target_operations, descriptor)?;
        self.bind_executable(
            Executable::compile(program, descriptor.quality_of_service)?,
            targets,
        )
    }

    pub fn compile_cached(
        &self,
        targets: &[Tensor],
        target_operations: &[&Operation],
        descriptor: Option<&CompilationDescriptor>,
        cache: &mut CompilationCache,
    ) -> Result<Executable, Error> {
        let default = CompilationDescriptor::default();
        let descriptor = descriptor.unwrap_or(&default);
        let program = self.compiled_program(targets, target_operations, descriptor)?;
        let executable = cache.compile(program, descriptor.quality_of_service)?;
        self.bind_executable((*executable).clone(), targets)
    }

    fn compiled_program(
        &self,
        targets: &[Tensor],
        target_operations: &[&Operation],
        descriptor: &CompilationDescriptor,
    ) -> Result<Program, GraphError> {
        let effects: Vec<_> = target_operations
            .iter()
            .map(|operation| operation_tensor(operation))
            .collect();
        let types = match &descriptor.output_types {
            Some(types) => types.clone(),
            None => targets
                .iter()
                .map(|t| {
                    if t.data_type() == DataType::Float16 {
                        DataType::Float32
                    } else {
                        t.data_type()
                    }
                })
                .collect(),
        };
        self.checked_program(targets, &types, &effects)
    }

    pub fn program(
        &self,
        outputs: &[Tensor],
        output_types: &[DataType],
    ) -> Result<Program, GraphError> {
        let LoweredGraph { ops, inputs } = self.lower(outputs, output_types, &[])?;
        Ok(Program::new(&ops, &inputs, outputs, output_types)?)
    }

    fn lower(
        &self,
        outputs: &[Tensor],
        output_types: &[DataType],
        effects: &[Tensor],
    ) -> Result<LoweredGraph, GraphError> {
        let state = self.state.borrow();
        ensure(
            !outputs.is_empty() || !effects.is_empty(),
            GraphError::InvalidTargets("select at least one graph output or state effect"),
        )?;
        ensure(
            output_types.len() == 1 || output_types.len() == outputs.len(),
            GraphError::InvalidTargets("output storage type count differs"),
        )?;
        let mut needed = HashSet::new();
        for &output in outputs {
            state.check_tensor(output)?;
            ensure(
                needed.insert(output),
                GraphError::InvalidTargets("duplicate graph output"),
            )?;
        }
        for &effect in effects {
            state.check_tensor(effect)?;
            ensure(
                state.ops.iter().any(|(op, t)| {
                    *t == effect && matches!(op, Op::StateUpdate(_) | Op::StateWrite(_))
                }),
                GraphError::InvalidTargets("effect target must be a state write"),
            )?;
            needed.insert(effect);
        }
        let mut live = Vec::new();
        for (op, _) in state.ops.iter().rev() {
            if op.tops().iter().any(|tensor| needed.contains(tensor)) {
                needed.extend(op.bottoms());
                live.push(op.clone());
            }
        }
        live.reverse();
        let mut constants: Vec<_> = state.constants.iter().collect();
        constants.sort_by_key(|(id, _)| **id);
        let mut ops = Vec::new();
        for (&id, (data, _)) in constants {
            let top = state.tensors[id];
            if needed.contains(&top) {
                ops.push(Op::Constant(ConstantOp {
                    top,
                    data: data.clone(),
                }));
            }
        }
        ops.extend(live);
        let mut deduplicated: Vec<Op> = Vec::with_capacity(ops.len());
        let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
        let mut replacements = HashMap::new();
        for op in ops {
            let fingerprint = match &op {
                Op::Constant(constant) if !outputs.contains(&constant.top) => {
                    Some(fingerprint([constant.data.bytes()]))
                }
                Op::Builtin(builtin)
                    if builtin.inputs.is_empty()
                        && builtin.outputs.len() == 1
                        && !outputs.contains(&builtin.outputs[0]) =>
                {
                    Some(fingerprint(
                        builtin.blobs.iter().map(|blob| blob.data.bytes()),
                    ))
                }
                _ => None,
            };
            if let Some(fingerprint) = fingerprint {
                let bucket = buckets.entry(fingerprint).or_default();
                if let Some(&original) = bucket.iter().find(|&&i| deduplicated[i].same_value(&op)) {
                    replacements.insert(op.tops()[0], deduplicated[original].tops()[0]);
                    continue;
                }
                bucket.push(deduplicated.len());
            }
            deduplicated.push(op);
        }
        for op in &mut deduplicated {
            op.substitute(&replacements);
        }
        let ops = deduplicated;
        let inputs: Vec<_> = state
            .inputs
            .iter()
            .filter(|(tensor, _)| needed.contains(tensor))
            .copied()
            .collect();
        Ok(LoweredGraph { ops, inputs })
    }

    fn checked_program(
        &self,
        outputs: &[Tensor],
        output_types: &[DataType],
        effects: &[Tensor],
    ) -> Result<Program, GraphError> {
        let state = self.state.borrow();
        let LoweredGraph { ops: live, inputs } = self.lower(outputs, output_types, effects)?;
        let program = Program::new(&live, &inputs, outputs, output_types)?;
        if program.inputs().is_empty() || program.outputs().is_empty() {
            return Err(GraphError::UnsupportedComposition(
                "ANE programs require a live input and an observable output; include the updated state as an output",
            ));
        }

        for op in &live {
            let Op::Builtin(floor) = op else {
                continue;
            };
            if floor.operation != Operator::Floor {
                continue;
            }
            let users: Vec<_> = live
                .iter()
                .filter(|op| op.bottoms().contains(&floor.outputs[0]))
                .collect();
            if let [Op::Builtin(user)] = &users[..] {
                let scaled = user.attributes.iter().any(|(key, value)| {
                    (user.operation == Operator::Mul
                        && *key == Parameter::Y
                        && *value != Value::Fp16(1.0))
                        || (user.operation == Operator::LinearActivation
                            && *key == Parameter::Alpha
                            && *value != Value::Fp32(1.0))
                });
                if scaled {
                    return Err(GraphError::UnsupportedComposition(
                        "single-use floor followed by scalar scaling",
                    ));
                }
            }
        }

        if output_types.contains(&DataType::Int32) {
            return Err(GraphError::UnsupportedComposition(
                "Int32 is only supported for state position inputs",
            ));
        }
        for output in outputs {
            if !matches!(output.data_type(), DataType::Int16 | DataType::UInt16) {
                continue;
            }
            let exact = state.ops.iter().any(|(op, _)| {
                let Op::Builtin(op) = op else {
                    return false;
                };
                op.outputs.contains(output)
                    && (op.operation == Operator::Topk
                        || (op.operation == Operator::Cast
                            && op.inputs.iter().all(|(_, source)| {
                                matches!(source.data_type(), DataType::Int8 | DataType::UInt8)
                            })))
            });
            if !exact {
                return Err(GraphError::UnsupportedComposition(
                    "16-bit integer outputs require ranking or an explicit cast from 8-bit data; ANE passthrough can round large integers",
                ));
            }
        }
        for &(input, dtype) in &inputs {
            if dtype == DataType::Int32
                && (input.physical_shape().iter().product::<usize>() != 1
                    || live.iter().any(|op| {
                        op.bottoms().contains(&input)
                            && !matches!(op, Op::StateUpdate(u) if u.position == input)
                            && !matches!(op, Op::Builtin(u) if u.operation == Operator::DynamicSlice && u.inputs.contains(&(Parameter::Begin, input)))
                    }))
            {
                return Err(GraphError::UnsupportedComposition(
                    "Int32 is only supported for scalar state and slice positions",
                ));
            }
        }

        Ok(program)
    }

    fn bind_executable(
        &self,
        executable: Executable,
        outputs: &[Tensor],
    ) -> Result<Executable, Error> {
        let state = self.state.borrow();
        let inputs: Vec<_> = executable
            .inputs()
            .iter()
            .map(|spec| {
                state
                    .inputs
                    .iter()
                    .find(|(t, _)| t.symbol() == spec.name())
                    .unwrap()
                    .0
            })
            .collect();
        let input_shapes: Vec<_> = inputs.iter().map(Tensor::shape).collect();
        let output_shapes: Vec<_> = outputs.iter().map(Tensor::shape).collect();
        let executable = executable.with_tensor_shapes(&input_shapes, &output_shapes)?;
        let variables = inputs
            .iter()
            .zip(executable.inputs())
            .filter_map(|(tensor, spec)| {
                state
                    .variables
                    .get(tensor)
                    .map(|data| data.initialize(spec).map(|data| (*tensor, data)))
            })
            .collect::<Result<HashMap<_, _>, Error>>()?;
        executable.with_graph_tensors(&inputs, outputs, variables)
    }
}

fn fingerprint<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    let mut hasher = DefaultHasher::new();
    for part in parts {
        part.hash(&mut hasher);
    }
    hasher.finish()
}
