#![deny(missing_docs)]

use crate::graph::{GraphError, ensure};
use crate::ir::{ConstantOp, Op, Operator, Parameter, Program, Value};
use std::cell::{RefCell, RefMut};
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::graph::{
    GraphBuilder, GraphState, LoweredGraph, Operation, Tensor, TensorHandle, operation_tensor,
};
use crate::{CompilationDescriptor, DataType, Error, Executable, TensorData};

/// A graph of native ANE operations, built like an `MPSGraph`: every method adds one MIL operation,
/// and Apple's ANE compiler fuses them when the graph is compiled.
pub struct Graph {
    state: RefCell<GraphState>,
}

impl Graph {
    /// An empty graph.
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
    /// Compiles the target tensors and target operations into one ANE program.
    pub fn compile(
        &self,
        target_tensors: &[Tensor],
        target_operations: &[&Operation],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Result<Executable, Error> {
        let mut executables =
            self.compile_shared(&[(target_tensors, target_operations)], descriptor)?;
        Ok(executables.remove(0))
    }

    /// Compiles several target sets into one ANE model. The executables share one program slot,
    /// one copy of every constant and the same variable surfaces.
    pub fn compile_shared(
        &self,
        targets: &[(&[Tensor], &[&Operation])],
        descriptor: Option<&CompilationDescriptor>,
    ) -> Result<Vec<Executable>, Error> {
        ensure(
            !targets.is_empty(),
            GraphError::InvalidTargets("compile at least one target set"),
        )?;
        let default = CompilationDescriptor::default();
        let descriptor = descriptor.unwrap_or(&default);
        let programs = targets
            .iter()
            .enumerate()
            .map(|(index, (tensors, operations))| {
                let effects: Vec<_> = operations
                    .iter()
                    .map(|operation| operation_tensor(operation))
                    .collect();
                let types = match &descriptor.output_types {
                    Some(types) => types.clone(),
                    None => tensors
                        .iter()
                        .map(|t| match t.data_type() {
                            DataType::Float16 => DataType::Float32,
                            dtype => dtype,
                        })
                        .collect(),
                };
                self.checked_program(&format!("f{index}"), tensors, &types, &effects)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Executable::compile(
            &programs,
            &self.state.borrow().states,
            descriptor.quality_of_service,
        )
    }

    /// Compiles the targets, runs them once with `feeds` and returns results in target order.
    pub fn run(
        &self,
        feeds: &[(Tensor, &TensorData)],
        target_tensors: &[Tensor],
        target_operations: &[&Operation],
    ) -> Result<Box<[TensorData]>, Error> {
        let executable = self.compile(target_tensors, target_operations, None)?;
        let inputs = executable
            .feed_tensors()
            .iter()
            .map(|tensor| {
                feeds
                    .iter()
                    .find_map(|(feed, data)| (feed == tensor).then_some(*data))
                    .ok_or(Error::Unbound("input"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        executable.run(&inputs, None, None)
    }

    /// The validated program for the targets, for inspecting its MIL with `Program::mil`.
    pub fn program(
        &self,
        outputs: &[Tensor],
        output_types: &[DataType],
    ) -> Result<Program, GraphError> {
        let LoweredGraph { ops, inputs } = self.lower(outputs, output_types, &[])?;
        Ok(Program::new("main", &ops, &inputs, outputs, output_types)?)
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
        for (&id, data) in constants {
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
        name: &str,
        outputs: &[Tensor],
        output_types: &[DataType],
        effects: &[Tensor],
    ) -> Result<Program, GraphError> {
        let state = self.state.borrow();
        let LoweredGraph { ops: live, inputs } = self.lower(outputs, output_types, effects)?;
        let program = Program::new(name, &live, &inputs, outputs, output_types)?;
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
                let scalar = |tensor: &Tensor| {
                    live.iter().any(|op| {
                        matches!(op, Op::Constant(c) if c.top == *tensor && c.data.element_count() == 1)
                    })
                };
                let scaled = (user.operation == Operator::Mul
                    && user.inputs.iter().any(|(_, tensor)| scalar(tensor)))
                    || (user.operation == Operator::LinearActivation
                        && user.attributes.iter().any(|(key, value)| {
                            *key == Parameter::Alpha && *value != Value::Fp16(1.0)
                        }));
                if scaled {
                    return Err(GraphError::UnsupportedComposition(
                        "single-use floor followed by scalar scaling",
                    ));
                }
            }
        }

        let attention = live.iter().any(|op| {
            matches!(op, Op::Builtin(op) if op.operation == Operator::ScaledDotProductAttention)
        });
        if attention
            && inputs.iter().any(|&(_, dtype)| dtype == DataType::Float32)
            && program
                .outputs()
                .iter()
                .any(|&(_, _, dtype)| dtype == DataType::Float16)
        {
            return Err(GraphError::UnsupportedComposition(
                "native attention with Float32 inputs and Float16 outputs writes Float32 data",
            ));
        }

        let written: HashSet<Tensor> = live
            .iter()
            .filter_map(|op| match op {
                Op::StateWrite(op) => Some(op.state),
                Op::StateUpdate(op) => Some(op.state),
                _ => None,
            })
            .collect();
        if written.len() > 7 {
            return Err(GraphError::UnsupportedComposition(
                "an ANE program loads at most 7 written variables; pack them into fewer variables with channel offsets or split the targets with compile_shared",
            ));
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
                    && (matches!(
                        op.operation,
                        Operator::Topk | Operator::ReduceArgmax | Operator::ReduceArgmin
                    ) || (op.operation == Operator::Cast
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
}

fn fingerprint<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    let mut hasher = DefaultHasher::new();
    for part in parts {
        part.hash(&mut hasher);
    }
    hasher.finish()
}
