use crate::graph::TensorHandle;
use crate::ir::{GraphCompiler, IrError, Op, validate_program};
use crate::{DataType, Tensor};

#[derive(Clone, PartialEq)]
pub struct Program {
    ops: Box<[Op]>,
    feed_tensors: Box<[Tensor]>,
    target_tensors: Box<[Tensor]>,
    inputs: Box<[(String, [usize; 4], DataType)]>,
    outputs: Box<[(String, [usize; 4], DataType)]>,
}

impl Program {
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }
    pub fn feed_tensors(&self) -> &[Tensor] {
        &self.feed_tensors
    }
    pub fn target_tensors(&self) -> &[Tensor] {
        &self.target_tensors
    }
    pub fn inputs(&self) -> &[(String, [usize; 4], DataType)] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[(String, [usize; 4], DataType)] {
        &self.outputs
    }
    pub fn new(
        ops: &[Op],
        inputs: &[(Tensor, DataType)],
        outputs: &[Tensor],
        output_types: &[DataType],
    ) -> Result<Self, IrError> {
        validate_program(ops, inputs, outputs, output_types)?;
        let normalize = |t: &mut Tensor| {
            *t = Tensor::new(t.id(), 0, t.physical_shape(), t.rank(), t.data_type())
        };
        let mut ops = ops.to_vec();
        for op in &mut ops {
            match op {
                Op::Constant(op) => normalize(&mut op.top),
                Op::StateRead(op) => {
                    normalize(&mut op.top);
                    normalize(&mut op.state);
                }
                Op::StateWrite(op) => {
                    normalize(&mut op.top);
                    normalize(&mut op.state);
                    normalize(&mut op.bottom);
                    if let Some(t) = &mut op.previous {
                        normalize(t);
                    }
                }
                Op::StateUpdate(op) => {
                    normalize(&mut op.top);
                    normalize(&mut op.state);
                    normalize(&mut op.bottom);
                    normalize(&mut op.position);
                    if let Some(t) = &mut op.previous {
                        normalize(t);
                    }
                }
                Op::Builtin(op) => {
                    for (_, t) in &mut op.inputs {
                        normalize(t);
                    }
                    for t in &mut op.outputs {
                        normalize(t);
                    }
                }
            }
        }
        let tensors = |values: &[Tensor]| {
            values
                .iter()
                .copied()
                .map(|mut t| {
                    normalize(&mut t);
                    t
                })
                .collect()
        };
        Ok(Self {
            ops: ops.into(),
            feed_tensors: tensors(&inputs.iter().map(|(t, _)| *t).collect::<Vec<_>>()),
            target_tensors: tensors(outputs),
            inputs: inputs
                .iter()
                .map(|(t, d)| (t.symbol(), t.physical_shape(), *d))
                .collect(),
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    (
                        t.symbol(),
                        t.physical_shape(),
                        output_types[i.min(output_types.len() - 1)],
                    )
                })
                .collect(),
        })
    }
    pub fn constant_bytes(&self) -> usize {
        self.ops
            .iter()
            .map(|op| match op {
                Op::Constant(op) => op.data.bytes().len(),
                Op::Builtin(op) => op.blobs.iter().map(|b| b.data.bytes().len()).sum(),
                _ => 0,
            })
            .sum()
    }
    pub fn source_bytes(&self) -> usize {
        self.constant_bytes()
            + self
                .ops
                .iter()
                .map(|op| {
                    std::mem::size_of_val(op)
                        + match op {
                            Op::Builtin(op) => {
                                std::mem::size_of_val(&*op.inputs)
                                    + std::mem::size_of_val(&*op.attributes)
                                    + std::mem::size_of_val(&*op.outputs)
                                    + op.attributes
                                        .iter()
                                        .map(|(_, v)| v.heap_bytes())
                                        .sum::<usize>()
                            }
                            _ => 0,
                        }
                })
                .sum::<usize>()
    }
    pub fn operations(&self) -> Vec<String> {
        self.ops
            .iter()
            .map(|op| {
                match op {
                    Op::Constant(_) => "constant",
                    Op::StateRead(_) => "read_variable",
                    Op::StateWrite(_) => "assign_variable",
                    Op::StateUpdate(_) => "assign_variable_rows",
                    Op::Builtin(op) => op.operation.as_str(),
                }
                .to_string()
            })
            .collect()
    }
    pub fn integer_io_adapters(&self) -> bool {
        use crate::ir::{Operator as O, Parameter as P};
        if self
            .ops
            .iter()
            .any(|op| matches!(op, Op::StateWrite(_) | Op::StateUpdate(_)))
        {
            return false;
        }
        self.target_tensors.iter().all(|&target| {
            let mut current = target;
            loop {
                let Some(Op::Builtin(op)) = self.ops.iter().find(|op| op.tops().contains(&current))
                else {
                    return false;
                };
                match op.operation {
                    O::Topk => return true,
                    O::Cast | O::Reshape => {
                        current = op.inputs.iter().find(|(p, _)| *p == P::X).unwrap().1
                    }
                    _ => return false,
                }
            }
        })
    }
    pub fn mlir(&self) -> Result<String, IrError> {
        objc2::rc::autoreleasepool(|_| self.mlir_source())
    }
    fn mlir_source(&self) -> Result<String, IrError> {
        let (executable, _) = GraphCompiler::new(self)?.executable()?;
        let text = unsafe {
            objc2::rc::Retained::retain_autoreleased(
                raw_message!(&*executable,c"debugDescription"; *mut objc2_foundation::NSString),
            )
        }
        .ok_or(IrError::CompilerFailed)?
        .to_string();
        let (_, text) = text.split_once("IR: ").ok_or(IrError::CompilerFailed)?;
        Ok(text.into())
    }
}
