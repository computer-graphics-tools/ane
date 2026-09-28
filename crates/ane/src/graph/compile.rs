use std::collections::HashSet;

use objc2_foundation::NSQualityOfService;

use crate::{ConstantOp, Op};

use super::Graph;
use super::Tensor;
use crate::{DataType, MilProgram};

pub const MIN_SPATIAL_WIDTH: usize = 64;

impl Graph {
    pub fn compile(
        &self,
        quality_of_service: NSQualityOfService,
    ) -> Result<crate::Executable, crate::Error> {
        self.compile_with_output_type(quality_of_service, DataType::Float32)
    }

    pub fn mil_program(&self, output_type: DataType) -> MilProgram {
        self.mil_program_with_output_types(&[output_type])
    }

    pub fn mil_program_with_output_types(&self, output_types: &[DataType]) -> MilProgram {
        let mut shapes: Vec<(String, [usize; 4])> = self
            .inputs
            .iter()
            .map(|(t, _)| (Self::tensor_name(*t), t.shape))
            .collect();

        let regular_ops: Vec<Op> = self
            .ops
            .iter()
            .map(|(op, output)| {
                shapes.push((Self::tensor_name(*output), output.shape));
                op.clone()
            })
            .collect();

        let all_bottoms: HashSet<&str> = regular_ops
            .iter()
            .flat_map(|op| op.bottom_names())
            .collect();

        let mut all_ops: Vec<Op> = Vec::new();
        let mut constants: Vec<_> = self.constants.iter().collect();
        constants.sort_by_key(|(id, _)| **id);
        for (&id, (data, shape)) in constants {
            let name = Self::tensor_name(Tensor { id, shape: *shape });
            if all_bottoms.contains(name.as_str()) {
                shapes.push((name.clone(), *shape));
                all_ops.push(Op::Constant(ConstantOp {
                    name: format!("const_{name}"),
                    top: name,
                    data: data.clone(),
                }));
            }
        }
        all_ops.extend(regular_ops);

        let inputs: Vec<_> = self
            .inputs
            .iter()
            .map(|(t, d)| (Self::tensor_name(*t), t.shape, *d))
            .collect();
        crate::ops::emit_mil(&all_ops, &shapes, &inputs, output_types)
    }

    pub fn compile_with_output_type(
        &self,
        qos: NSQualityOfService,
        output_type: DataType,
    ) -> Result<crate::Executable, crate::Error> {
        self.compile_with_output_types(qos, &[output_type])
    }

    pub fn compile_with_output_types(
        &self,
        qos: NSQualityOfService,
        output_types: &[DataType],
    ) -> Result<crate::Executable, crate::Error> {
        for (op, _) in &self.ops {
            let Op::Elementwise(floor) = op else { continue };
            if floor.operation != crate::ElementwiseOpType::Floor {
                continue;
            }
            let users: Vec<_> = self
                .ops
                .iter()
                .filter(|(op, _)| op.bottom_names().contains(&floor.top.as_str()))
                .collect();
            if users.len() != 1 {
                continue;
            }
            let scales_floor = match &users[0].0 {
                Op::ScalarOp(s) => s.op == crate::ScalarOpType::Mul && s.scalar != 1.0,
                Op::Activation(a) => {
                    matches!(a.mode, crate::ActivationMode::Linear { alpha, .. } if alpha != 1.0)
                }
                _ => false,
            };
            if scales_floor {
                return Err(crate::Error::UnsupportedComposition(
                    "single-use floor followed by scalar scaling; use explicit additions or restructure the graph",
                ));
            }
        }
        let program = self.mil_program_with_output_types(output_types);
        if output_types.contains(&DataType::Int32) {
            return Err(crate::Error::UnsupportedComposition(
                "Int32 is only supported for state position inputs",
            ));
        }
        for (name, shape, dtype) in &program.inputs {
            if *dtype == DataType::Int32 {
                if shape.iter().product::<usize>() != 1
                    || self.ops.iter().any(|(op, _)| {
                        op.bottom_names().contains(&name.as_str())
                            && !matches!(op, Op::StateUpdate(u) if &u.position == name)
                    })
                {
                    return Err(crate::Error::UnsupportedComposition(
                        "Int32 is only supported for scalar state positions",
                    ));
                }
                continue;
            }
            if shape[3] < MIN_SPATIAL_WIDTH {
                return Err(crate::Error::SpatialWidthTooSmall {
                    name: name.clone(),
                    width: shape[3],
                    min: MIN_SPATIAL_WIDTH,
                });
            }
        }
        crate::Executable::compile(program, qos)
    }
}
