use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    fn stabilized_rsqrt(&self, variance: &Tensor, epsilon: f32) -> Result<Tensor, GraphError> {
        ensure(
            epsilon > 0.0
                && half::f16::from_f32(epsilon).is_finite()
                && half::f16::from_f32(epsilon).to_f32() > 0.0,
            GraphError::InvalidArgument(
                "normalization epsilon must be positive and finite in FP16",
            ),
        )?;
        self.numeric(*variance)?;
        self.builtin(
            Operator::Rsqrt,
            &[(Parameter::X, *variance)],
            &[(Parameter::Epsilon, Value::Fp16(epsilon))],
            variance.shape(),
            DataType::Float16,
        )
    }

    pub fn layer_norm(
        &self,
        x: &Tensor,
        axes: &[i64],
        scale: &Tensor,
        bias: Option<&Tensor>,
        epsilon: f32,
    ) -> Result<Tensor, GraphError> {
        let bias = bias.copied();
        let mean = self.mean(x, axes)?;
        let centered = self.subtraction(x, &mean)?;
        let squared = self.square(&centered)?;
        let variance = self.mean(&squared, axes)?;
        let inverse = self.stabilized_rsqrt(&variance, epsilon)?;
        let normalized = self.multiplication(&centered, &inverse)?;
        let scaled = self.multiplication(&normalized, scale)?;
        match bias {
            Some(bias) => self.addition(&scaled, &bias),
            None => Ok(scaled),
        }
    }

    pub fn rms_norm(
        &self,
        x: &Tensor,
        axes: &[i64],
        scale: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor, GraphError> {
        let squared = self.square(x)?;
        let variance = self.mean(&squared, axes)?;
        let inverse = self.stabilized_rsqrt(&variance, epsilon)?;
        let normalized = self.multiplication(x, &inverse)?;
        self.multiplication(&normalized, scale)
    }

    pub fn group_norm(
        &self,
        x: &Tensor,
        groups: usize,
        scale: &Tensor,
        bias: Option<&Tensor>,
        epsilon: f32,
    ) -> Result<Tensor, GraphError> {
        let bias = bias.copied();
        self.numeric(*x)?;
        ensure(
            x.rank() == 4 && groups > 0 && x.physical_shape()[1].is_multiple_of(groups),
            GraphError::ShapeMismatch(
                "group normalization requires NCHW with channels divisible by groups",
            ),
        )?;
        let grouped = self.reshape_to(
            *x,
            &[
                x.physical_shape()[0],
                groups,
                x.physical_shape()[1] / groups,
                x.physical_shape()[2] * x.physical_shape()[3],
            ],
        )?;
        let one = self.constant_scalar(1.0, &[])?;
        let normalized = self.layer_norm(&grouped, &[2, 3], &one, None, epsilon)?;
        let restored = self.reshape_to(normalized, x.shape())?;
        let scaled = self.multiplication(&restored, scale)?;
        match bias {
            Some(bias) => self.addition(&scaled, &bias),
            None => Ok(scaled),
        }
    }

    pub fn batch_norm(
        &self,
        x: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        scale: &Tensor,
        bias: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor, GraphError> {
        let centered = self.subtraction(x, mean)?;
        let inverse = self.stabilized_rsqrt(variance, epsilon)?;
        let normalized = self.multiplication(&centered, &inverse)?;
        let scaled = self.multiplication(&normalized, scale)?;
        self.addition(&scaled, bias)
    }

    pub fn l2_normalize(&self, x: &Tensor, axis: i64, epsilon: f32) -> Result<Tensor, GraphError> {
        let sum = self.reduction_sum_square(x, axis)?;
        let inverse = self.stabilized_rsqrt(&sum, epsilon)?;
        self.multiplication(x, &inverse)
    }

    pub fn local_response_norm(
        &self,
        x: &Tensor,
        size: usize,
        alpha: f32,
        beta: f32,
        k: f32,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*x)?;
        ensure(
            size > 0 && size <= i32::MAX as usize && [alpha, beta, k].iter().all(|v| v.is_finite()),
            GraphError::InvalidArgument("invalid local response normalization parameters"),
        )?;
        self.builtin(
            Operator::LocalResponseNorm,
            &[(Parameter::X, *x)],
            &[
                (Parameter::Size, Value::Int32(size)),
                (Parameter::Alpha, Value::Fp16(alpha)),
                (Parameter::Beta, Value::Fp16(beta)),
                (Parameter::K, Value::Fp16(k)),
            ],
            x.shape(),
            DataType::Float16,
        )
    }

    pub fn instance_norm(
        &self,
        input: &Tensor,
        scale: &Tensor,
        bias: Option<&Tensor>,
        epsilon: f32,
    ) -> Result<Tensor, GraphError> {
        ensure(
            input.rank() >= 3,
            GraphError::ShapeMismatch(
                "instance normalization requires channel and spatial dimensions",
            ),
        )?;
        self.layer_norm(
            input,
            &(2..input.rank())
                .map(|axis| axis as i64)
                .collect::<Vec<_>>(),
            scale,
            bias,
            epsilon,
        )
    }
}
