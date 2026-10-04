use crate::DataType;
use crate::GeluMode;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::GraphError;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::ir::{Operator, Parameter, Value};

macro_rules! activation {
    ($($(#[$doc:meta])* $name:ident => $operator:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&self, x: &Tensor) -> Result<Tensor, GraphError> {
                self.unary(*x, Operator::$operator, &[])
            }
        )*
    };
}

macro_rules! alpha {
    ($($(#[$doc:meta])* $name:ident => $operator:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&self, x: &Tensor, alpha: f32) -> Result<Tensor, GraphError> {
                self.unary(*x, Operator::$operator, &[(Parameter::Alpha, Value::Fp16(alpha))])
            }
        )*
    };
}

macro_rules! alpha_beta {
    ($($(#[$doc:meta])* $name:ident => $operator:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&self, x: &Tensor, alpha: f32, beta: f32) -> Result<Tensor, GraphError> {
                self.unary(
                    *x,
                    Operator::$operator,
                    &[
                        (Parameter::Alpha, Value::Fp16(alpha)),
                        (Parameter::Beta, Value::Fp16(beta)),
                    ],
                )
            }
        )*
    };
}

impl Graph {
    activation!(
        /// Elementwise `max(x, 0)`. MIL `relu`.
        relu => Relu,
        /// Elementwise `min(max(x, 0), 6)`. MIL `relu6`.
        relu6 => Relu6,
        /// Elementwise logistic sigmoid. MIL `sigmoid`; the ANE result is off by up to 3e-3.
        sigmoid => Sigmoid,
        /// Elementwise `x · sigmoid(x)`. MIL `silu`; the ANE result is off by up to 1.5e-2.
        silu => Silu,
        /// Elementwise `ln(1 + e^x)`. MIL `softplus`.
        softplus => Softplus,
        /// Elementwise `x / (1 + |x|)`. MIL `softsign`.
        softsign => Softsign,
        /// Elementwise hyperbolic tangent. MIL `tanh`.
        tanh => Tanh,
    );

    alpha!(
        /// `x` for positive inputs, `alpha · x` otherwise. MIL `leaky_relu`.
        leaky_relu => LeakyRelu,
        /// `x` for positive inputs, `alpha · (e^x - 1)` otherwise. MIL `elu`.
        elu => Elu,
        /// `x` where `x > alpha`, 0 elsewhere. MIL `thresholded_relu`.
        thresholded_relu => ThresholdedRelu,
    );

    alpha_beta!(
        /// `alpha · x + beta`. MIL `linear_activation`.
        linear_activation => LinearActivation,
        /// `clamp(alpha · x + beta, 0, 1)`. MIL `sigmoid_hard`.
        hard_sigmoid => SigmoidHard,
        /// `alpha · tanh(beta · x)`. MIL `scaled_tanh`.
        scaled_tanh => ScaledTanh,
        /// `min(x, beta)` for non-negative inputs, `alpha · x` otherwise. MIL `clamped_relu`.
        clamped_relu => ClampedRelu,
    );

    /// GELU in the given approximation mode. MIL `gelu`; the ANE result is off by up to 6e-3 in every
    /// mode. Composing the formula from other ops is slower but closer.
    pub fn gelu(&self, x: &Tensor, mode: GeluMode) -> Result<Tensor, GraphError> {
        self.unary(
            *x,
            Operator::Gelu,
            &[(Parameter::Mode, Value::String(mode.as_str()))],
        )
    }

    /// Softmax along `axis`. MIL `softmax`.
    pub fn soft_max(&self, x: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        let axis = self.axis(*x, axis)?;
        self.unary(
            *x,
            Operator::Softmax,
            &[(Parameter::Axis, Value::Int32(axis))],
        )
    }

    /// Leaky ReLU with a constant per-channel `alpha` of shape `[C]`. MIL `prelu`.
    pub fn prelu(&self, x: &Tensor, alpha: &Tensor) -> Result<Tensor, GraphError> {
        self.numeric(*x)?;
        let channels = x.physical_shape()[1];
        let alpha = self.constant_input(*alpha, Parameter::Alpha, &[channels])?;
        self.builtin_with_constants(
            Operator::Prelu,
            &[(Parameter::X, *x)],
            &[],
            vec![alpha],
            x.shape(),
            DataType::Float16,
        )
    }

    /// Per-channel `alpha · ln(1 + e^(beta · x))` with constant `[C]` parameters. MIL
    /// `softplus_parametric`.
    pub fn softplus_parametric(
        &self,
        x: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*x)?;
        let channels = x.physical_shape()[1];
        let constants = vec![
            self.constant_input(*alpha, Parameter::Alpha, &[channels])?,
            self.constant_input(*beta, Parameter::Beta, &[channels])?,
        ];
        self.builtin_with_constants(
            Operator::SoftplusParametric,
            &[(Parameter::X, *x)],
            &[],
            constants,
            x.shape(),
            DataType::Float16,
        )
    }
}
