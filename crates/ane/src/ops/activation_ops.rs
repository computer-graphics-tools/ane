use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    fn activation(
        &self,
        input: Tensor,
        operation: Operator,
        parameters: &[(Parameter, f64)],
    ) -> Result<Tensor, GraphError> {
        self.numeric(input)?;
        let attributes: Vec<_> = parameters
            .iter()
            .map(|&(key, value)| (key, Value::Fp32(value as f32)))
            .collect();
        self.builtin(
            operation,
            &[(Parameter::X, input)],
            &attributes,
            input.shape(),
            DataType::Float16,
        )
    }

    pub fn relu(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Relu)
    }

    pub fn tanh(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Tanh)
    }

    pub fn sigmoid(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Sigmoid)
    }

    pub fn leaky_relu(&self, input: &Tensor, negative_slope: f64) -> Result<Tensor, GraphError> {
        self.activation(
            *input,
            Operator::LeakyRelu,
            &[(Parameter::Alpha, negative_slope)],
        )
    }

    pub fn elu(&self, input: &Tensor, alpha: f64) -> Result<Tensor, GraphError> {
        self.activation(*input, Operator::Elu, &[(Parameter::Alpha, alpha)])
    }

    pub fn hard_sigmoid(
        &self,
        input: &Tensor,
        alpha: f64,
        beta: f64,
    ) -> Result<Tensor, GraphError> {
        let scaled = self.linear_activation(input, alpha, beta)?;
        self.clamp(&scaled, 0.0, 1.0)
    }

    pub fn linear_activation(
        &self,
        input: &Tensor,
        alpha: f64,
        beta: f64,
    ) -> Result<Tensor, GraphError> {
        self.activation(
            *input,
            Operator::LinearActivation,
            &[(Parameter::Alpha, alpha), (Parameter::Beta, beta)],
        )
    }

    pub fn softplus(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Softplus)
    }

    pub fn softsign(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Softsign)
    }

    pub fn soft_max(&self, input: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        let axis = self.axis(*input, axis)?;
        self.builtin(
            Operator::Softmax,
            &[(Parameter::X, *input)],
            &[(Parameter::Axis, Value::Int32(axis))],
            input.shape(),
            DataType::Float16,
        )
    }

    pub fn threshold(&self, input: &Tensor, minimum: f64) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        ensure(
            minimum.is_finite(),
            GraphError::InvalidArgument("threshold must be finite"),
        )?;
        self.builtin(
            Operator::Threshold,
            &[(Parameter::X, *input)],
            &[(Parameter::Alpha, Value::Fp16(minimum as f32))],
            input.shape(),
            DataType::Float16,
        )
    }

    pub fn relu6(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.clamp(input, 0.0, 6.0)
    }

    pub fn silu(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Silu)
    }

    pub fn gelu(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.builtin(
            Operator::Gelu,
            &[(Parameter::X, *input)],
            &[(Parameter::Mode, Value::String("TANH_APPROXIMATION"))],
            input.shape(),
            DataType::Float16,
        )
    }

    pub fn gelu_exact(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.builtin(
            Operator::Gelu,
            &[(Parameter::X, *input)],
            &[(Parameter::Mode, Value::String("EXACT"))],
            input.shape(),
            DataType::Float16,
        )
    }

    pub fn prelu(&self, input: &Tensor, slope: &Tensor) -> Result<Tensor, GraphError> {
        let zero = self.constant_scalar(0.0, &[])?;
        let positive = self.greater_than_or_equal_to(input, &zero)?;
        let negative = self.multiplication(input, slope)?;
        self.select(&positive, input, &negative)
    }

    pub fn clamped_relu(
        &self,
        input: &Tensor,
        alpha: f32,
        beta: f32,
    ) -> Result<Tensor, GraphError> {
        ensure(
            alpha.is_finite() && beta.is_finite(),
            GraphError::InvalidArgument("activation parameters must be finite"),
        )?;
        let leaky = self.leaky_relu(input, alpha as f64)?;
        self.minimum_scalar(&leaky, beta)
    }

    pub fn scaled_tanh(&self, input: &Tensor, alpha: f32, beta: f32) -> Result<Tensor, GraphError> {
        let scaled = self.multiply_scalar(input, beta)?;
        let activation = self.tanh(&scaled)?;
        self.multiply_scalar(&activation, alpha)
    }

    pub fn softplus_parametric(
        &self,
        input: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor, GraphError> {
        let scaled = self.multiplication(input, beta)?;
        let activation = self.softplus(&scaled)?;
        self.multiplication(&activation, alpha)
    }

    pub fn thresholded_relu(&self, input: &Tensor, threshold: f32) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        ensure(
            threshold.is_finite(),
            GraphError::InvalidArgument("threshold must be finite"),
        )?;
        self.builtin(
            Operator::ThresholdedRelu,
            &[(Parameter::X, *input)],
            &[(Parameter::Alpha, Value::Fp16(threshold))],
            input.shape(),
            DataType::Float16,
        )
    }

    pub fn hard_swish(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        let offset = self.add_scalar(input, 3.0)?;
        let clipped = self.clamp(&offset, 0.0, 6.0)?;
        let product = self.multiplication(input, &clipped)?;
        self.multiply_scalar(&product, 1.0 / 6.0)
    }

    pub fn log_softmax(&self, x: &Tensor, axis: i64) -> Result<Tensor, GraphError> {
        let sum = self.reduction_log_sum_exp(x, axis)?;
        self.subtraction(x, &sum)
    }
}
