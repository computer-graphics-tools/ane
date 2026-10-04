use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{BuiltinOp, Operator, Parameter, Value};

impl Graph {
    /// Matrix product of the last two axes with optional transposes, broadcasting the batch axes. Both
    /// operands have rank 2 or more. MIL `matmul`.
    pub fn matrix_multiplication(
        &self,
        x: &Tensor,
        y: &Tensor,
        transpose_x: bool,
        transpose_y: bool,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*x)?;
        self.numeric(*y)?;
        ensure(
            x.rank() >= 2 && y.rank() >= 2,
            GraphError::ShapeMismatch("matmul operands require rank 2 or higher"),
        )?;
        let shape = BuiltinOp::matmul_shape(*x, *y, transpose_x, transpose_y)?;
        let rank = x.rank().max(y.rank());
        self.builtin(
            Operator::Matmul,
            &[(Parameter::X, *x), (Parameter::Y, *y)],
            &[
                (Parameter::TransposeX, Value::Bool(transpose_x)),
                (Parameter::TransposeY, Value::Bool(transpose_y)),
            ],
            &shape[4 - rank..],
            DataType::Float16,
        )
    }

    /// `x · weightᵀ + bias` with a constant `weight` of shape `[N, K]` and an optional constant `bias`
    /// of shape `[N]`. MIL `linear`.
    pub fn linear(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*x)?;
        let [.., outputs, inputs] = weight.physical_shape();
        ensure(
            weight.rank() == 2 && x.physical_shape()[3] == inputs,
            GraphError::ShapeMismatch("linear requires x [..., K] and weight [N, K]"),
        )?;
        let mut constants =
            vec![self.constant_input(*weight, Parameter::Weight, &[outputs, inputs])?];
        if let Some(bias) = bias {
            constants.push(self.constant_input(*bias, Parameter::Bias, &[outputs])?);
        }
        let mut shape = x.physical_shape();
        shape[3] = outputs;
        self.builtin_with_constants(
            Operator::Linear,
            &[(Parameter::X, *x)],
            &[],
            constants,
            &shape[4 - x.rank()..],
            DataType::Float16,
        )
    }

    /// `softmax(query · keyᵀ / sqrt(E) + attn_mask) · value`. The ANE needs as many key and value heads
    /// as query heads and a Float16 additive mask; it rejects broadcast heads and Boolean masks.
    /// Compilation rejects Float32 inputs with Float16 outputs, which the ANE stores as Float32 data.
    /// MIL `scaled_dot_product_attention`.
    pub fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor, GraphError> {
        for tensor in [query, key, value] {
            self.numeric(*tensor)?;
        }
        let (q, k, v) = (
            query.physical_shape(),
            key.physical_shape(),
            value.physical_shape(),
        );
        ensure(
            query.rank() >= 2 && q[3] == k[3] && k[2] == v[2] && k[..2] == v[..2],
            GraphError::ShapeMismatch(
                "attention requires query [..., L, E], key [..., S, E] and value [..., S, Ev]",
            ),
        )?;
        ensure(
            (0..2).all(|axis| q[axis] == k[axis] || k[axis] == 1),
            GraphError::ShapeMismatch("attention key batch dimensions must match or be 1"),
        )?;
        let shape = [q[0], q[1], q[2], v[3]];
        let mut inputs = vec![
            (Parameter::Query, *query),
            (Parameter::Key, *key),
            (Parameter::Value, *value),
        ];
        if let Some(mask) = attn_mask {
            self.numeric(*mask)?;
            let scores = [q[0], q[1], q[2], k[2]];
            ensure(
                (0..4).all(|a| {
                    mask.physical_shape()[a] == 1 || mask.physical_shape()[a] == scores[a]
                }),
                GraphError::ShapeMismatch("attention mask cannot broadcast to scores"),
            )?;
            inputs.push((Parameter::AttnMask, *mask));
        }
        self.builtin(
            Operator::ScaledDotProductAttention,
            &inputs,
            &[],
            &shape[4 - query.rank()..],
            DataType::Float16,
        )
    }
}
