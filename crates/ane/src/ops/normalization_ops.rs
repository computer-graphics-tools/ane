use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{ConstantInput, Operator, Parameter, Value};

impl Graph {
    /// Normalizes over `axes` to zero mean and unit variance, then applies the optional constant `gamma`
    /// and `beta`, shaped like the normalized axes. MIL `layer_norm`.
    pub fn layer_norm(
        &self,
        x: &Tensor,
        axes: &[i64],
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
    ) -> Result<Tensor, GraphError> {
        let mut axes = axes
            .iter()
            .map(|&axis| self.axis(*x, axis))
            .collect::<Result<Vec<_>, _>>()?;
        axes.sort_unstable();
        axes.dedup();
        ensure(
            !axes.is_empty(),
            GraphError::InvalidAxes("layer normalization requires axes"),
        )?;
        let normalized: Vec<_> = axes.iter().map(|&axis| x.physical_shape()[axis]).collect();
        let constants = self.affine_constants(gamma, beta, &normalized)?;
        self.normalization(
            *x,
            Operator::LayerNorm,
            &[
                (Parameter::Axes, Value::int32_list(&axes)),
                (Parameter::Epsilon, Value::Fp16(epsilon)),
            ],
            constants,
        )
    }

    /// Normalizes every channel over its spatial axes, then applies the optional constant per-channel
    /// `gamma` and `beta`. The input is `[N, C, H, W]` or `[C, H, W]`. MIL `instance_norm`.
    pub fn instance_norm(
        &self,
        x: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
    ) -> Result<Tensor, GraphError> {
        ensure(
            x.rank() >= 3,
            GraphError::ShapeMismatch("instance normalization requires [N, C, H, W] or [C, H, W]"),
        )?;
        let constants = self.affine_constants(gamma, beta, &[x.physical_shape()[1]])?;
        self.normalization(
            *x,
            Operator::InstanceNorm,
            &[(Parameter::Epsilon, Value::Fp16(epsilon))],
            constants,
        )
    }

    /// Inference batch normalization with constant per-channel `mean` and `variance` and the optional
    /// constant `gamma` and `beta`. MIL `batch_norm`.
    pub fn batch_norm(
        &self,
        x: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
    ) -> Result<Tensor, GraphError> {
        ensure(
            x.rank() >= 3,
            GraphError::ShapeMismatch("batch normalization requires [N, C, H, W] or [C, H, W]"),
        )?;
        let channels = [x.physical_shape()[1]];
        let mut constants = vec![
            self.constant_input(*mean, Parameter::Mean, &channels)?,
            self.constant_input(*variance, Parameter::Variance, &channels)?,
        ];
        constants.extend(self.affine_constants(gamma, beta, &channels)?);
        self.normalization(
            *x,
            Operator::BatchNorm,
            &[(Parameter::Epsilon, Value::Fp16(epsilon))],
            constants,
        )
    }

    /// Divides every batch element by `sqrt(max(Σx², epsilon))` over its last three axes. MIL `l2_norm`.
    pub fn l2_normalize(&self, x: &Tensor, epsilon: f32) -> Result<Tensor, GraphError> {
        self.normalization(
            *x,
            Operator::L2Norm,
            &[(Parameter::Epsilon, Value::Fp16(epsilon))],
            Vec::new(),
        )
    }

    /// Local response normalization across `size` neighbouring channels. MIL `local_response_norm`.
    pub fn local_response_norm(
        &self,
        x: &Tensor,
        size: usize,
        alpha: f32,
        beta: f32,
        k: f32,
    ) -> Result<Tensor, GraphError> {
        ensure(
            size > 0 && size <= x.physical_shape()[1],
            GraphError::InvalidArgument("local response size must cover 1 to C channels"),
        )?;
        self.normalization(
            *x,
            Operator::LocalResponseNorm,
            &[
                (Parameter::Size, Value::Int32(size)),
                (Parameter::Alpha, Value::Fp16(alpha)),
                (Parameter::Beta, Value::Fp16(beta)),
                (Parameter::K, Value::Fp16(k)),
            ],
            Vec::new(),
        )
    }

    fn affine_constants(
        &self,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        shape: &[usize],
    ) -> Result<Vec<ConstantInput>, GraphError> {
        [(gamma, Parameter::Gamma), (beta, Parameter::Beta)]
            .into_iter()
            .filter_map(|(tensor, name)| {
                tensor.map(|tensor| self.constant_input(*tensor, name, shape))
            })
            .collect()
    }

    fn normalization(
        &self,
        x: Tensor,
        operation: Operator,
        attributes: &[(Parameter, Value)],
        constants: Vec<ConstantInput>,
    ) -> Result<Tensor, GraphError> {
        self.numeric(x)?;
        self.builtin_with_constants(
            operation,
            &[(Parameter::X, x)],
            attributes,
            constants,
            x.shape(),
            DataType::Float16,
        )
    }
}
