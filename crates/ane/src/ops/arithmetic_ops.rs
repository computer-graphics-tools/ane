use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};

macro_rules! unary {
    ($($(#[$doc:meta])* $name:ident => $operator:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&self, x: &Tensor) -> Result<Tensor, GraphError> {
                self.unary(*x, Operator::$operator, &[])
            }
        )*
    };
}

macro_rules! binary {
    ($($(#[$doc:meta])* $name:ident => $operator:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
                self.binary(*x, *y, Operator::$operator)
            }
        )*
    };
}

macro_rules! comparison {
    ($($(#[$doc:meta])* $name:ident => $operator:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
                self.comparison(*x, *y, Operator::$operator)
            }
        )*
    };
}

impl Graph {
    unary!(
        /// Elementwise `|x|`. MIL `abs`.
        absolute => Abs,
        /// Elementwise arctangent. MIL `atan`; the ANE result is off by up to 3e-2 for `|x| > 2`.
        atan => Atan,
        /// Elementwise ceiling. MIL `ceil`.
        ceil => Ceil,
        /// Elementwise cosine. MIL `cos`.
        cos => Cos,
        /// Elementwise error function. MIL `erf`.
        erf => Erf,
        /// Elementwise `e^x`. MIL `exp`.
        exponent => Exp,
        /// Elementwise `2^x`. MIL `exp2`.
        exponent_base2 => Exp2,
        /// Elementwise floor. MIL `floor`. Compilation rejects a single-use floor followed by a
        /// scalar multiplication, which the ANE miscompiles.
        floor => Floor,
        /// Elementwise rounding with ties away from zero. MIL `round`.
        round => Round,
        /// Elementwise sign: -1, 0 or 1. MIL `sign`.
        sign => Sign,
        /// Elementwise sine. MIL `sin`.
        sin => Sin,
        /// Elementwise square root. MIL `sqrt`.
        square_root => Sqrt,
        /// Elementwise `x²`. MIL `square`.
        square => Square,
    );

    binary!(
        /// Broadcasting `x + y`. MIL `add`.
        addition => Add,
        /// Broadcasting `x - y`. MIL `sub`.
        subtraction => Sub,
        /// Broadcasting `x · y`. MIL `mul`.
        multiplication => Mul,
        /// Broadcasting `x / y`. MIL `real_div`.
        division => RealDiv,
        /// Broadcasting `x^y`. MIL `pow`.
        power => Pow,
        /// Broadcasting elementwise maximum. MIL `maximum`.
        maximum => Maximum,
        /// Broadcasting elementwise minimum. MIL `minimum`.
        minimum => Minimum,
    );

    comparison!(
        /// Broadcasting `x == y` as a Boolean tensor of Float16, Int8 or UInt8 operands. MIL `equal`.
        equal => Equal,
        /// Broadcasting `x != y` as a Boolean tensor. MIL `not_equal`.
        not_equal => NotEqual,
        /// Broadcasting `x < y` as a Boolean tensor. MIL `less`.
        less_than => Less,
        /// Broadcasting `x <= y` as a Boolean tensor. MIL `less_equal`.
        less_than_or_equal_to => LessEqual,
        /// Broadcasting `x > y` as a Boolean tensor. MIL `greater`.
        greater_than => Greater,
        /// Broadcasting `x >= y` as a Boolean tensor. MIL `greater_equal`.
        greater_than_or_equal_to => GreaterEqual,
    );

    /// Elementwise `ln(x + epsilon)`. MIL `log`.
    pub fn logarithm(&self, x: &Tensor, epsilon: f32) -> Result<Tensor, GraphError> {
        self.unary(
            *x,
            Operator::Log,
            &[(Parameter::Epsilon, Value::Fp16(epsilon))],
        )
    }

    /// Elementwise `1 / (x + epsilon)`. MIL `inverse`.
    pub fn reciprocal(&self, x: &Tensor, epsilon: f32) -> Result<Tensor, GraphError> {
        self.unary(
            *x,
            Operator::Inverse,
            &[(Parameter::Epsilon, Value::Fp16(epsilon))],
        )
    }

    /// Elementwise `1 / sqrt(x + epsilon)`. MIL `rsqrt`.
    pub fn reciprocal_square_root(&self, x: &Tensor, epsilon: f32) -> Result<Tensor, GraphError> {
        self.unary(
            *x,
            Operator::Rsqrt,
            &[(Parameter::Epsilon, Value::Fp16(epsilon))],
        )
    }

    /// Clamps every element to `[alpha, beta]`. MIL `clip`.
    pub fn clamp(&self, x: &Tensor, alpha: f32, beta: f32) -> Result<Tensor, GraphError> {
        ensure(
            alpha <= beta,
            GraphError::InvalidArgument("clip requires alpha <= beta"),
        )?;
        self.unary(
            *x,
            Operator::Clip,
            &[
                (Parameter::Alpha, Value::Fp16(alpha)),
                (Parameter::Beta, Value::Fp16(beta)),
            ],
        )
    }

    /// Elementwise AND of Boolean tensors. MIL `logical_and`.
    pub fn logical_and(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        ensure(
            x.data_type() == DataType::Bool && y.data_type() == DataType::Bool,
            GraphError::UnsupportedDataType("logical_and requires Boolean inputs"),
        )?;
        let (shape, rank) = self.broadcast(&[*x, *y])?;
        self.builtin(
            Operator::LogicalAnd,
            &[(Parameter::X, *x), (Parameter::Y, *y)],
            &[],
            &shape[4 - rank..],
            DataType::Bool,
        )
    }

    /// `a` where `cond` is true and `b` elsewhere, broadcasting all three. MIL `select`.
    pub fn select(&self, cond: &Tensor, a: &Tensor, b: &Tensor) -> Result<Tensor, GraphError> {
        let (shape, rank) = self.broadcast(&[*cond, *a, *b])?;
        ensure(
            cond.data_type() == DataType::Bool
                && a.data_type() == b.data_type()
                && matches!(
                    a.data_type(),
                    DataType::Float16 | DataType::Int8 | DataType::UInt8
                ),
            GraphError::UnsupportedDataType(
                "select requires a Boolean condition and matching Float16, Int8 or UInt8 values",
            ),
        )?;
        self.builtin(
            Operator::Select,
            &[
                (Parameter::Cond, *cond),
                (Parameter::A, *a),
                (Parameter::B, *b),
            ],
            &[],
            &shape[4 - rank..],
            a.data_type(),
        )
    }

    fn binary(&self, x: Tensor, y: Tensor, operation: Operator) -> Result<Tensor, GraphError> {
        self.numeric(x)?;
        self.numeric(y)?;
        let (shape, rank) = self.broadcast(&[x, y])?;
        self.builtin(
            operation,
            &[(Parameter::X, x), (Parameter::Y, y)],
            &[],
            &shape[4 - rank..],
            DataType::Float16,
        )
    }

    fn comparison(&self, x: Tensor, y: Tensor, operation: Operator) -> Result<Tensor, GraphError> {
        let (shape, rank) = self.broadcast(&[x, y])?;
        ensure(
            x.data_type() == y.data_type()
                && matches!(
                    x.data_type(),
                    DataType::Float16 | DataType::Int8 | DataType::UInt8
                ),
            GraphError::UnsupportedDataType(
                "comparisons require matching Float16, Int8 or UInt8 operands",
            ),
        )?;
        self.builtin(
            operation,
            &[(Parameter::X, x), (Parameter::Y, y)],
            &[],
            &shape[4 - rank..],
            DataType::Bool,
        )
    }

    fn broadcast(&self, inputs: &[Tensor]) -> Result<([usize; 4], usize), GraphError> {
        let mut shape = [1; 4];
        let mut rank = 0;
        for &input in inputs {
            self.check_tensor(input)?;
            rank = rank.max(input.rank());
            for (i, value) in shape.iter_mut().enumerate() {
                ensure(
                    *value == 1
                        || input.physical_shape()[i] == 1
                        || *value == input.physical_shape()[i],
                    GraphError::ShapeMismatch("incompatible broadcast shapes"),
                )?;
                *value = (*value).max(input.physical_shape()[i]);
            }
        }
        Ok((shape, rank))
    }
}
