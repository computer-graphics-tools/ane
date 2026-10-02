use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{Operator, Parameter, Value};

impl Graph {
    fn elementwise_binary(
        &self,
        left: Tensor,
        right: Tensor,
        operation: Operator,
    ) -> Result<Tensor, GraphError> {
        self.numeric(left)?;
        self.numeric(right)?;
        let (shape, rank) = self.broadcast(&[left, right])?;
        self.builtin(
            operation,
            &[(Parameter::X, left), (Parameter::Y, right)],
            &[],
            &shape[4 - rank..],
            DataType::Float16,
        )
    }

    pub fn addition(
        &self,
        left_hand_side: &Tensor,
        right_hand_side: &Tensor,
    ) -> Result<Tensor, GraphError> {
        self.elementwise_binary(*left_hand_side, *right_hand_side, Operator::Add)
    }

    pub fn subtraction(
        &self,
        left_hand_side: &Tensor,
        right_hand_side: &Tensor,
    ) -> Result<Tensor, GraphError> {
        self.elementwise_binary(*left_hand_side, *right_hand_side, Operator::Sub)
    }

    pub fn multiplication(
        &self,
        left_hand_side: &Tensor,
        right_hand_side: &Tensor,
    ) -> Result<Tensor, GraphError> {
        self.elementwise_binary(*left_hand_side, *right_hand_side, Operator::Mul)
    }

    pub fn division(
        &self,
        left_hand_side: &Tensor,
        right_hand_side: &Tensor,
    ) -> Result<Tensor, GraphError> {
        self.elementwise_binary(*left_hand_side, *right_hand_side, Operator::RealDiv)
    }

    pub fn power(
        &self,
        left_hand_side: &Tensor,
        right_hand_side: &Tensor,
    ) -> Result<Tensor, GraphError> {
        self.elementwise_binary(*left_hand_side, *right_hand_side, Operator::Pow)
    }

    pub fn maximum(
        &self,
        left_hand_side: &Tensor,
        right_hand_side: &Tensor,
    ) -> Result<Tensor, GraphError> {
        self.elementwise_binary(*left_hand_side, *right_hand_side, Operator::Maximum)
    }

    pub fn minimum(
        &self,
        left_hand_side: &Tensor,
        right_hand_side: &Tensor,
    ) -> Result<Tensor, GraphError> {
        self.elementwise_binary(*left_hand_side, *right_hand_side, Operator::Minimum)
    }

    pub fn absolute(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Abs)
    }

    pub fn square_root(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Sqrt)
    }

    pub fn reciprocal_square_root(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Rsqrt)
    }

    pub fn exponent(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Exp)
    }

    pub fn logarithm(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Log)
    }

    pub fn reciprocal(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Inverse)
    }

    fn scalar(
        &self,
        input: Tensor,
        operation: Operator,
        scalar: f32,
        reverse: bool,
    ) -> Result<Tensor, GraphError> {
        self.numeric(input)?;
        ensure(
            scalar.is_finite(),
            GraphError::InvalidArgument("scalar must be finite"),
        )?;
        let (input_key, scalar_key) = if reverse {
            (Parameter::Y, Parameter::X)
        } else {
            (Parameter::X, Parameter::Y)
        };
        self.builtin(
            operation,
            &[(input_key, input)],
            &[(scalar_key, Value::Fp16(scalar))],
            input.shape(),
            DataType::Float16,
        )
    }

    pub fn multiply_scalar(&self, input: &Tensor, value: f32) -> Result<Tensor, GraphError> {
        self.scalar(*input, Operator::Mul, value, false)
    }

    pub fn add_scalar(&self, input: &Tensor, value: f32) -> Result<Tensor, GraphError> {
        self.scalar(*input, Operator::Add, value, false)
    }

    pub fn reverse_subtract_scalar(
        &self,
        input: &Tensor,
        value: f32,
    ) -> Result<Tensor, GraphError> {
        self.scalar(*input, Operator::Sub, value, true)
    }

    pub fn power_scalar(&self, input: &Tensor, value: f32) -> Result<Tensor, GraphError> {
        self.scalar(*input, Operator::Pow, value, false)
    }

    pub fn minimum_scalar(&self, input: &Tensor, value: f32) -> Result<Tensor, GraphError> {
        self.scalar(*input, Operator::Minimum, value, false)
    }

    pub fn maximum_scalar(&self, input: &Tensor, value: f32) -> Result<Tensor, GraphError> {
        self.scalar(*input, Operator::Maximum, value, false)
    }

    pub fn clamp(&self, input: &Tensor, minimum: f32, maximum: f32) -> Result<Tensor, GraphError> {
        ensure(
            minimum <= maximum,
            GraphError::InvalidArgument("invalid clipping interval"),
        )?;
        let lower = self.maximum_scalar(input, minimum)?;
        self.minimum_scalar(&lower, maximum)
    }

    pub fn floor(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Floor)
    }

    pub fn identity(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        Ok(*input)
    }

    pub fn ceil(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Ceil)
    }

    pub fn round(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Round)
    }

    pub fn sign(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Sign)
    }

    pub fn square(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.multiplication(input, input)
    }

    pub fn negative(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.multiply_scalar(input, -1.0)
    }

    pub fn erf(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Erf)
    }

    pub fn exponent_base2(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Exp2)
    }

    pub fn sin(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Sin)
    }

    pub fn cos(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Cos)
    }

    pub fn atan(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        self.unary(*input, Operator::Atan)
    }

    pub fn floor_divide(&self, left: &Tensor, right: &Tensor) -> Result<Tensor, GraphError> {
        let quotient = self.division(left, right)?;
        self.floor(&quotient)
    }

    pub fn truncate(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        let absolute = self.absolute(input)?;
        let floor = self.floor(&absolute)?;
        let sign = self.sign(input)?;
        self.multiplication(&sign, &floor)
    }

    pub fn logarithm_base2(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        let log = self.logarithm(input)?;
        self.multiply_scalar(&log, std::f32::consts::LOG2_E)
    }

    pub fn logarithm_base10(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        let log = self.logarithm(input)?;
        self.multiply_scalar(&log, std::f32::consts::LOG10_E)
    }

    pub fn exponent_base10(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        let scaled = self.multiply_scalar(input, std::f32::consts::LN_10)?;
        self.exponent(&scaled)
    }

    pub fn tan(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        let sin = self.sin(input)?;
        let cos = self.cos(input)?;
        self.division(&sin, &cos)
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

    fn compare(&self, x: Tensor, y: Tensor, op: Operator) -> Result<Tensor, GraphError> {
        let (shape, rank) = self.broadcast(&[x, y])?;
        ensure(
            x.data_type() == y.data_type(),
            GraphError::UnsupportedDataType("comparison types differ"),
        )?;
        ensure(
            !matches!(
                x.data_type(),
                DataType::Int16 | DataType::UInt16 | DataType::Int32
            ),
            GraphError::UnsupportedDataType(
                "integer comparisons wider than 8 bits cannot guarantee exact ANE values",
            ),
        )?;
        let (x, y) = if x.data_type() == DataType::Bool {
            (
                self.cast(&x, DataType::Float16)?,
                self.cast(&y, DataType::Float16)?,
            )
        } else {
            (x, y)
        };
        self.builtin(
            op,
            &[(Parameter::X, x), (Parameter::Y, y)],
            &[],
            &shape[4 - rank..],
            DataType::Bool,
        )
    }

    pub fn equal(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        self.compare(*x, *y, Operator::Equal)
    }

    pub fn not_equal(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        self.compare(*x, *y, Operator::NotEqual)
    }

    pub fn less_than(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        self.compare(*x, *y, Operator::Less)
    }

    pub fn less_than_or_equal_to(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        self.compare(*x, *y, Operator::LessEqual)
    }

    pub fn greater_than(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        self.compare(*x, *y, Operator::Greater)
    }

    pub fn greater_than_or_equal_to(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        self.compare(*x, *y, Operator::GreaterEqual)
    }

    pub fn not(&self, x: &Tensor) -> Result<Tensor, GraphError> {
        ensure(
            x.data_type() == DataType::Bool,
            GraphError::UnsupportedDataType("logical operation requires Boolean input"),
        )?;
        self.builtin(
            Operator::LogicalNot,
            &[(Parameter::X, *x)],
            &[],
            x.shape(),
            DataType::Bool,
        )
    }

    pub fn select(
        &self,
        condition: &Tensor,
        yes: &Tensor,
        no: &Tensor,
    ) -> Result<Tensor, GraphError> {
        let yes = *yes;
        let no = *no;
        let (shape, rank) = self.broadcast(&[*condition, yes, no])?;
        ensure(
            condition.data_type() == DataType::Bool && yes.data_type() == no.data_type(),
            GraphError::UnsupportedDataType(
                "select requires a Boolean condition and matching value types",
            ),
        )?;
        ensure(
            !matches!(
                yes.data_type(),
                DataType::Int16 | DataType::UInt16 | DataType::Int32
            ),
            GraphError::UnsupportedDataType(
                "integer selection wider than 8 bits cannot guarantee exact ANE values",
            ),
        )?;
        if yes.data_type() == DataType::Bool {
            let yes = self.cast(&yes, DataType::Float16)?;
            let no = self.cast(&no, DataType::Float16)?;
            let selected = self.select(condition, &yes, &no)?;
            return self.cast(&selected, DataType::Bool);
        }
        self.builtin(
            Operator::Select,
            &[
                (Parameter::Cond, *condition),
                (Parameter::A, yes),
                (Parameter::B, no),
            ],
            &[],
            &shape[4 - rank..],
            yes.data_type(),
        )
    }

    fn logical(&self, x: Tensor, y: Tensor, op: Operator) -> Result<Tensor, GraphError> {
        ensure(
            x.data_type() == DataType::Bool && y.data_type() == DataType::Bool,
            GraphError::UnsupportedDataType("logical operation requires Boolean inputs"),
        )?;
        let (shape, rank) = self.broadcast(&[x, y])?;
        self.builtin(
            op,
            &[(Parameter::X, x), (Parameter::Y, y)],
            &[],
            &shape[4 - rank..],
            DataType::Bool,
        )
    }

    pub fn logical_and(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        self.logical(*x, *y, Operator::LogicalAnd)
    }

    pub fn logical_or(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        self.logical(*x, *y, Operator::LogicalOr)
    }

    pub fn logical_xor(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, GraphError> {
        ensure(
            x.data_type() == DataType::Bool && y.data_type() == DataType::Bool,
            GraphError::UnsupportedDataType("logical operation requires Boolean inputs"),
        )?;
        self.not_equal(x, y)
    }
}
