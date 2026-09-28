use super::ElementwiseOpType;

#[derive(Clone)]
pub struct ElementwiseOp {
    pub name: String,
    pub bottoms: Box<[String]>,
    pub top: String,
    pub operation: ElementwiseOpType,
    pub alpha: f64,
    pub beta: f64,
    pub fused_relu: bool,
}

impl ElementwiseOp {
    fn binary(
        name: impl Into<String>,
        bottoms: &[&str],
        top: impl Into<String>,
        operation: ElementwiseOpType,
    ) -> Self {
        Self {
            name: name.into(),
            bottoms: bottoms.iter().map(|s| (*s).to_owned()).collect(),
            top: top.into(),
            operation,
            alpha: 1.0,
            beta: 0.0,
            fused_relu: false,
        }
    }

    fn unary(
        name: impl Into<String>,
        bottom: &str,
        top: impl Into<String>,
        operation: ElementwiseOpType,
    ) -> Self {
        Self {
            name: name.into(),
            bottoms: vec![bottom.to_owned()].into_boxed_slice(),
            top: top.into(),
            operation,
            alpha: 1.0,
            beta: 0.0,
            fused_relu: false,
        }
    }

    pub fn add(name: impl Into<String>, bottoms: &[&str], top: impl Into<String>) -> Self {
        Self::binary(name, bottoms, top, ElementwiseOpType::Add)
    }

    pub fn multiply(name: impl Into<String>, bottoms: &[&str], top: impl Into<String>) -> Self {
        Self::binary(name, bottoms, top, ElementwiseOpType::Multiply)
    }

    pub fn sub(name: impl Into<String>, bottoms: &[&str], top: impl Into<String>) -> Self {
        Self::binary(name, bottoms, top, ElementwiseOpType::Sub)
    }

    pub fn div(name: impl Into<String>, bottoms: &[&str], top: impl Into<String>) -> Self {
        Self::binary(name, bottoms, top, ElementwiseOpType::Div)
    }

    pub fn max(name: impl Into<String>, bottoms: &[&str], top: impl Into<String>) -> Self {
        Self::binary(name, bottoms, top, ElementwiseOpType::Max)
    }

    pub fn min(name: impl Into<String>, bottoms: &[&str], top: impl Into<String>) -> Self {
        Self::binary(name, bottoms, top, ElementwiseOpType::Min)
    }

    pub fn pow(name: impl Into<String>, bottoms: &[&str], top: impl Into<String>) -> Self {
        Self::binary(name, bottoms, top, ElementwiseOpType::Pow)
    }

    pub fn abs(name: impl Into<String>, bottom: &str, top: impl Into<String>) -> Self {
        Self::unary(name, bottom, top, ElementwiseOpType::Abs)
    }

    pub fn sqrt(name: impl Into<String>, bottom: &str, top: impl Into<String>) -> Self {
        Self::unary(name, bottom, top, ElementwiseOpType::Sqrt)
    }

    pub fn rsqrt(name: impl Into<String>, bottom: &str, top: impl Into<String>) -> Self {
        Self::unary(name, bottom, top, ElementwiseOpType::Rsqrt)
    }

    pub fn inverse(name: impl Into<String>, bottom: &str, top: impl Into<String>) -> Self {
        Self::unary(name, bottom, top, ElementwiseOpType::Inverse)
    }

    pub fn exp(name: impl Into<String>, bottom: &str, top: impl Into<String>) -> Self {
        Self::unary(name, bottom, top, ElementwiseOpType::Exp)
    }

    pub fn log(name: impl Into<String>, bottom: &str, top: impl Into<String>) -> Self {
        Self::unary(name, bottom, top, ElementwiseOpType::Log)
    }
}
