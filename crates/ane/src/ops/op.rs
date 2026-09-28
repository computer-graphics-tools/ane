use crate::ops::activation::ActivationOp;
use crate::ops::concat::ConcatOp;
use crate::ops::constant::ConstantOp;
use crate::ops::conv::ConvOp;
use crate::ops::deconv::DeconvOp;
use crate::ops::elementwise::ElementwiseOp;
use crate::ops::flatten::FlattenOp;
use crate::ops::inner_product::InnerProductOp;
use crate::ops::instance_norm::InstanceNormOp;
use crate::ops::matmul::MatmulOp;
use crate::ops::padding::PaddingOp;
use crate::ops::pooling::PoolingOp;
use crate::ops::reduction::ReductionOp;
use crate::ops::reshape::ReshapeOp;
use crate::ops::scalar::ScalarOp;
use crate::ops::slice::SliceBySizeOp;
use crate::ops::softmax::SoftmaxOp;
use crate::ops::transpose::TransposeOp;

#[derive(Clone)]
pub enum Op {
    Constant(ConstantOp),
    InnerProduct(InnerProductOp),
    Conv(ConvOp),
    Elementwise(ElementwiseOp),
    Activation(ActivationOp),
    Softmax(SoftmaxOp),
    Concat(ConcatOp),
    Reshape(ReshapeOp),
    InstanceNorm(InstanceNormOp),
    Pooling(PoolingOp),
    Deconv(DeconvOp),
    Padding(PaddingOp),
    Flatten(FlattenOp),
    Reduction(ReductionOp),
    Matmul(MatmulOp),
    Transpose(TransposeOp),
    SliceBySize(SliceBySizeOp),
    ScalarOp(ScalarOp),
    StateUpdate(super::StateUpdateOp),
}

impl Op {
    pub fn name(&self) -> &str {
        match self {
            Self::Constant(operation) => &operation.name,
            Self::InnerProduct(operation) => &operation.name,
            Self::Conv(operation) => &operation.name,
            Self::Elementwise(operation) => &operation.name,
            Self::Activation(operation) => &operation.name,
            Self::Softmax(operation) => &operation.name,
            Self::Concat(operation) => &operation.name,
            Self::Reshape(operation) => &operation.name,
            Self::InstanceNorm(operation) => &operation.name,
            Self::Pooling(operation) => &operation.name,
            Self::Deconv(operation) => &operation.name,
            Self::Padding(operation) => &operation.name,
            Self::Flatten(operation) => &operation.name,
            Self::Reduction(operation) => &operation.name,
            Self::Matmul(operation) => &operation.name,
            Self::Transpose(operation) => &operation.name,
            Self::SliceBySize(operation) => &operation.name,
            Self::ScalarOp(operation) => &operation.name,
            Self::StateUpdate(operation) => &operation.name,
        }
    }

    pub fn top(&self) -> &str {
        match self {
            Self::Constant(operation) => &operation.top,
            Self::InnerProduct(operation) => &operation.top,
            Self::Conv(operation) => &operation.top,
            Self::Elementwise(operation) => &operation.top,
            Self::Activation(operation) => &operation.top,
            Self::Softmax(operation) => &operation.top,
            Self::Concat(operation) => &operation.top,
            Self::Reshape(operation) => &operation.top,
            Self::InstanceNorm(operation) => &operation.top,
            Self::Pooling(operation) => &operation.top,
            Self::Deconv(operation) => &operation.top,
            Self::Padding(operation) => &operation.top,
            Self::Flatten(operation) => &operation.top,
            Self::Reduction(operation) => &operation.top,
            Self::Matmul(operation) => &operation.top,
            Self::Transpose(operation) => &operation.top,
            Self::SliceBySize(operation) => &operation.top,
            Self::ScalarOp(operation) => &operation.top,
            Self::StateUpdate(operation) => &operation.top,
        }
    }

    pub fn bottom_names(&self) -> Vec<&str> {
        match self {
            Self::Constant(_) => vec![],
            Self::Concat(l) => l.bottoms.iter().map(|string| string.as_str()).collect(),
            Self::Elementwise(l) => l.bottoms.iter().map(|string| string.as_str()).collect(),
            Self::Matmul(l) => vec![l.bottom_x.as_str(), l.bottom_y.as_str()],
            Self::InnerProduct(l) => vec![l.bottom.as_str()],
            Self::Conv(l) => vec![l.bottom.as_str()],
            Self::Deconv(l) => vec![l.bottom.as_str()],
            Self::Activation(l) => vec![l.bottom.as_str()],
            Self::Softmax(l) => vec![l.bottom.as_str()],
            Self::Reshape(l) => vec![l.bottom.as_str()],
            Self::InstanceNorm(l) => vec![l.bottom.as_str()],
            Self::Pooling(l) => vec![l.bottom.as_str()],
            Self::Padding(l) => vec![l.bottom.as_str()],
            Self::Flatten(l) => vec![l.bottom.as_str()],
            Self::Reduction(l) => vec![l.bottom.as_str()],
            Self::Transpose(l) => vec![l.bottom.as_str()],
            Self::SliceBySize(l) => vec![l.bottom.as_str()],
            Self::ScalarOp(l) => vec![l.bottom.as_str()],
            Self::StateUpdate(l) => vec![&l.bottom, &l.state, &l.position],
        }
    }
}
