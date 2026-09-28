#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationMode {
    Relu,
    Tanh,
    LeakyRelu { negative_slope: f64 },
    Sigmoid,
    Elu { alpha: f64 },
    Linear { alpha: f64, beta: f64 },
    SigmoidHard { alpha: f64, beta: f64 },
    SoftPlus,
    SoftSign,
}
