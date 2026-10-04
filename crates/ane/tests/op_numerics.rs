use ane::{
    ConvolutionTranspose2dDescriptor, DataType, GeluMode, Graph, GraphError, Pooling2dDescriptor,
    ResizeSamplingMode, Tensor,
};

const SHAPE: [usize; 4] = [1, 2, 8, 16];
const COUNT: usize = 256;

type Build = fn(&Graph, &Tensor) -> Result<Tensor, GraphError>;
type Reference = fn(&[f64]) -> Vec<f64>;

struct Case {
    name: &'static str,
    build: Build,
    reference: Reference,
    domain: (f32, f32),
    tolerance: f64,
}

fn case(name: &'static str, build: Build, reference: Reference) -> Case {
    Case {
        name,
        build,
        reference,
        domain: (-4., 4.),
        tolerance: 4e-3,
    }
}

/// Ops the ANE evaluates with a coarser approximation, at their measured precision.
fn approximate(tolerance: f64, case: Case) -> Case {
    Case { tolerance, ..case }
}

fn operand() -> Vec<f32> {
    (0..COUNT)
        .map(|i| half::f16::from_f32(0.5 + (i * 37 % 64) as f32 / 32.).to_f32())
        .collect()
}

fn signed_operand() -> Vec<f32> {
    operand().iter().map(|v| v - 1.5).collect()
}

fn channels() -> Vec<f32> {
    vec![0.75, -0.5]
}

fn map(x: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> {
    x.iter().map(|&v| f(v)).collect()
}

fn zip(x: &[f64], y: &[f32], f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    x.iter().zip(y).map(|(&a, &b)| f(a, f64::from(b))).collect()
}

fn rows(x: &[f64], width: usize, f: impl Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
    x.chunks(width).flat_map(f).collect()
}

fn erf(x: f64) -> f64 {
    let t = 1. / (1. + 0.5 * x.abs());
    let y = 1.
        - t * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0. { y } else { -y }
}

fn sigmoid(x: f64) -> f64 {
    1. / (1. + (-x).exp())
}

fn compare(g: &Graph, condition: Tensor) -> Result<Tensor, GraphError> {
    let ones = g.constant(&[1.], [1])?;
    let zeros = g.constant(&[0.], [1])?;
    g.select(&condition, &ones, &zeros)
}

fn cases() -> Vec<Case> {
    let mut cases = vec![
        case("absolute", |g, x| g.absolute(x), |x| map(x, f64::abs)),
        approximate(2e-2, case("atan", |g, x| g.atan(x), |x| map(x, f64::atan))),
        case("ceil", |g, x| g.ceil(x), |x| map(x, f64::ceil)),
        case("cos", |g, x| g.cos(x), |x| map(x, f64::cos)),
        case("erf", |g, x| g.erf(x), |x| map(x, erf)),
        case("exponent", |g, x| g.exponent(x), |x| map(x, f64::exp)),
        case(
            "exponent_base2",
            |g, x| g.exponent_base2(x),
            |x| map(x, f64::exp2),
        ),
        case("floor", |g, x| g.floor(x), |x| map(x, f64::floor)),
        case(
            "sign",
            |g, x| g.sign(x),
            |x| map(x, |v| if v == 0. { 0. } else { v.signum() }),
        ),
        case("sin", |g, x| g.sin(x), |x| map(x, f64::sin)),
        case("square", |g, x| g.square(x), |x| map(x, |v| v * v)),
        case(
            "relu6",
            |g, x| g.relu6(&g.multiplication(x, &g.constant(&[2.], [1])?)?),
            |x| map(x, |v| (2. * v).clamp(0., 6.)),
        ),
        case("sigmoid", |g, x| g.sigmoid(x), |x| map(x, sigmoid)),
        case(
            "softplus",
            |g, x| g.softplus(x),
            |x| map(x, |v| v.exp().ln_1p()),
        ),
        case(
            "softsign",
            |g, x| g.softsign(x),
            |x| map(x, |v| v / (1. + v.abs())),
        ),
        case("tanh", |g, x| g.tanh(x), |x| map(x, f64::tanh)),
        approximate(
            1.5e-2,
            case("silu", |g, x| g.silu(x), |x| map(x, |v| v * sigmoid(v))),
        ),
        case(
            "clamp",
            |g, x| g.clamp(x, -1., 2.),
            |x| map(x, |v| v.clamp(-1., 2.)),
        ),
        case(
            "elu",
            |g, x| g.elu(x, 0.5),
            |x| map(x, |v| if v > 0. { v } else { 0.5 * v.exp_m1() }),
        ),
        case(
            "leaky_relu",
            |g, x| g.leaky_relu(x, 0.1),
            |x| map(x, |v| if v > 0. { v } else { 0.1 * v }),
        ),
        case(
            "thresholded_relu",
            |g, x| g.thresholded_relu(x, 1.),
            |x| map(x, |v| if v > 1. { v } else { 0. }),
        ),
        case(
            "linear_activation",
            |g, x| g.linear_activation(x, 1.5, -0.5),
            |x| map(x, |v| 1.5 * v - 0.5),
        ),
        case(
            "hard_sigmoid",
            |g, x| g.hard_sigmoid(x, 0.2, 0.5),
            |x| map(x, |v| (0.2 * v + 0.5).clamp(0., 1.)),
        ),
        case(
            "scaled_tanh",
            |g, x| g.scaled_tanh(x, 1.5, 0.5),
            |x| map(x, |v| 1.5 * (0.5 * v).tanh()),
        ),
        case(
            "clamped_relu",
            |g, x| g.clamped_relu(x, 0.1, 2.),
            |x| map(x, |v| if v >= 0. { v.min(2.) } else { 0.1 * v }),
        ),
        approximate(
            6e-3,
            case(
                "gelu_exact",
                |g, x| g.gelu(x, GeluMode::Exact),
                |x| map(x, |v| 0.5 * v * (1. + erf(v / std::f64::consts::SQRT_2))),
            ),
        ),
        approximate(
            6e-3,
            case(
                "gelu_tanh",
                |g, x| g.gelu(x, GeluMode::TanhApproximation),
                |x| {
                    map(x, |v| {
                        0.5 * v
                            * (1.
                                + ((2. / std::f64::consts::PI).sqrt() * (v + 0.044715 * v.powi(3)))
                                    .tanh())
                    })
                },
            ),
        ),
        approximate(
            6e-3,
            case(
                "gelu_sigmoid",
                |g, x| g.gelu(x, GeluMode::SigmoidApproximation),
                |x| map(x, |v| v * sigmoid(1.702 * v)),
            ),
        ),
        case(
            "addition",
            |g, x| g.addition(x, &g.constant(&signed_operand(), SHAPE)?),
            |x| zip(x, &signed_operand(), |a, b| a + b),
        ),
        case(
            "subtraction",
            |g, x| g.subtraction(x, &g.constant(&signed_operand(), SHAPE)?),
            |x| zip(x, &signed_operand(), |a, b| a - b),
        ),
        case(
            "multiplication",
            |g, x| g.multiplication(x, &g.constant(&signed_operand(), SHAPE)?),
            |x| zip(x, &signed_operand(), |a, b| a * b),
        ),
        case(
            "division",
            |g, x| g.division(x, &g.constant(&operand(), SHAPE)?),
            |x| zip(x, &operand(), |a, b| a / b),
        ),
        case(
            "maximum",
            |g, x| g.maximum(x, &g.constant(&signed_operand(), SHAPE)?),
            |x| zip(x, &signed_operand(), f64::max),
        ),
        case(
            "minimum",
            |g, x| g.minimum(x, &g.constant(&signed_operand(), SHAPE)?),
            |x| zip(x, &signed_operand(), f64::min),
        ),
        case(
            "equal",
            |g, x| compare(g, g.equal(&g.round(x)?, &g.constant(&[1.], [1])?)?),
            |x| map(x, |v| f64::from(u8::from(v.round() == 1.))),
        ),
        case(
            "not_equal",
            |g, x| compare(g, g.not_equal(&g.round(x)?, &g.constant(&[1.], [1])?)?),
            |x| map(x, |v| f64::from(u8::from(v.round() != 1.))),
        ),
        case(
            "less_than",
            |g, x| compare(g, g.less_than(x, &g.constant(&signed_operand(), SHAPE)?)?),
            |x| zip(x, &signed_operand(), |a, b| f64::from(u8::from(a < b))),
        ),
        case(
            "less_than_or_equal_to",
            |g, x| {
                let y = g.constant(&signed_operand(), SHAPE)?;
                compare(g, g.less_than_or_equal_to(x, &y)?)
            },
            |x| zip(x, &signed_operand(), |a, b| f64::from(u8::from(a <= b))),
        ),
        case(
            "greater_than",
            |g, x| {
                compare(
                    g,
                    g.greater_than(x, &g.constant(&signed_operand(), SHAPE)?)?,
                )
            },
            |x| zip(x, &signed_operand(), |a, b| f64::from(u8::from(a > b))),
        ),
        case(
            "greater_than_or_equal_to",
            |g, x| {
                let y = g.constant(&signed_operand(), SHAPE)?;
                compare(g, g.greater_than_or_equal_to(x, &y)?)
            },
            |x| zip(x, &signed_operand(), |a, b| f64::from(u8::from(a >= b))),
        ),
        case(
            "soft_max",
            |g, x| g.soft_max(x, -1),
            |x| {
                rows(x, 16, |row| {
                    let peak = row.iter().copied().fold(f64::MIN, f64::max);
                    let total: f64 = row.iter().map(|v| (v - peak).exp()).sum();
                    row.iter().map(|v| (v - peak).exp() / total).collect()
                })
            },
        ),
        case(
            "reduction_minimum",
            |g, x| g.reduction_minimum(x, &[-1]),
            |x| {
                rows(x, 16, |row| {
                    vec![row.iter().copied().fold(f64::MAX, f64::min)]
                })
            },
        ),
        case(
            "reduction_maximum",
            |g, x| g.reduction_maximum(x, &[-1]),
            |x| {
                rows(x, 16, |row| {
                    vec![row.iter().copied().fold(f64::MIN, f64::max)]
                })
            },
        ),
        case(
            "reduction_l1_norm",
            |g, x| g.reduction_l1_norm(x, &[-1]),
            |x| rows(x, 16, |row| vec![row.iter().map(|v| v.abs()).sum()]),
        ),
        case(
            "reduction_l2_norm",
            |g, x| g.reduction_l2_norm(x, &[-1]),
            |x| {
                rows(x, 16, |row| {
                    vec![row.iter().map(|v| v * v).sum::<f64>().sqrt()]
                })
            },
        ),
        case(
            "reduction_sum_square",
            |g, x| g.reduction_sum_square(x, &[-1]),
            |x| rows(x, 16, |row| vec![row.iter().map(|v| v * v).sum()]),
        ),
        case(
            "reduction_log_sum_exp",
            |g, x| g.reduction_log_sum_exp(x, &[-1]),
            |x| {
                rows(x, 16, |row| {
                    vec![row.iter().map(|v| v.exp()).sum::<f64>().ln()]
                })
            },
        ),
        case(
            "l2_normalize",
            |g, x| g.l2_normalize(x, 1e-6),
            |x| {
                rows(x, COUNT, |row| {
                    let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                    row.iter().map(|v| v / norm).collect()
                })
            },
        ),
        case(
            "instance_norm",
            |g, x| g.instance_norm(x, None, None, 1e-5),
            |x| {
                rows(x, COUNT / 2, |plane| {
                    let mean = plane.iter().sum::<f64>() / plane.len() as f64;
                    let variance =
                        plane.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / plane.len() as f64;
                    plane
                        .iter()
                        .map(|v| (v - mean) / (variance + 1e-5).sqrt())
                        .collect()
                })
            },
        ),
        case(
            "batch_norm",
            |g, x| {
                let mean = g.constant(&[0.5, -0.25], [2])?;
                let variance = g.constant(&[4., 0.25], [2])?;
                g.batch_norm(x, &mean, &variance, None, None, 1e-5)
            },
            |x| {
                x.iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let (mean, variance) = [(0.5, 4.), (-0.25, 0.25)][i / (COUNT / 2)];
                        (v - mean) / (variance + 1e-5_f64).sqrt()
                    })
                    .collect()
            },
        ),
        case(
            "prelu",
            |g, x| g.prelu(x, &g.constant(&channels(), [2])?),
            |x| {
                x.iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let alpha = f64::from(channels()[i / (COUNT / 2)]);
                        if v > 0. { v } else { alpha * v }
                    })
                    .collect()
            },
        ),
        case(
            "softplus_parametric",
            |g, x| {
                let alpha = g.constant(&[1.5, 0.5], [2])?;
                let beta = g.constant(&[0.5, 2.], [2])?;
                g.softplus_parametric(x, &alpha, &beta)
            },
            |x| {
                x.iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let (alpha, beta) = [(1.5, 0.5), (0.5, 2.)][i / (COUNT / 2)];
                        alpha * (beta * v).exp().ln_1p()
                    })
                    .collect()
            },
        ),
        case(
            "reverse",
            |g, x| g.reverse(x, &[-1]),
            |x| rows(x, 16, |row| row.iter().rev().copied().collect()),
        ),
        case(
            "tile",
            |g, x| g.tile(x, &[1, 1, 1, 2]),
            |x| rows(x, 16, |row| row.iter().chain(row).copied().collect()),
        ),
        case(
            "strided_slice",
            |g, x| g.strided_slice(x, &[0, 1, 2, 1], &[1, 2, 8, 16], &[1, 1, 3, 4]),
            |x| {
                let mut out = Vec::new();
                for h in (2..8).step_by(3) {
                    for w in (1..16).step_by(4) {
                        out.push(x[128 + h * 16 + w]);
                    }
                }
                out
            },
        ),
        case(
            "concat_interleave",
            |g, x| g.concat(&[x, &g.constant(&signed_operand(), SHAPE)?], 3, true),
            |x| {
                let y = signed_operand();
                (0..COUNT * 2)
                    .map(|i| {
                        if i % 2 == 0 {
                            x[i / 2]
                        } else {
                            f64::from(y[i / 2])
                        }
                    })
                    .collect()
            },
        ),
        case(
            "depth_to_space",
            |g, x| g.depth_to_space(&g.reshape(x, [1, 8, 4, 8])?, 2),
            |x| {
                (0..COUNT)
                    .map(|i| {
                        let (c, h, w) = (i / 128, i / 16 % 8, i % 16);
                        let channel = (h % 2 * 2 + w % 2) * 2 + c;
                        x[(channel * 4 + h / 2) * 8 + w / 2]
                    })
                    .collect()
            },
        ),
        case(
            "space_to_depth",
            |g, x| g.space_to_depth(x, 2),
            |x| {
                let mut out = vec![0.; COUNT];
                for c in 0..2 {
                    for h in 0..8 {
                        for w in 0..16 {
                            let channel = (h % 2 * 2 + w % 2) * 2 + c;
                            out[(channel * 4 + h / 2) * 8 + w / 2] = x[(c * 8 + h) * 16 + w];
                        }
                    }
                }
                out
            },
        ),
        case(
            "pixel_shuffle",
            |g, x| g.pixel_shuffle(&g.reshape(x, [1, 8, 4, 8])?, 2),
            |x| {
                (0..COUNT)
                    .map(|i| {
                        let (c, h, w) = (i / 128, i / 16 % 8, i % 16);
                        let channel = c * 4 + h % 2 * 2 + w % 2;
                        x[(channel * 4 + h / 2) * 8 + w / 2]
                    })
                    .collect()
            },
        ),
        case(
            "avg_pooling_2d",
            |g, x| g.avg_pooling_2d(x, &Pooling2dDescriptor::new([2, 2], [2, 2])),
            |x| pool(x, |window| window.iter().sum::<f64>() / 4.),
        ),
        case(
            "l2_norm_pooling_2d",
            |g, x| g.l2_norm_pooling_2d(x, &Pooling2dDescriptor::new([2, 2], [2, 2])),
            |x| pool(x, |window| window.iter().map(|v| v * v).sum::<f64>().sqrt()),
        ),
        case(
            "resize_nearest",
            |g, x| g.resize_nearest(x, [16, 32]),
            |x| {
                (0..COUNT * 4)
                    .map(|i| {
                        let (c, h, w) = (i / 512, i / 32 % 16, i % 32);
                        x[(c * 8 + h / 2) * 16 + w / 2]
                    })
                    .collect()
            },
        ),
        case(
            "resize_bilinear_align_corners",
            |g, x| g.resize_bilinear(x, [15, 31], ResizeSamplingMode::AlignCorners),
            |x| {
                (0..2 * 15 * 31)
                    .map(|i| {
                        let (c, h, w) = (i / (15 * 31), i / 31 % 15, i % 31);
                        bilinear(x, c, h as f64 * 7. / 14., w as f64 * 15. / 30.)
                    })
                    .collect()
            },
        ),
        case(
            "linear",
            |g, x| {
                let weight = g.constant(&operand()[..48], [3, 16])?;
                let bias = g.constant(&[0.5, -1., 0.25], [3])?;
                g.linear(x, &weight, Some(&bias))
            },
            |x| {
                let weight = operand();
                rows(x, 16, |row| {
                    (0..3)
                        .map(|n| {
                            row.iter()
                                .enumerate()
                                .map(|(k, v)| v * f64::from(weight[n * 16 + k]))
                                .sum::<f64>()
                                + [0.5, -1., 0.25][n]
                        })
                        .collect()
                })
            },
        ),
        case(
            "convolution_transpose_2d",
            |g, x| {
                let weight = g.constant(&operand()[..8], [2, 1, 2, 2])?;
                let descriptor = ConvolutionTranspose2dDescriptor {
                    strides: [2, 2],
                    ..Default::default()
                };
                g.convolution_transpose_2d(x, &weight, None, &descriptor)
            },
            |x| {
                let weight = operand();
                (0..16 * 32)
                    .map(|i| {
                        let (h, w) = (i / 32, i % 32);
                        (0..2)
                            .map(|c| {
                                x[(c * 8 + h / 2) * 16 + w / 2]
                                    * f64::from(weight[c * 4 + (h % 2) * 2 + w % 2])
                            })
                            .sum()
                    })
                    .collect()
            },
        ),
    ];
    for (name, domain) in [
        ("square_root", (0.05, 16.)),
        ("logarithm", (0.05, 16.)),
        ("reciprocal", (0.25, 16.)),
        ("reciprocal_square_root", (0.05, 16.)),
        ("power", (0.25, 4.)),
        ("reduction_log_sum", (0.25, 4.)),
    ] {
        let mut positive = match name {
            "square_root" => case(name, |g, x| g.square_root(x), |x| map(x, f64::sqrt)),
            "logarithm" => case(name, |g, x| g.logarithm(x, 0.), |x| map(x, f64::ln)),
            "reciprocal" => case(name, |g, x| g.reciprocal(x, 0.), |x| map(x, f64::recip)),
            "reciprocal_square_root" => case(
                name,
                |g, x| g.reciprocal_square_root(x, 0.),
                |x| map(x, |v| v.sqrt().recip()),
            ),
            "power" => case(
                name,
                |g, x| g.power(x, &g.constant(&operand(), SHAPE)?),
                |x| zip(x, &operand(), f64::powf),
            ),
            _ => case(
                name,
                |g, x| g.reduction_log_sum(x, &[-1]),
                |x| rows(x, 16, |row| vec![row.iter().sum::<f64>().ln()]),
            ),
        };
        positive.domain = domain;
        cases.push(positive);
    }
    cases
}

fn pool(x: &[f64], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    (0..2 * 4 * 8)
        .map(|i| {
            let (c, h, w) = (i / 32, i / 8 % 4, i % 8);
            let window: Vec<f64> = (0..4)
                .map(|k| x[(c * 8 + h * 2 + k / 2) * 16 + w * 2 + k % 2])
                .collect();
            f(&window)
        })
        .collect()
}

fn bilinear(x: &[f64], c: usize, y: f64, x_position: f64) -> f64 {
    let (y0, x0) = (y.floor() as usize, x_position.floor() as usize);
    let (y1, x1) = ((y0 + 1).min(7), (x0 + 1).min(15));
    let (dy, dx) = (y - y0 as f64, x_position - x0 as f64);
    let at = |h: usize, w: usize| x[(c * 8 + h) * 16 + w];
    at(y0, x0) * (1. - dy) * (1. - dx)
        + at(y0, x1) * (1. - dy) * dx
        + at(y1, x0) * dy * (1. - dx)
        + at(y1, x1) * dy * dx
}

#[test]
fn every_op_matches_its_reference_on_ane() -> Result<(), ane::Error> {
    let mut failures = Vec::new();
    for case in cases() {
        let (low, high) = case.domain;
        let input: Vec<f32> = (0..COUNT)
            .map(|i| {
                let t = (i * 7919 % 1000) as f32 / 999.;
                half::f16::from_f32(low + (high - low) * t).to_f32()
            })
            .collect();
        let graph = Graph::new();
        let x = graph.placeholder(SHAPE, DataType::Float32)?;
        let y = (case.build)(&graph, &x)?;
        let executable = graph.compile(&[y], &[], None)?;
        let data = executable.input(x)?.allocate()?;
        data.copy_from_f32(&input)?;
        let actual = executable.run(&[&data], None, None)?[0]
            .read_f32()?
            .to_vec();
        let expected = (case.reference)(&input.iter().map(|&v| f64::from(v)).collect::<Vec<_>>());
        let (index, error) = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| (f64::from(*a) - e).abs() / (1. + e.abs()))
            .enumerate()
            .fold(
                (0, 0.),
                |worst, (i, error)| if error > worst.1 { (i, error) } else { worst },
            );
        if actual.len() != expected.len() || error > case.tolerance {
            failures.push(format!(
                "{}: error {error:.2e} at {index}: got {} want {}",
                case.name, actual[index], expected[index]
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    Ok(())
}
