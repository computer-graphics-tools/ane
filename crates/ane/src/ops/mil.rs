use super::{ConvOp, MilProgram};
use crate::DataType;
use std::collections::HashSet;

use super::{
    activation_mode::ActivationMode,
    elementwise::ElementwiseOpType,
    op::Op,
    pad_fill_mode::PadFillMode,
    pad_mode::PadMode,
    pool_type::PoolType,
    reduction_mode::ReductionMode,
    scalar::ScalarOpType,
    weights::{WeightBlob, build_mil_weight_blob, mil_blob_chunk_offset},
};

const MIL_BUILD_INFO: &str = r#"[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]"#;

pub fn emit_mil(
    ops: &[Op],
    shapes: &[(String, [usize; 4])],
    inputs: &[(String, [usize; 4], DataType)],
    output_types: &[DataType],
) -> MilProgram {
    let shape_map: std::collections::HashMap<&str, [usize; 4]> = shapes
        .iter()
        .map(|(name, shape)| (name.as_str(), *shape))
        .collect();

    let all_tops: HashSet<&str> = ops.iter().map(Op::top).collect();
    let all_bottoms: HashSet<&str> = ops.iter().flat_map(Op::bottom_names).collect();
    let states: HashSet<&str> = ops
        .iter()
        .filter_map(|op| match op {
            Op::StateUpdate(op) => Some(op.state.as_str()),
            _ => None,
        })
        .collect();

    let input_specs: Vec<_> = inputs
        .iter()
        .filter(|(name, _, _)| {
            all_bottoms.contains(name.as_str()) && !all_tops.contains(name.as_str())
        })
        .cloned()
        .collect();
    let input_names: Vec<&str> = input_specs
        .iter()
        .map(|(name, _, _)| name.as_str())
        .collect();

    let output_names: Vec<&str> = ops
        .iter()
        .map(Op::top)
        .filter(|t| !all_bottoms.contains(t))
        .collect::<Vec<_>>()
        .into_iter()
        .fold(vec![], |mut acc, n| {
            if !acc.contains(&n) {
                acc.push(n);
            }
            acc
        });

    let mut weight_blobs: Vec<&WeightBlob> = Vec::new();
    for layer in ops.iter() {
        collect_weights(layer, &mut weight_blobs);
    }

    let weight_bytes: Box<[u8]> = if weight_blobs.is_empty() {
        Box::new([])
    } else {
        build_mil_weight_blob(&weight_blobs)
    };

    let mut out = String::new();
    out.push_str("program(1.3)\n");
    out.push_str(MIL_BUILD_INFO);
    out.push_str("\n{\n");

    out.push_str("    func main<ios18>(");
    let sig_parts: Vec<String> = input_specs
        .iter()
        .map(|(name, shape, data_type)| {
            if *data_type == DataType::Int32 {
                format!("tensor<int32, [1]> {name}")
            } else if states.contains(name.as_str()) {
                format!("state<tensor<fp16, {}>> {name}", mil_shape(*shape))
            } else {
                format!(
                    "tensor<{}, {}> {}",
                    data_type.mil_name(),
                    mil_shape(*shape),
                    name
                )
            }
        })
        .collect();
    out.push_str(&sig_parts.join(", "));
    out.push_str(") {\n");

    out.push_str(
        "        string _to_fp16 = const()[name = string(\"_to_fp16\"), val = string(\"fp16\")];\n",
    );

    for name in &input_names {
        if states.contains(name)
            || input_specs
                .iter()
                .any(|(n, _, d)| n == name && *d == DataType::Int32)
        {
            continue;
        }
        let shape = shape_map.get(name).copied().expect("missing tensor shape");
        out.push_str(&format!(
            "        tensor<fp16, {s}> {n}_f16 = cast(dtype = _to_fp16, x = {n})[name = string(\"cast_{n}\")];\n",
            s = mil_shape(shape),
            n = name,
        ));
    }

    let mut blob_index = 0usize;
    for layer in ops.iter() {
        emit_layer(layer, &shape_map, &weight_blobs, &mut blob_index, &mut out);
    }

    assert!(
        output_types.len() == 1 || output_types.len() == output_names.len(),
        "{} output types for {} outputs",
        output_types.len(),
        output_names.len()
    );
    let output_type = |index: usize| output_types[index.min(output_types.len() - 1)];
    for (index, name) in output_names.iter().enumerate() {
        let shape = shape_map.get(name).copied().expect("missing tensor shape");
        out.push_str(&format!(
            "        tensor<{dt}, {s}> {n} = cast(dtype = string(\"{dt}\"), x = {n}_f16)[name = string(\"cast_out_{n}\")];\n",
            s = mil_shape(shape),
            n = name,
            dt = output_type(index).mil_name(),
        ));
    }

    let ret = output_names.join(", ");
    out.push_str(&format!("    }} -> ({ret});\n"));
    out.push_str("}\n");

    let outputs = output_names
        .iter()
        .enumerate()
        .map(|(index, &name)| (name.to_string(), shape_map[name], output_type(index)))
        .collect();
    MilProgram {
        text: out,
        weights: weight_bytes,
        inputs: input_specs.into_boxed_slice(),
        outputs,
    }
}

fn mil_shape(s: [usize; 4]) -> String {
    format!("[{}, {}, {}, {}]", s[0], s[1], s[2], s[3])
}

fn collect_weights<'a>(layer: &'a Op, out: &mut Vec<&'a WeightBlob>) {
    match layer {
        Op::Constant(l) => {
            out.push(&l.data);
        }
        Op::InnerProduct(l) => {
            out.push(&l.weights);
            if let Some(b) = &l.bias {
                out.push(b);
            }
        }
        Op::Conv(l) => {
            out.push(&l.weights);
            if let Some(b) = &l.bias {
                out.push(b);
            }
        }
        Op::Deconv(l) => {
            out.push(&l.weights);
            if let Some(b) = &l.bias {
                out.push(b);
            }
        }
        Op::InstanceNorm(l) => {
            out.push(&l.params);
        }
        Op::Matmul(_)
        | Op::StateUpdate(_)
        | Op::Transpose(_)
        | Op::SliceBySize(_)
        | Op::ScalarOp(_)
        | Op::Elementwise(_)
        | Op::Activation(_)
        | Op::Softmax(_)
        | Op::Concat(_)
        | Op::Reshape(_)
        | Op::Pooling(_)
        | Op::Padding(_)
        | Op::Flatten(_)
        | Op::Reduction(_) => {}
    }
}

fn blobfile_ref(
    all_blobs: &[&WeightBlob],
    blob_index: usize,
    shape_str: &str,
    var_name: &str,
    out: &mut String,
) {
    let offset = mil_blob_chunk_offset(all_blobs, blob_index);
    out.push_str(&format!(
        "        tensor<fp16, {s}> {v} = const()[name = string(\"{v}\"), \
         val = tensor<fp16, {s}>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), \
         offset = uint64({o})))];\n",
        s = shape_str,
        v = var_name,
        o = offset,
    ));
}

fn emit_layer(
    layer: &Op,
    shape_map: &std::collections::HashMap<&str, [usize; 4]>,
    all_blobs: &[&WeightBlob],
    blob_index: &mut usize,
    out: &mut String,
) {
    match layer {
        Op::StateUpdate(l) => {
            let shape = shape_map[l.top.as_str()];
            let sh = mil_shape(shape);
            let n = &l.name;
            for (key, value) in [
                ("zero", 0),
                ("batch", shape[0]),
                ("channel", l.channel),
                ("channels", l.channel + l.channels),
                ("width", shape[3]),
                ("rows", l.rows),
            ] {
                out.push_str(&format!("        tensor<int32, [1]> {n}_{key} = const()[name = string(\"{n}_{key}\"), val = tensor<int32, [1]>([{value}])];\n"));
            }
            out.push_str(&format!(r#"        int32 {n}_axis = const()[name = string("{n}_axis"), val = int32(0)];
        bool {n}_interleave = const()[name = string("{n}_interleave"), val = bool(false)];
        tensor<int32, [1]> {n}_limit = add(x = {position}, y = {n}_rows)[name = string("{n}_limit")];
        tensor<int32, [4]> {n}_begin = concat(axis = {n}_axis, interleave = {n}_interleave, values = ({n}_zero, {n}_channel, {position}, {n}_zero))[name = string("{n}_begin")];
        tensor<int32, [4]> {n}_end = concat(axis = {n}_axis, interleave = {n}_interleave, values = ({n}_batch, {n}_channels, {n}_limit, {n}_width))[name = string("{n}_end")];
        tensor<int32, [4]> {n}_stride = const()[name = string("{n}_stride"), val = tensor<int32, [4]>([1, 1, 1, 1])];
        tensor<bool, [4]> {n}_mask = const()[name = string("{n}_mask"), val = tensor<bool, [4]>([false, false, false, false])];
        tensor<fp16, {sh}> {n}_old = read_state(input = {state})[name = string("{n}_read")];
        tensor<fp16, {sh}> {n}_next = slice_update(x = {n}_old, update = {bottom}_f16, begin = {n}_begin, end = {n}_end, stride = {n}_stride, begin_mask = {n}_mask, end_mask = {n}_mask, squeeze_mask = {n}_mask)[name = string("{n}_insert")];
        write_state(input = {state}, data = {n}_next)[name = string("{n}_write")];
        tensor<fp16, {sh}> {top}_f16 = read_state(input = {state})[name = string("{n}")];
"#, position = l.position, state = l.state, bottom = l.bottom, top = l.top));
        }
        Op::Constant(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(shape);
            blobfile_ref(all_blobs, *blob_index, &sh, &format!("{}_f16", l.top), out);
            *blob_index += 1;
        }

        Op::InnerProduct(l) => {
            let conv = Op::Conv(ConvOp {
                name: l.name.clone(),
                bottom: l.bottom.clone(),
                top: l.top.clone(),
                input_channels: l.input_channels,
                output_channels: l.output_channels,
                kernel_height: 1,
                kernel_width: 1,
                groups: 1,
                pad_mode: PadMode::Valid,
                pad_top: 0,
                pad_bottom: 0,
                pad_left: 0,
                pad_right: 0,
                weights: l.weights.clone(),
                bias: l.bias.clone(),
                fused_relu: l.has_relu,
                fused_tanh: l.has_tanh,
            });
            emit_layer(&conv, shape_map, all_blobs, blob_index, out);
        }

        Op::Conv(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let n = &l.name;
            let in_ch = l.input_channels;
            let out_ch = l.output_channels;
            let kh = l.kernel_height;
            let kw = l.kernel_width;
            let groups = l.groups;

            let pad_type_str = match l.pad_mode {
                PadMode::Valid => "valid",
                PadMode::Same => "same_lower",
            };
            emit_conv_constants(
                n,
                [l.pad_top, l.pad_bottom, l.pad_left, l.pad_right],
                [1, 1],
                groups,
                pad_type_str,
                out,
            );

            let w_shape = format!(
                "[{out_ch}, {per_group}, {kh}, {kw}]",
                per_group = in_ch / groups
            );
            let w_var = format!("{n}_W");
            blobfile_ref(all_blobs, *blob_index, &w_shape, &w_var, out);
            *blob_index += 1;

            let out_sh = mil_shape(out_shape);

            let bias_param = if l.bias.is_some() {
                let b_shape = format!("[{out_ch}]");
                let b_var = format!("{n}_b");
                blobfile_ref(all_blobs, *blob_index, &b_shape, &b_var, out);
                *blob_index += 1;
                format!(", bias = {b_var}")
            } else {
                String::new()
            };

            let conv_out = if l.fused_relu || l.fused_tanh {
                format!("{n}_conv_out")
            } else {
                format!("{}_f16", l.top)
            };

            out.push_str(&format!(
                "        tensor<fp16, {out_sh}> {conv_out} = conv(\
                 dilations = {n}_dilations, groups = {n}_groups, pad = {n}_pad, \
                 pad_type = {n}_pad_type, strides = {n}_strides{bias_param}, weight = {w_var}, \
                 x = {bot}_f16)[name = string(\"{n}\")];\n",
                bot = l.bottom,
            ));

            if l.fused_relu {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = relu(x = {conv_out})[name = string(\"{n}_relu\")];\n",
                    top = l.top,
                ));
            } else if l.fused_tanh {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = tanh(x = {conv_out})[name = string(\"{n}_tanh\")];\n",
                    top = l.top,
                ));
            }
        }

        Op::Deconv(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let n = &l.name;
            let in_ch = l.input_channels;
            let out_ch = l.output_channels;
            let kh = l.kernel_height;
            let kw = l.kernel_width;
            let groups = l.groups;
            let sh = l.stride_height;
            let sw = l.stride_width;

            let pad_type_str = match l.pad_mode {
                PadMode::Valid => "valid",
                PadMode::Same => "same_lower",
            };

            emit_conv_constants(
                n,
                [l.pad_top, l.pad_bottom, l.pad_left, l.pad_right],
                [sh, sw],
                groups,
                pad_type_str,
                out,
            );
            if l.output_padding_height > 0 || l.output_padding_width > 0 {
                out.push_str(&format!(
                    "        tensor<int32, [2]> {n}_out_pad = const()[name = string(\"{n}_out_pad\"), val = tensor<int32, [2]>([{}, {}])];\n",
                    l.output_padding_height, l.output_padding_width,
                ));
            }

            let w_shape = format!(
                "[{in_ch}, {per_group}, {kh}, {kw}]",
                per_group = out_ch / groups
            );
            let w_var = format!("{n}_W");
            blobfile_ref(all_blobs, *blob_index, &w_shape, &w_var, out);
            *blob_index += 1;

            let out_sh = mil_shape(out_shape);
            let deconv_out = if l.bias.is_some() || l.fused_relu || l.fused_tanh {
                format!("{n}_deconv_out")
            } else {
                format!("{}_f16", l.top)
            };

            if l.output_padding_height > 0 || l.output_padding_width > 0 {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {deconv_out} = conv_transpose(\
                     dilations = {n}_dilations, groups = {n}_groups, pad = {n}_pad, \
                     pad_type = {n}_pad_type, strides = {n}_strides, weight = {w_var}, \
                     output_padding = {n}_out_pad, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    bot = l.bottom,
                ));
            } else {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {deconv_out} = conv_transpose(\
                     dilations = {n}_dilations, groups = {n}_groups, pad = {n}_pad, \
                     pad_type = {n}_pad_type, strides = {n}_strides, weight = {w_var}, \
                     x = {bot}_f16)[name = string(\"{n}\")];\n",
                    bot = l.bottom,
                ));
            }

            let after_bias = if l.bias.is_some() {
                let b_shape = format!("[{out_ch}]");
                let b_var = format!("{n}_b");
                blobfile_ref(all_blobs, *blob_index, &b_shape, &b_var, out);
                *blob_index += 1;
                let biased = if l.fused_relu || l.fused_tanh {
                    format!("{n}_biased")
                } else {
                    format!("{}_f16", l.top)
                };
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {biased} = add(x = {deconv_out}, y = {b_var})[name = string(\"{n}_bias\")];\n",
                ));
                biased
            } else {
                deconv_out
            };

            if l.fused_relu {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = relu(x = {after_bias})[name = string(\"{n}_relu\")];\n",
                    top = l.top,
                ));
            } else if l.fused_tanh {
                out.push_str(&format!(
                    "        tensor<fp16, {out_sh}> {top}_f16 = tanh(x = {after_bias})[name = string(\"{n}_tanh\")];\n",
                    top = l.top,
                ));
            }
        }

        Op::Activation(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(shape);
            let n = &l.name;
            let bot = &l.bottom;
            let top = &l.top;

            match l.mode {
                ActivationMode::Relu => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = relu(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::Tanh => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = tanh(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::LeakyRelu { negative_slope } => {
                    out.push_str(&format!(
                        "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({negative_slope})];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = leaky_relu(alpha = {n}_alpha, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::Sigmoid => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = sigmoid(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::Elu { alpha } => {
                    out.push_str(&format!(
                        "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({alpha})];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = elu(alpha = {n}_alpha, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::Linear { alpha, beta } => {
                    out.push_str(&format!(
                        "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({alpha})];\n",
                    ));
                    out.push_str(&format!(
                        "        fp32 {n}_beta = const()[name = string(\"{n}_beta\"), val = fp32({beta})];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = linear_activation(alpha = {n}_alpha, beta = {n}_beta, x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::SigmoidHard { alpha, beta } => {
                    out.push_str(&format!(
                        "        fp32 {n}_alpha = const()[name = string(\"{n}_alpha\"), val = fp32({alpha})];\n",
                    ));
                    out.push_str(&format!(
                        "        fp32 {n}_beta = const()[name = string(\"{n}_beta\"), val = fp32({beta})];\n",
                    ));
                    out.push_str(&format!(
                        "        fp32 {n}_clip_lo = const()[name = string(\"{n}_clip_lo\"), val = fp32(0.0)];\n",
                    ));
                    out.push_str(&format!(
                        "        fp32 {n}_clip_hi = const()[name = string(\"{n}_clip_hi\"), val = fp32(1.0)];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {n}_linear = linear_activation(alpha = {n}_alpha, beta = {n}_beta, x = {bot}_f16)[name = string(\"{n}_linear\")];\n",
                    ));
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = clip(alpha = {n}_clip_lo, beta = {n}_clip_hi, x = {n}_linear)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::SoftPlus => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = softplus(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
                ActivationMode::SoftSign => {
                    out.push_str(&format!(
                        "        tensor<fp16, {sh}> {top}_f16 = softsign(x = {bot}_f16)[name = string(\"{n}\")];\n",
                    ));
                }
            }
        }

        Op::Elementwise(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(shape);
            let n = &l.name;
            let top = &l.top;

            if l.operation == ElementwiseOpType::Threshold {
                let bot = &l.bottoms[0];
                let alpha = l.alpha;
                out.push_str(&format!("        tensor<fp16, {sh}> {top}_f16 = threshold(x = {bot}_f16, alpha = fp16({alpha}))[name = string(\"{n}\")];\n"));
                return;
            }
            let (mil_op, is_binary) = match l.operation {
                ElementwiseOpType::Add => ("add", true),
                ElementwiseOpType::Multiply => ("mul", true),
                ElementwiseOpType::Max => ("maximum", true),
                ElementwiseOpType::Min => ("minimum", true),
                ElementwiseOpType::Sub => ("sub", true),
                ElementwiseOpType::Div => ("real_div", true),
                ElementwiseOpType::Pow => ("pow", true),
                ElementwiseOpType::Abs => ("abs", false),
                ElementwiseOpType::Sqrt => ("sqrt", false),
                ElementwiseOpType::Rsqrt => ("rsqrt", false),
                ElementwiseOpType::Inverse => ("inverse", false),
                ElementwiseOpType::Exp => ("exp", false),
                ElementwiseOpType::Floor => ("floor", false),
                ElementwiseOpType::Log => ("log", false),
                ElementwiseOpType::Threshold => ("threshold", false),
            };

            if is_binary && l.bottoms.len() >= 2 {
                let a = &l.bottoms[0];
                let b = &l.bottoms[1];
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = {mil_op}(x = {a}_f16, y = {b}_f16)[name = string(\"{n}\")];\n",
                ));
            } else {
                let bot = l.bottoms.first().map(|s| s.as_str()).unwrap_or("");
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16 = {mil_op}(x = {bot}_f16)[name = string(\"{n}\")];\n",
                ));
            }

            if l.fused_relu {
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_relu_f16 = relu(x = {top}_f16)[name = string(\"{n}_relu\")];\n",
                ));
                out.push_str(&format!(
                    "        tensor<fp16, {sh}> {top}_f16_final = identity(x = {top}_relu_f16)[name = string(\"{n}_id\")];\n",
                ));
            }
        }

        Op::Softmax(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(shape);
            let n = &l.name;
            let axis = l.axis;
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({axis})];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = softmax(axis = {n}_axis, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Concat(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(shape);
            let n = &l.name;
            let axis = l.axis;
            let inputs: Vec<String> = l.bottoms.iter().map(|b| format!("{b}_f16")).collect();
            let inputs_str = inputs.join(", ");
            out.push_str(&format!(
                "        int32 {n}_axis = const()[name = string(\"{n}_axis\"), val = int32({axis})];\n",
            ));
            out.push_str(&format!(
                "        bool {n}_interleave = const()[name = string(\"{n}_interleave\"), val = bool(false)];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = concat(axis = {n}_axis, interleave = {n}_interleave, values = ({inputs_str}))[name = string(\"{n}\")];\n",
                top = l.top,
            ));
        }

        Op::Reshape(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(shape);
            let n = &l.name;
            let [s0, s1, s2, s3] = l.target_shape;
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_shape = const()[name = string(\"{n}_shape\"), val = tensor<int32, [4]>([{s0}, {s1}, {s2}, {s3}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = reshape(shape = {n}_shape, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Flatten(l) => {
            let in_shape = shape_map
                .get(l.bottom.as_str())
                .copied()
                .expect("missing tensor shape");
            let flat = in_shape.iter().product::<usize>();
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing flatten output shape");
            let sh = mil_shape(out_shape);
            let n = &l.name;
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_shape = const()[name = string(\"{n}_shape\"), val = tensor<int32, [4]>([1, {flat}, 1, 1])];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = reshape(shape = {n}_shape, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::InstanceNorm(l) => {
            let shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(shape);
            let n = &l.name;
            let ch = l.channels;
            let eps = l.epsilon;

            let gamma_shape = format!("[{ch}]");
            let gamma_var = format!("{n}_gamma");
            let beta_var = format!("{n}_beta");

            blobfile_ref(all_blobs, *blob_index, &gamma_shape, &gamma_var, out);
            *blob_index += 1;

            out.push_str(&format!(
                "        tensor<fp16, [{ch}]> {beta_var} = const()[name = string(\"{beta_var}\"), val = tensor<fp16, [{ch}]>({})];\n",
                format_args!("[{}]", vec!["0.0"; ch].join(", ")),
            ));
            out.push_str(&format!(
                "        fp32 {n}_eps = const()[name = string(\"{n}_eps\"), val = fp32({eps})];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = instance_norm(beta = {beta_var}, eps = {n}_eps, gamma = {gamma_var}, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Pooling(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let kh = l.kernel_height;
            let kw = l.kernel_width;
            let sh_s = l.stride_height;
            let sw_s = l.stride_width;

            let pad_type_str = match l.pad_mode {
                PadMode::Valid => "valid",
                PadMode::Same => "same_lower",
            };

            out.push_str(&format!(
                "        tensor<int32, [2]> {n}_kernel = const()[name = string(\"{n}_kernel\"), val = tensor<int32, [2]>([{kh}, {kw}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<int32, [2]> {n}_strides = const()[name = string(\"{n}_strides\"), val = tensor<int32, [2]>([{sh_s}, {sw_s}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_pad = const()[name = string(\"{n}_pad\"), val = tensor<int32, [4]>([{}, {}, {}, {}])];\n",
                l.pad_top, l.pad_bottom, l.pad_left, l.pad_right,
            ));
            out.push_str(&format!(
                "        string {n}_pad_type = const()[name = string(\"{n}_pad_type\"), val = string(\"{pad_type_str}\")];\n",
            ));
            out.push_str(&format!(
                "        bool {n}_ceil = const()[name = string(\"{n}_ceil\"), val = bool(false)];\n",
            ));

            let mil_pool_op = match l.pool_type {
                PoolType::Max => "max_pool",
                PoolType::Average | PoolType::L2 => "avg_pool",
            };

            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = {mil_pool_op}(ceil_mode = {n}_ceil, \
                 kernel_sizes = {n}_kernel, pad = {n}_pad, pad_type = {n}_pad_type, strides = {n}_strides, \
                 x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Padding(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(out_shape);
            let n = &l.name;

            let mode_str = match l.pad_fill_mode {
                PadFillMode::Constant => "constant",
                PadFillMode::Reflect => "reflect",
                PadFillMode::Replicate => "replicate",
            };

            let (pt, pb, pl, pr) = (l.pad_top, l.pad_bottom, l.pad_left, l.pad_right);
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_amounts = const()[name = string(\"{n}_amounts\"), val = tensor<int32, [4]>([{pt}, {pb}, {pl}, {pr}])];\n",
            ));
            out.push_str(&format!(
                "        string {n}_mode = const()[name = string(\"{n}_mode\"), val = string(\"{mode_str}\")];\n",
            ));
            out.push_str(&format!(
                "        fp16 {n}_val = const()[name = string(\"{n}_val\"), val = fp16({})];\n",
                l.pad_value,
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = pad(constant_val = {n}_val, mode = {n}_mode, \
                 pad = {n}_amounts, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Reduction(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(out_shape);
            let n = &l.name;

            let mil_reduce_op = match l.mode {
                ReductionMode::Sum => "reduce_sum",
                ReductionMode::Mean => "reduce_mean",
                ReductionMode::Min => "reduce_min",
                ReductionMode::Max => "reduce_max",
            };

            out.push_str(&format!(
                "        tensor<int32, [1]> {n}_axes = const()[name = string(\"{n}_axes\"), val = tensor<int32, [1]>([{}])];\n",
                l.axis,
            ));
            out.push_str(&format!(
                "        bool {n}_keep = const()[name = string(\"{n}_keep\"), val = bool(true)];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = {mil_reduce_op}(axes = {n}_axes, keep_dims = {n}_keep, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::Matmul(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let tx = if l.transpose_x { "true" } else { "false" };
            let ty = if l.transpose_y { "true" } else { "false" };
            out.push_str(&format!(
                "        bool {n}_tx = const()[name = string(\"{n}_tx\"), val = bool({tx})];\n",
            ));
            out.push_str(&format!(
                "        bool {n}_ty = const()[name = string(\"{n}_ty\"), val = bool({ty})];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = matmul(transpose_x = {n}_tx, transpose_y = {n}_ty, \
                 x = {bx}_f16, y = {by}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bx = l.bottom_x,
                by = l.bottom_y,
            ));
        }

        Op::Transpose(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let [p0, p1, p2, p3] = l.perm;
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_perm = const()[name = string(\"{n}_perm\"), val = tensor<int32, [4]>([{p0}, {p1}, {p2}, {p3}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = transpose(perm = {n}_perm, x = {bot}_f16)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::SliceBySize(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let [b0, b1, b2, b3] = l.begin;
            let [s0, s1, s2, s3] = l.size;
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_begin = const()[name = string(\"{n}_begin\"), val = tensor<int32, [4]>([{b0}, {b1}, {b2}, {b3}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<int32, [4]> {n}_size = const()[name = string(\"{n}_size\"), val = tensor<int32, [4]>([{s0}, {s1}, {s2}, {s3}])];\n",
            ));
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = slice_by_size(x = {bot}_f16, begin = {n}_begin, size = {n}_size)[name = string(\"{n}\")];\n",
                top = l.top,
                bot = l.bottom,
            ));
        }

        Op::ScalarOp(l) => {
            let out_shape = shape_map
                .get(l.top.as_str())
                .copied()
                .expect("missing tensor shape");
            let sh = mil_shape(out_shape);
            let n = &l.name;
            let s = l.scalar;
            out.push_str(&format!(
                "        fp16 {n}_s = const()[name = string(\"{n}_s\"), val = fp16({s})];\n",
            ));
            let op_str = match l.op {
                ScalarOpType::Mul => format!("mul(x = {bot}_f16, y = {n}_s)", bot = l.bottom),
                ScalarOpType::Add => format!("add(x = {bot}_f16, y = {n}_s)", bot = l.bottom),
                ScalarOpType::RSub => format!("sub(x = {n}_s, y = {bot}_f16)", bot = l.bottom),
                ScalarOpType::Min => format!("minimum(x = {bot}_f16, y = {n}_s)", bot = l.bottom),
                ScalarOpType::Max => format!("maximum(x = {bot}_f16, y = {n}_s)", bot = l.bottom),
                ScalarOpType::Pow => format!("pow(x = {bot}_f16, y = {n}_s)", bot = l.bottom),
            };
            out.push_str(&format!(
                "        tensor<fp16, {sh}> {top}_f16 = {op_str}[name = string(\"{n}\")];\n",
                top = l.top,
            ));
        }
    }
}

fn emit_conv_constants(
    n: &str,
    pad: [usize; 4],
    strides: [usize; 2],
    groups: usize,
    pad_type: &str,
    out: &mut String,
) {
    let [pt, pb, pl, pr] = pad;
    let [sh, sw] = strides;
    out.push_str(&format!(
        "        string {n}_pad_type = const()[name = string(\"{n}_pad_type\"), val = string(\"{pad_type}\")];\n",
    ));
    out.push_str(&format!(
        "        tensor<int32, [2]> {n}_strides = const()[name = string(\"{n}_strides\"), val = tensor<int32, [2]>([{sh}, {sw}])];\n",
    ));
    out.push_str(&format!(
        "        tensor<int32, [4]> {n}_pad = const()[name = string(\"{n}_pad\"), val = tensor<int32, [4]>([{pt}, {pb}, {pl}, {pr}])];\n",
    ));
    out.push_str(&format!(
        "        tensor<int32, [2]> {n}_dilations = const()[name = string(\"{n}_dilations\"), val = tensor<int32, [2]>([1, 1])];\n",
    ));
    out.push_str(&format!(
        "        int32 {n}_groups = const()[name = string(\"{n}_groups\"), val = int32({groups})];\n",
    ));
}
