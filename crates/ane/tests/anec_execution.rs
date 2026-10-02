use ane::{
    CompilationCache, CompilationDescriptor, Convolution2dDescriptor, DataType, Graph, PadFillMode,
    SamplingMode, Tensor, TensorData,
};
use half::f16;

#[test]
fn cached_effects_keep_independent_state() -> Result<(), ane::Error> {
    let mut cache = CompilationCache::new(2, 1024 * 1024)?;
    let mut executables = Vec::new();
    for initial in [0.0, 10.0] {
        let graph = Graph::new();
        let input = graph.placeholder([64, 64], DataType::Float32)?;
        let state = graph.variable_with_data(&vec![initial; 4096], [64, 64])?;
        let old = graph.read_variable(&state)?;
        let next = graph.addition(&old, &input)?;
        let update = graph.assign_variable(&state, &next)?;
        let output = graph.square(&input)?;
        let executable = graph.compile_cached(&[output], &[&update], None, &mut cache)?;
        executables.push((executable, input, output, state));
    }
    assert_eq!(cache.hits(), 1);
    for index in [0, 1, 0] {
        let (executable, input, _, _) = &executables[index];
        let data = executable.input(*input)?.allocate()?;
        data.copy_from_f32(&[1.0; 4096])?;
        let result = executable.run(&[&data], None, None)?;
        assert!(result[0].read_f32()?.iter().all(|&v| v == 1.0));
    }
    for ((executable, _, _, state), expected) in executables.iter().zip([2.0, 11.0]) {
        assert!(
            executable
                .variable_data(state)?
                .read_f32()?
                .iter()
                .all(|&v| v == expected)
        );
    }
    Ok(())
}

#[test]
fn large_embedding_rows() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let position = graph.placeholder([1], DataType::Int32)?;
    let rows = 248_320;
    let data: Vec<f32> = (0..rows * 64)
        .map(|i| ((i / 64) % 1024) as f32 + (i % 64) as f32)
        .collect();
    let table = graph.placeholder([1, 1, rows, 64], DataType::Float16)?;
    let result = graph.slice_dynamic(&table, &position, 2, 1)?;
    let executable = graph.compile(&[result], &[], None)?;
    let input = executable.input(position)?.allocate()?;
    let table_data = executable.input(table)?.allocate()?;
    table_data.copy_from_f32(&data)?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let feeds = [&input, &table_data];
    for row in [0, 2049, 65_537, 248_319] {
        input.write(&[row as i32])?;
        executable.run(&feeds, Some(&results), None)?;
        let output = outputs[0].read_f32()?;
        assert_eq!(&*output, &data[row * 64..(row + 1) * 64]);
    }
    Ok(())
}

#[test]
fn direct_anec_execution_preserves_bindings_state_and_packed_io() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float32)?;
    let state = graph.variable_with_data(&[0.; 4096], [64, 64])?;
    let previous = graph.read_variable(&state)?;
    let next = graph.addition(&previous, &input)?;
    graph.assign_variable(&state, &next)?;
    let result = graph.read_variable(&state)?;
    let executable = graph.compile(&[result], &[], None)?;
    assert!(executable.report().anec_bytes > 0);
    let input_data = executable.input(input)?.allocate()?;
    input_data.copy_from_f32(&[1.; 4096])?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let feeds = [&input_data];
    for expected in [1., 2., 3.] {
        executable.run(&feeds, Some(&results), None)?;
        assert!(outputs[0].read_f32()?.iter().all(|&v| v == expected));
        assert!(
            executable
                .variable_data(&state)?
                .read_f32()?
                .iter()
                .all(|&v| v == expected)
        );
    }
    drop(executable);

    let graph = Graph::new();
    let input = graph.placeholder([1, 1, 1, 64], DataType::Float32)?;
    let position = graph.placeholder([1], DataType::Int32)?;
    let state = graph.variable_with_data(&[1.; 8192], [1, 2, 64, 64])?;
    graph.assign_variable_rows(&state, &input, &position, 1)?;
    let result = graph.read_variable(&state)?;
    let executable = graph.compile(&[result], &[], None)?;
    let input_data = executable.input(input)?.allocate()?;
    let position_data = executable.input(position)?.allocate()?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let feeds = [&input_data, &position_data];
    for (row, value) in [(3, 2.), (7, 4.)] {
        input_data.copy_from_f32(&[value; 64])?;
        position_data.write(&[row])?;
        executable.run(&feeds, Some(&results), None)?;
        let actual = outputs[0].read_f32()?;
        for (index, &value) in actual.iter().enumerate() {
            let expected = if (4096 + 3 * 64..4096 + 4 * 64).contains(&index) {
                2.
            } else if row == 7 && (4096 + 7 * 64..4096 + 8 * 64).contains(&index) {
                4.
            } else {
                1.
            };
            assert_eq!(value, expected, "row update at {index}");
        }
    }
    drop(executable);

    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float32)?;
    let (values, indices) = graph.top_k(&input, 1, 1)?;
    let executable = graph.compile(&[indices], &[], None)?;
    let input_data = executable.input(input)?.allocate()?;
    let data = (0..4096).map(|i| (i % 64) as f32).collect::<Vec<_>>();
    input_data.copy_from_f32(&data)?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let feeds = [&input_data];
    executable.run(&feeds, Some(&results), None)?;
    assert!(outputs[0].read::<u16>()?.iter().all(|&v| v == 63));
    drop(executable);
    let executable = graph.compile(&[values], &[], None)?;
    let input_data = executable.input(input)?.allocate()?;
    input_data.copy_from_f32(&data)?;
    let result = executable.run(&[&input_data], None, None)?;
    assert!(result[0].read_f32()?.iter().all(|&v| v == 63.));
    drop(executable);

    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float16)?;
    let indices = graph.placeholder([64], DataType::UInt16)?;
    let output = graph.gather(&input, &indices, 0)?;
    let executable = graph.compile(&[output], &[], None)?;
    let input_data = executable.input(input)?.allocate()?;
    let index_data = executable.input(indices)?.allocate()?;
    input_data.copy_from_f32(&(0..4096).map(|i| (i / 64) as f32).collect::<Vec<_>>())?;
    index_data.write(&(0..64).rev().map(|i| i as u16).collect::<Vec<_>>())?;
    let result = executable.run(&[&input_data, &index_data], None, None)?;
    assert!(
        result[0]
            .read_f32()?
            .iter()
            .enumerate()
            .all(|(i, &v)| v == (63 - i / 64) as f32)
    );
    drop(executable);

    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float16)?;
    let weights = graph.palettized_weights(&[0xe4; 1024], 2, [64, 64], &[0., 1., 2., 3.])?;
    let output = graph.matrix_multiplication(&input, &weights, false, false)?;
    let executable = graph.compile(&[output], &[], None)?;
    let input_data = executable.input(input)?.allocate()?;
    let identity = (0..4096)
        .map(|i| f16::from_f32(if i / 64 == i % 64 { 1. } else { 0. }))
        .collect::<Vec<_>>();
    input_data.write(&identity)?;
    let result = executable.run(&[&input_data], None, None)?;
    assert!(
        result[0]
            .read_f32()?
            .iter()
            .enumerate()
            .all(|(i, &v)| v == (i % 4) as f32)
    );
    drop(executable);

    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float16)?;
    let output = graph.quantize(&input, &[0.25], Some(&[0]), None, DataType::Int8)?;
    let executable = graph.compile(
        &[output],
        &[],
        Some(&CompilationDescriptor {
            output_types: Some(vec![DataType::Int8]),
            ..Default::default()
        }),
    )?;
    let data = executable.input(input)?.allocate()?;
    data.copy_from_f32(&[1.5; 4096])?;
    let result = executable.run(&[&data], None, None)?;
    assert!(result[0].read::<i8>()?.iter().all(|&v| v == 6));
    drop(executable);

    let mut cache = CompilationCache::new(2, 1024 * 1024)?;
    let build = |initial: f32, cache: &mut CompilationCache| -> Result<_, ane::Error> {
        let graph = Graph::new();
        let x = graph.placeholder([64, 64], DataType::Float32)?;
        let w = graph.variable_with_data(&[initial; 4096], [64, 64])?;
        let value = graph.read_variable(&w)?;
        let y = graph.matrix_multiplication(&x, &value, false, false)?;
        Ok((graph.compile_cached(&[y], &[], None, cache)?, x, w, y))
    };
    let (a, x, wa, ya) = build(1., &mut cache)?;
    let (b, _, wb, _) = build(2., &mut cache)?;
    assert_eq!(cache.hits(), 1);
    assert_ne!(
        a.variable_data(&wa)?.surface().surfaceID(),
        b.variable_data(&wb)?.surface().surfaceID()
    );
    let input = a.input(x)?.allocate()?;
    input.copy_from_f32(&[1.; 4096])?;
    assert!(
        a.run(&[&input], None, None)?[0]
            .read_f32()?
            .iter()
            .all(|&v| v == 64.)
    );
    assert!(
        b.run(&[&input], None, None)?[0]
            .read_f32()?
            .iter()
            .all(|&v| v == 128.)
    );
    let output = a.output(ya)?.allocate()?;
    for _ in 0..3 {
        let results = a.run_async(&[&input], Some(&[&output]), None)?.wait()?;
        assert_eq!(
            results[0].surface().surfaceID(),
            output.surface().surfaceID()
        );
        assert!(output.read_f32()?.iter().all(|&v| v == 64.));
    }
    Ok(())
}

#[test]
fn runtime_convolution_weights() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([1, 64, 1, 64], DataType::Float16)?;
    let weights = graph.placeholder([64, 64, 1, 1], DataType::Float16)?;
    let output = graph.convolution_2d_1x1(&input, &weights, None)?;
    let executable = graph.compile(&[output], &[], None)?;
    let input_data = executable.input(input)?.allocate()?;
    let weight_data = executable.input(weights)?.allocate()?;
    input_data.copy_from_f32(&[1.; 4096])?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let feeds = [&input_data, &weight_data];
    for value in [1., 2.] {
        weight_data.copy_from_f32(&[value; 4096])?;
        executable.run(&feeds, Some(&results), None)?;
        assert!(outputs[0].read_f32()?.iter().all(|&v| v == 64. * value));
    }
    drop(executable);
    Ok(())
}

#[test]
fn native_palettized_linear() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([128, 128], DataType::Float16)?;
    let weights = graph.palettized_weights(
        &[0x10; 8192],
        4,
        [128, 128],
        &[
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        ],
    )?;
    let output = graph.matrix_multiplication(&input, &weights, false, false)?;
    let mlir = graph.program(&[output], &[DataType::Float32])?.mlir()?;
    assert!(mlir.contains("\"mps.dequantize_lut\""));
    assert!(mlir.contains("tensor<128x128x1x1xui4>"));
    let executable = graph.compile(&[output], &[], None)?;
    let data = executable.input(input)?.allocate()?;
    data.copy_from_f32(
        &(0..16384)
            .map(|i| if i / 128 == i % 128 { 1. } else { 0. })
            .collect::<Vec<_>>(),
    )?;
    let result = executable.run(&[&data], None, None)?;
    assert!(
        result[0]
            .read_f32()?
            .iter()
            .enumerate()
            .all(|(i, &v)| v == (i % 2) as f32)
    );
    Ok(())
}

#[test]
fn runtime_quantization_scales() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float16)?;
    let scale = graph.placeholder([1], DataType::Float16)?;
    let codes = graph.quantize_with_scale_tensor(&input, &scale, 0, None, DataType::Int8)?;
    let restored = graph.dequantize_with_scale_tensor(&codes, &scale, 0, None)?;
    let executable = graph.compile(&[codes, restored], &[], None)?;
    let data = executable.input(input)?.allocate()?;
    let scale_data = executable.input(scale)?.allocate()?;
    data.copy_from_f32(&[1.5; 4096])?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let feeds = [&data, &scale_data];
    for (factor, expected) in [(0.25, 6i8), (0.5, 3)] {
        scale_data.copy_from_f32(&[factor])?;
        executable.run(&feeds, Some(&results), None)?;
        let actual = outputs[0].read::<i8>()?;
        assert!(
            actual.iter().all(|&v| v == expected),
            "scale {factor}, codes {:?}",
            &actual[..8]
        );
        assert!(outputs[1].read_f32()?.iter().all(|&v| v == 1.5));
    }
    Ok(())
}

#[test]
fn byte_casts_and_packed_results() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float16)?;
    let bytes = graph.cast(&input, DataType::Int8)?;
    let packed = graph.pack_signed_int4(&input)?;
    assert_eq!(packed.data_type(), DataType::UInt8);
    let executable = graph.compile(&[bytes, packed], &[], None)?;
    let data = executable.input(input)?.allocate()?;
    data.copy_from_f32(
        &(0..4096)
            .map(|i| if i % 2 == 0 { -8. } else { 7. })
            .collect::<Vec<_>>(),
    )?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let feeds = [&data];
    executable.run(&feeds, Some(&results), None)?;
    assert!(
        outputs[0]
            .read::<i8>()?
            .iter()
            .enumerate()
            .all(|(i, &v)| v == if i % 2 == 0 { -8 } else { 7 })
    );
    assert!(outputs[1].read::<u8>()?.iter().all(|&v| v == 0x78));
    data.copy_from_f32(&[1.75; 4096])?;
    executable.run(&feeds, Some(&results), None)?;
    assert!(outputs[0].read::<i8>()?.iter().all(|&v| v == 1));
    Ok(())
}

#[test]
fn byte_state() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let state_data = TensorData::from_slice(&[0i8; 4096], [1, 1, 64, 64])?;
    let state = graph.variable_with_tensor_data(&state_data)?;
    let update = graph.placeholder([1, 1, 1, 64], DataType::Int8)?;
    let position = graph.placeholder([1], DataType::Int32)?;
    graph.assign_variable_rows(&state, &update, &position, 0)?;
    let output = graph.read_variable(&state)?;
    let executable = graph.compile(&[output], &[], None)?;
    let update_data = executable.input(update)?.allocate()?;
    let position_data = executable.input(position)?.allocate()?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let feeds = [&update_data, &position_data];
    for (row, value) in [(3i32, -100i8), (7, 100)] {
        update_data.write(&[value; 64])?;
        position_data.write(&[row])?;
        executable.run(&feeds, Some(&results), None)?;
        let actual = state_data.read::<i8>()?;
        assert!(actual[3 * 64..4 * 64].iter().all(|&v| v == -100));
        if row == 7 {
            assert!(actual[7 * 64..8 * 64].iter().all(|&v| v == 100));
        }
        assert_eq!(*outputs[0].read::<i8>()?, *actual);
    }
    drop(executable);
    let graph = Graph::new();
    let state = graph.variable_with_tensor_data(&state_data)?;
    let state = graph.read_variable(&state)?;
    let position = graph.placeholder([1], DataType::Int32)?;
    let window = graph.slice_dynamic(&state, &position, 2, 1)?;
    let executable = graph.compile(&[window], &[], None)?;
    let position_data = executable.input(position)?.allocate()?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let feeds = [&position_data];
    for (row, value) in [(3i32, -100i8), (7, 100)] {
        position_data.write(&[row])?;
        executable.run(&feeds, Some(&results), None)?;
        let view = outputs[0].read::<i8>()?;
        assert!(
            view.iter().all(|&v| v == value),
            "row {row}, first window bytes {:?}",
            &view[..8]
        );
    }
    Ok(())
}

#[test]
fn palette_bit_widths_and_groups() -> Result<(), ane::Error> {
    for bits in [1usize, 2, 3, 4, 6, 8] {
        let entries = 1usize << bits;
        let palette = (0..entries)
            .map(|v| (v % 7) as f32 * 0.125)
            .collect::<Vec<_>>();
        let mut packed = vec![0u8; (16384 * bits).div_ceil(8)];
        for index in 0..16384 {
            let code = ((index / 128 * 5 + index % 128 * 3) % entries) as u16;
            let bit = index * bits;
            let code = code << (bit % 8);
            packed[bit / 8] |= code as u8;
            if bit % 8 + bits > 8 {
                packed[bit / 8 + 1] |= (code >> 8) as u8;
            }
        }
        for transpose in [false, true] {
            println!("native {bits}-bit palette, transpose={transpose}");
            let graph = Graph::new();
            let input = graph.placeholder([128, 128], DataType::Float16)?;
            let weights = graph.palettized_weights(&packed, bits, [128, 128], &palette)?;
            let output = graph.matrix_multiplication(&input, &weights, false, transpose)?;
            let executable = graph.compile(&[output], &[], None)?;
            let data = executable.input(input)?.allocate()?;
            data.copy_from_f32(
                &(0..16384)
                    .map(|i| if i / 128 == i % 128 { 1. } else { 0. })
                    .collect::<Vec<_>>(),
            )?;
            let result = executable.run(&[&data], None, None)?;
            for (index, &actual) in result[0].read_f32()?.iter().enumerate() {
                let (row, column) = if transpose {
                    (index % 128, index / 128)
                } else {
                    (index / 128, index % 128)
                };
                assert_eq!(
                    actual,
                    palette[(row * 5 + column * 3) % entries],
                    "{bits}-bit weight at {index}, transpose={transpose}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn grouped_palette() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([256, 256], DataType::Float16)?;
    let mut palette = (0..16).map(|v| v as f32 * 0.125).collect::<Vec<_>>();
    palette.extend((0..16).map(|v| v as f32 * 0.25));
    let weights =
        graph.palettized_weights_with_groups(&[0x10; 16384], 4, [256, 128], [2, 1], &palette)?;
    let output = graph.matrix_multiplication(&input, &weights, false, false)?;
    let executable = graph.compile(&[output], &[], None)?;
    let data = executable.input(input)?.allocate()?;
    data.copy_from_f32(
        &(0..65536)
            .map(|i| if i / 256 == i % 256 { 1. } else { 0. })
            .collect::<Vec<_>>(),
    )?;
    let result = executable.run(&[&data], None, None)?;
    assert!(
        result[0]
            .read_f32()?
            .iter()
            .enumerate()
            .all(|(i, &v)| v == (i % 2) as f32 * if i / 128 < 128 { 0.125 } else { 0.25 })
    );
    Ok(())
}

fn evaluate(
    graph: &Graph,
    feeds: &[(Tensor, &[f32])],
    output: Tensor,
) -> Result<Vec<f32>, ane::Error> {
    let executable = graph.compile(&[output], &[], None)?;
    let data = executable
        .input_tensors()
        .iter()
        .map(|tensor| {
            let (_, values) = feeds.iter().find(|(t, _)| t == tensor).unwrap();
            let data = executable.input(*tensor)?.allocate()?;
            data.copy_from_f32(values)?;
            Ok(data)
        })
        .collect::<Result<Vec<_>, ane::Error>>()?;
    let inputs: Vec<_> = data.iter().collect();
    Ok(executable.run(&inputs, None, None)?[0].read_f32()?.into())
}

#[test]
fn bilinear_resize_uses_half_pixel_centers() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([1, 1, 2, 2], DataType::Float32)?;
    let output = graph.resize(&input, [4, 4], SamplingMode::Bilinear)?;
    let actual = evaluate(&graph, &[(input, &[0., 1., 2., 3.])], output)?;
    let axis = [0., 0.25, 0.75, 1.];
    let expected: Vec<f32> = (0..16).map(|i| 2. * axis[i / 4] + axis[i % 4]).collect();
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn boolean_attention_mask_excludes_false_positions() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let query = graph.placeholder([1, 1, 2, 4], DataType::Float32)?;
    let key = graph.placeholder([1, 1, 2, 4], DataType::Float32)?;
    let value = graph.placeholder([1, 1, 2, 4], DataType::Float32)?;
    let keep = graph.placeholder([1, 1, 2, 2], DataType::Float32)?;
    let mask = graph.cast(&keep, DataType::Bool)?;
    let output = graph.scaled_dot_product_attention(&query, &key, &value, Some(&mask))?;
    let actual = evaluate(
        &graph,
        &[
            (query, &[1.; 8]),
            (key, &[1.; 8]),
            (value, &[0., 0., 0., 0., 10., 10., 10., 10.]),
            (keep, &[1., 0., 1., 1.]),
        ],
        output,
    )?;
    assert_eq!(actual, [0., 0., 0., 0., 5., 5., 5., 5.]);
    Ok(())
}

#[test]
fn crop_resize_samples_aligned_box_corners() -> Result<(), ane::Error> {
    let image: Vec<f32> = (0..32)
        .map(|i| (i % 16) as f32 + 100. * (i / 16) as f32)
        .collect();
    let full = [0., 1.5, 3.];
    for (normalized, corners, rows, columns) in [
        (
            true,
            [0.25, 0., 1., 0.5, 0., 0., 1., 1.],
            [0., 0.75, 1.5],
            [0.75, 1.875, 3.],
        ),
        (
            false,
            [1., 0., 3., 2., 0., 0., 3., 3.],
            [0., 1., 2.],
            [1., 2., 3.],
        ),
    ] {
        let graph = Graph::new();
        let input = graph.placeholder([2, 1, 4, 4], DataType::Float32)?;
        let boxes = graph.placeholder([2, 4], DataType::Float32)?;
        let selection = graph.placeholder([2], DataType::Float32)?;
        let indices = graph.cast(&selection, DataType::UInt16)?;
        let output = graph.crop_resize(&input, &boxes, &indices, [3, 3], normalized)?;
        let actual = evaluate(
            &graph,
            &[(input, &image), (boxes, &corners), (selection, &[1., 0.])],
            output,
        )?;
        let expected = (0..18).map(|i| match (i / 9, i / 3 % 3, i % 3) {
            (0, r, c) => 100. + 4. * rows[r] + columns[c],
            (_, r, c) => 4. * full[r] + full[c],
        });
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 0.05,
                "{normalized}: {actual} vs {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn runtime_pointwise_convolution_adds_bias() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([1, 2, 1, 2], DataType::Float32)?;
    let weights = graph.placeholder([2, 2, 1, 1], DataType::Float16)?;
    let bias = graph.constant(&[10., 20.], [2])?;
    let output = graph.convolution_2d_1x1(&input, &weights, Some(&bias))?;
    let actual = evaluate(
        &graph,
        &[(input, &[1., 2., 3., 4.]), (weights, &[1., 0., 1., 1.])],
        output,
    )?;
    assert_eq!(actual, [11., 12., 24., 26.]);
    Ok(())
}

#[test]
fn round_breaks_ties_away_from_zero() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([8], DataType::Float32)?;
    let output = graph.round(&input)?;
    let actual = evaluate(
        &graph,
        &[(input, &[0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.5, 0.49])],
        output,
    )?;
    assert_eq!(actual, [1., 2., 3., -1., -2., -3., 4., 0.]);
    Ok(())
}

#[test]
fn normalizations_fold_epsilon_into_reciprocal_square_root() -> Result<(), ane::Error> {
    let values = [1., 2., 3., 4., -1., 0., 1., 6.];
    for rms in [false, true] {
        let graph = Graph::new();
        let input = graph.placeholder([2, 4], DataType::Float32)?;
        let scale = graph.constant(&[1., 2., 1., 0.5], [4])?;
        let output = if rms {
            graph.rms_norm(&input, &[-1], &scale, 1e-3)?
        } else {
            graph.layer_norm(&input, &[-1], &scale, None, 1e-3)?
        };
        let actual = evaluate(&graph, &[(input, &values)], output)?;
        for (row, actual) in values.chunks(4).zip(actual.chunks(4)) {
            let mean = if rms {
                0.
            } else {
                row.iter().sum::<f32>() / 4.
            };
            let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.;
            for ((value, actual), scale) in row.iter().zip(actual).zip([1., 2., 1., 0.5]) {
                let expected = (value - mean) / (variance + 1e-3).sqrt() * scale;
                assert!(
                    (actual - expected).abs() < 1e-2,
                    "{rms}: {actual} vs {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn padding_modes_and_mirrored_width_limits() -> Result<(), ane::Error> {
    for (mode, width, expected) in [
        (
            PadFillMode::Reflect,
            3,
            Some(vec![4., 3., 2., 1., 2., 3., 4., 3., 2., 1.]),
        ),
        (PadFillMode::Reflect, 4, None),
        (
            PadFillMode::Symmetric,
            4,
            Some(vec![4., 3., 2., 1., 1., 2., 3., 4., 4., 3., 2., 1.]),
        ),
        (PadFillMode::Symmetric, 5, None),
        (
            PadFillMode::Replicate,
            3,
            Some(vec![1., 1., 1., 1., 2., 3., 4., 4., 4., 4.]),
        ),
    ] {
        let graph = Graph::new();
        let input = graph.placeholder([1, 1, 1, 4], DataType::Float32)?;
        let output = graph.pad(&input, 0, 0, width, width, mode, 0.0);
        match expected {
            Some(expected) => {
                assert_eq!(
                    evaluate(&graph, &[(input, &[1., 2., 3., 4.])], output?)?,
                    expected
                )
            }
            None => assert!(matches!(output, Err(ane::GraphError::OutOfBounds(_)))),
        }
    }
    Ok(())
}

#[test]
fn matrix_multiplication_treats_vectors_like_numpy() -> Result<(), ane::Error> {
    let matrix = [1., 2., 3., 4., 5., 6.];
    let vector = [1., 0., 1.];
    let graph = Graph::new();
    let x = graph.placeholder([2, 3], DataType::Float32)?;
    let y = graph.placeholder([3], DataType::Float32)?;
    let column = graph.matrix_multiplication(&x, &y, false, false)?;
    assert_eq!(column.shape(), [2]);
    assert_eq!(
        evaluate(&graph, &[(x, &matrix), (y, &vector)], column)?,
        [4., 10.]
    );
    let graph = Graph::new();
    let x = graph.placeholder([2], DataType::Float32)?;
    let y = graph.placeholder([2, 3], DataType::Float32)?;
    let row = graph.matrix_multiplication(&x, &y, false, false)?;
    assert_eq!(row.shape(), [3]);
    assert_eq!(
        evaluate(&graph, &[(x, &[1., 1.]), (y, &matrix)], row)?,
        [5., 7., 9.]
    );
    let graph = Graph::new();
    let x = graph.placeholder([3], DataType::Float32)?;
    let y = graph.placeholder([3], DataType::Float32)?;
    let dot = graph.matrix_multiplication(&x, &y, false, false)?;
    assert!(dot.shape().is_empty());
    assert_eq!(
        evaluate(&graph, &[(x, &[1., 2., 3.]), (y, &vector)], dot)?,
        [4.]
    );
    Ok(())
}

#[test]
fn sixteen_bit_casts_come_from_bytes_not_floats() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let floats = graph.placeholder([4], DataType::Float16)?;
    let narrowed = graph.cast(&floats, DataType::Int16)?;
    assert!(matches!(
        graph.compile(&[narrowed], &[], None),
        Err(ane::Error::Graph(ane::GraphError::UnsupportedComposition(
            _
        )))
    ));
    let graph = Graph::new();
    let bytes = graph.placeholder([8], DataType::Int8)?;
    let wide = graph.cast(&bytes, DataType::Int16)?;
    let executable = graph.compile(&[wide], &[], None)?;
    let data = executable.input(bytes)?.allocate()?;
    data.write(&[-128i8, -3, 0, 1, 7, 100, 126, 127])?;
    let result = executable.run(&[&data], None, None)?;
    assert_eq!(
        &*result[0].read::<i16>()?,
        [-128, -3, 0, 1, 7, 100, 126, 127]
    );
    Ok(())
}

#[test]
fn boolean_inputs_and_outputs_cross_the_ane_boundary_as_bytes() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let flags = graph.placeholder([8], DataType::Bool)?;
    let inverted = graph.not(&flags)?;
    let executable = graph.compile(&[inverted], &[], None)?;
    let data = executable.input(flags)?.allocate()?;
    let values = [true, false, false, true, true, true, false, false];
    data.write(&values)?;
    let result = executable.run(&[&data], None, None)?;
    let expected: Vec<_> = values.iter().map(|v| !v).collect();
    assert_eq!(&*result[0].read::<bool>()?, expected);
    Ok(())
}

#[test]
fn slice_update_places_on_ane() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([4, 4], DataType::Float32)?;
    let update = graph.placeholder([2, 2], DataType::Float32)?;
    let output = graph.slice_update(&input, &update, &[1, 2])?;
    let actual = evaluate(
        &graph,
        &[(input, &[0.; 16]), (update, &[1., 2., 3., 4.])],
        output,
    )?;
    let mut expected = [0.; 16];
    expected[6] = 1.;
    expected[7] = 2.;
    expected[10] = 3.;
    expected[11] = 4.;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn space_to_batch_round_trips_through_batch_to_space() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([2, 3, 4, 6], DataType::Float32)?;
    let batched = graph.space_to_batch(&input, [2, 2], [0, 2, 1, 1])?;
    assert_eq!(batched.shape(), [8, 3, 3, 4]);
    let restored = graph.batch_to_space(&batched, [2, 2], [0, 2, 1, 1])?;
    let values: Vec<f32> = (0..144).map(|v| v as f32).collect();
    let executable = graph.compile(&[batched, restored], &[], None)?;
    let data = executable.input(input)?.allocate()?;
    data.copy_from_f32(&values)?;
    let result = executable.run(&[&data], None, None)?;
    assert_eq!(&*result[1].read_f32()?, &values[..]);
    let batched = result[0].read_f32()?;
    for (index, &value) in batched.iter().enumerate() {
        let (block, channel, row, column) = (index / 36, index / 12 % 3, index / 4 % 3, index % 4);
        let (offset, batch) = (block / 2, block % 2);
        let (y, x) = (row * 2 + offset / 2, column * 2 + offset % 2);
        let source = if (1..7).contains(&x) && y < 4 {
            ((batch * 3 + channel) * 4 + y) * 6 + (x - 1)
        } else {
            usize::MAX
        };
        let expected = values.get(source).copied().unwrap_or(0.);
        assert_eq!(value, expected, "element {index}");
    }
    Ok(())
}

#[test]
fn logical_operations_follow_truth_tables() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let x = graph.placeholder([4], DataType::Bool)?;
    let y = graph.placeholder([4], DataType::Bool)?;
    let and = graph.logical_and(&x, &y)?;
    let or = graph.logical_or(&x, &y)?;
    let xor = graph.logical_xor(&x, &y)?;
    let executable = graph.compile(&[and, or, xor], &[], None)?;
    let left = executable.input(x)?.allocate()?;
    left.write(&[false, false, true, true])?;
    let right = executable.input(y)?.allocate()?;
    right.write(&[false, true, false, true])?;
    let result = executable.run(&[&left, &right], None, None)?;
    assert_eq!(&*result[0].read::<bool>()?, [false, false, false, true]);
    assert_eq!(&*result[1].read::<bool>()?, [false, true, true, true]);
    assert_eq!(&*result[2].read::<bool>()?, [false, true, true, false]);
    Ok(())
}

#[test]
fn band_part_and_one_hot_build_masks_on_ane() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([1, 2, 4, 4], DataType::Float32)?;
    let causal = graph.band_part(&input, -1, 0)?;
    let banded = graph.band_part(&input, 1, 1)?;
    let positions = graph.placeholder([3], DataType::Float32)?;
    let indices = graph.cast(&positions, DataType::UInt16)?;
    let hot = graph.one_hot(&indices, 5)?;
    assert_eq!(hot.shape(), [3, 5]);
    let executable = graph.compile(&[causal, banded, hot], &[], None)?;
    let data = executable.input(input)?.allocate()?;
    let values: Vec<f32> = (1..=32).map(|v| v as f32).collect();
    data.copy_from_f32(&values)?;
    let position_data = executable.input(positions)?.allocate()?;
    position_data.copy_from_f32(&[4., 0., 2.])?;
    let result = executable.run(&[&data, &position_data], None, None)?;
    let keep = |lower: i64, upper: i64| -> Vec<f32> {
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let (row, column) = ((i / 4 % 4) as i64, (i % 4) as i64);
                let inside =
                    (lower < 0 || row - column <= lower) && (upper < 0 || column - row <= upper);
                if inside { v } else { 0. }
            })
            .collect()
    };
    assert_eq!(&*result[0].read_f32()?, keep(-1, 0));
    assert_eq!(&*result[1].read_f32()?, keep(1, 1));
    let mut expected = [0.; 15];
    for (row, class) in [4, 0, 2].into_iter().enumerate() {
        expected[row * 5 + class] = 1.;
    }
    assert_eq!(&*result[2].read_f32()?, expected);
    Ok(())
}

#[test]
fn multi_axis_reductions_lower_to_one_operation() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([2, 3, 4], DataType::Float32)?;
    let mean = graph.mean(&input, &[1, 2])?;
    let sum = graph.sum(&input, &[0, -1])?;
    let program = graph.program(&[mean, sum], &[DataType::Float32])?;
    assert_eq!(program.operations(), ["reduce_mean", "reduce_sum"]);
    let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let executable = graph.compile(&[mean, sum], &[], None)?;
    let data = executable.input(input)?.allocate()?;
    data.copy_from_f32(&values)?;
    let result = executable.run(&[&data], None, None)?;
    assert_eq!(&*result[0].read_f32()?, [5.5, 17.5]);
    let expected: Vec<f32> = (0..3)
        .map(|row| {
            (0..2)
                .flat_map(|b| (0..4).map(move |c| (b * 12 + row * 4 + c) as f32))
                .sum()
        })
        .collect();
    assert_eq!(&*result[1].read_f32()?, expected);
    Ok(())
}

#[test]
fn palettized_convolution_weights_stay_packed() -> Result<(), ane::Error> {
    let (outputs, inputs) = (32, 64);
    let codes: Vec<u8> = (0..outputs * inputs / 2)
        .map(|i| (i * 37 % 251) as u8)
        .collect();
    let palette: Vec<f32> = (0..16).map(|i| 0.5 * (i as f32 - 8.)).collect();
    let graph = Graph::new();
    let input = graph.placeholder([1, inputs, 1, 1], DataType::Float32)?;
    let weights = graph.palettized_weights(&codes, 4, [outputs, inputs, 1, 1], &palette)?;
    let output =
        graph.convolution_2d(&input, &weights, None, &Convolution2dDescriptor::default())?;
    let mlir = graph.program(&[output], &[DataType::Float32])?.mlir()?;
    assert!(mlir.contains("tensor<32x64x1x1xui4>"));
    let values: Vec<f32> = (0..inputs).map(|i| (i % 7) as f32 * 0.25 - 0.75).collect();
    let actual = evaluate(&graph, &[(input, &values)], output)?;
    for (row, actual) in actual.iter().enumerate() {
        let expected: f32 = (0..inputs)
            .map(|column| {
                let byte = codes[(row * inputs + column) / 2];
                let code = if column % 2 == 0 {
                    byte & 15
                } else {
                    byte >> 4
                };
                palette[code as usize] * values[column]
            })
            .sum();
        assert!(
            (actual - expected).abs() < 0.02,
            "{row}: {actual} vs {expected}"
        );
    }
    Ok(())
}

#[test]
fn int8_weights_and_activations_with_channel_scales() -> Result<(), ane::Error> {
    let (outputs, inputs, tokens) = (32, 64, 16);
    let (input_scale, weight_scale) = (1. / 64., 1. / 128.);
    let codes: Vec<i8> = (0..outputs * inputs)
        .map(|i| ((i * 37 % 255) as i32 - 127) as i8)
        .collect();
    let channel_scales: Vec<f32> = (0..outputs).map(|o| 0.5 + o as f32 * 0.03).collect();
    let graph = Graph::new();
    let input = graph.placeholder([1, inputs, 1, tokens], DataType::Float32)?;
    let activations = graph.quantize(&input, &[input_scale], None, None, DataType::Int8)?;
    let activations = graph.dequantize(&activations, &[input_scale], None, None)?;
    let bytes: Vec<u8> = codes.iter().map(|&code| code as u8).collect();
    let weights = graph.constant_with_bytes(&bytes, [outputs, inputs, 1, 1], DataType::Int8)?;
    let weights = graph.dequantize(&weights, &[weight_scale], None, None)?;
    let output = graph.convolution_2d(
        &activations,
        &weights,
        None,
        &Convolution2dDescriptor::default(),
    )?;
    let scales = graph.constant(&channel_scales, [1, outputs, 1, 1])?;
    let output = graph.multiplication(&output, &scales)?;
    let mlir = graph.program(&[output], &[DataType::Float32])?.mlir()?;
    assert!(mlir.contains("tensor<32x64x1x1xsi8>"));
    let quantized: Vec<i32> = (0..inputs * tokens)
        .map(|i| (i * 13 % 201) as i32 - 100)
        .collect();
    let values: Vec<f32> = quantized.iter().map(|&q| q as f32 * input_scale).collect();
    let actual = evaluate(&graph, &[(input, &values)], output)?;
    for row in 0..outputs {
        for token in 0..tokens {
            let expected = channel_scales[row]
                * (0..inputs)
                    .map(|column| {
                        (quantized[column * tokens + token] * codes[row * inputs + column] as i32)
                            as f32
                    })
                    .sum::<f32>()
                * input_scale
                * weight_scale;
            let actual = actual[row * tokens + token];
            assert!(
                (actual - expected).abs() <= 0.01 * expected.abs().max(1.),
                "{row},{token}: {actual} vs {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn identical_constants_compile_once() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([256, 256], DataType::Float32)?;
    let weights: Vec<f32> = (0..65536)
        .map(|i| if i / 256 == i % 256 { 0.5 } else { 0. })
        .collect();
    let first = graph.constant(&weights, [256, 256])?;
    let second = graph.constant(&weights, [256, 256])?;
    let hidden = graph.matrix_multiplication(&input, &first, false, false)?;
    let output = graph.matrix_multiplication(&hidden, &second, false, false)?;
    let program = graph.program(&[output], &[DataType::Float32])?;
    assert_eq!(program.constant_bytes(), 65536 * 2);
    assert_eq!(program.operations(), ["constant", "matmul", "matmul"]);
    let values: Vec<f32> = (0..65536).map(|i| (i % 7) as f32).collect();
    let actual = evaluate(&graph, &[(input, &values)], output)?;
    assert!(actual.iter().zip(&values).all(|(a, v)| *a == v * 0.25));
    Ok(())
}
