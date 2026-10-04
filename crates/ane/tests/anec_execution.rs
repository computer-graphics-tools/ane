use ane::{
    BlockwiseQuantization, CompilationDescriptor, Convolution2dDescriptor, DataType, Graph,
    PadFillMode, PadMode, Palettization, Pooling2dDescriptor, ResizeSamplingMode, Tensor,
    TensorData, WeightDataType,
};
use half::f16;
use std::future::Future;

#[test]
fn separately_compiled_graphs_keep_independent_state() -> Result<(), ane::Error> {
    let mut executables = Vec::new();
    for initial in [0.0, 10.0] {
        let graph = Graph::new();
        let input = graph.placeholder([64, 64], DataType::Float32)?;
        let state = graph.variable_with_data(&vec![initial; 4096], [64, 64])?;
        let old = graph.read_variable(&state)?;
        let next = graph.addition(&old, &input)?;
        let update = graph.assign_variable(&state, &next)?;
        let output = graph.square(&input)?;
        let executable = graph.compile(&[output], &[&update], None)?;
        executables.push((executable, input, state));
    }
    for index in [0, 1, 0] {
        let (executable, input, _) = &executables[index];
        let data = executable.input(*input)?.allocate()?;
        data.copy_from_f32(&[1.0; 4096])?;
        let result = executable.run(&[&data], None, None)?;
        assert!(result[0].read_f32()?.iter().all(|&v| v == 1.0));
    }
    for ((executable, _, state), expected) in executables.iter().zip([2.0, 11.0]) {
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
    let assign = graph.assign_variable(&state, &next)?;
    let result = graph.read_variable(&state)?;
    let executable = graph.compile(&[result], &[&assign], None)?;
    assert!(executable.report().mil_bytes > 0);
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
    let assign = graph.assign_variable_rows(&state, &input, &position, 1)?;
    let result = graph.read_variable(&state)?;
    let executable = graph.compile(&[result], &[&assign], None)?;
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
    let output = graph.gather(&input, &indices, 1)?;
    let executable = graph.compile(&[output], &[], None)?;
    let input_data = executable.input(input)?.allocate()?;
    let index_data = executable.input(indices)?.allocate()?;
    input_data.copy_from_f32(&(0..4096).map(|i| (i % 64) as f32).collect::<Vec<_>>())?;
    index_data.write(&(0..64).rev().map(|i| i as u16).collect::<Vec<_>>())?;
    let result = executable.run(&[&input_data, &index_data], None, None)?;
    assert!(
        result[0]
            .read_f32()?
            .iter()
            .enumerate()
            .all(|(i, &v)| v == (63 - i % 64) as f32)
    );
    drop(executable);

    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float16)?;
    let weights = graph.palettized_weights(
        &[0xe4; 1024],
        [64, 64],
        &Palettization::new(2, &[0., 1., 2., 3.]),
    )?;
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

    let build = |initial: f32| -> Result<_, ane::Error> {
        let graph = Graph::new();
        let x = graph.placeholder([64, 64], DataType::Float32)?;
        let w = graph.variable_with_data(&[initial; 4096], [64, 64])?;
        let value = graph.read_variable(&w)?;
        let y = graph.matrix_multiplication(&x, &value, false, false)?;
        Ok((graph.compile(&[y], &[], None)?, x, w))
    };
    let (a, x, wa) = build(1.)?;
    let (b, _, wb) = build(2.)?;
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
    let output = a.allocate_outputs()?.remove(0);
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
fn convolution_rejects_runtime_weights() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([1, 64, 1, 64], DataType::Float16)?;
    let weights = graph.placeholder([64, 64, 1, 1], DataType::Float16)?;
    assert!(matches!(
        graph.convolution_2d(&input, &weights, None, &Convolution2dDescriptor::default()),
        Err(ane::GraphError::NonConstantWeights(_))
    ));
    Ok(())
}

#[test]
fn native_palettized_linear() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([128, 128], DataType::Float16)?;
    let weights = graph.palettized_weights(
        &[0x10; 8192],
        [128, 128],
        &Palettization::new(
            4,
            &[
                0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
            ],
        ),
    )?;
    let output = graph.matrix_multiplication(&input, &weights, false, false)?;
    let mil = graph.program(&[output], &[DataType::Float32])?.mil()?;
    assert!(
        mil.text
            .contains("constexpr_lut_to_dense(indices = tensor<uint4, [1, 1, 128, 128]>")
    );
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
fn byte_casts_round_half_away_from_zero() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float16)?;
    let bytes = graph.cast(&input, DataType::Int8)?;
    let executable = graph.compile(&[bytes], &[], None)?;
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
    data.copy_from_f32(&[1.75; 4096])?;
    executable.run(&feeds, Some(&results), None)?;
    assert!(outputs[0].read::<i8>()?.iter().all(|&v| v == 2));
    Ok(())
}

#[test]
fn byte_state() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let state_data = TensorData::from_slice(&[0i8; 4096], [1, 1, 64, 64])?;
    let state = graph.variable_with_tensor_data(&state_data)?;
    let update = graph.placeholder([1, 1, 1, 64], DataType::Int8)?;
    let position = graph.placeholder([1], DataType::Int32)?;
    let assign = graph.assign_variable_rows(&state, &update, &position, 0)?;
    let output = graph.read_variable(&state)?;
    let executable = graph.compile(&[output], &[&assign], None)?;
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
            let weights = graph.palettized_weights(
                &packed,
                [128, 128],
                &Palettization::new(bits, &palette),
            )?;
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
    let weights = graph.palettized_weights(
        &[0x10; 16384],
        [256, 128],
        &Palettization {
            group_shape: [2, 1],
            ..Palettization::new(4, &palette)
        },
    )?;
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
        .feed_tensors()
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
    let output = graph.resize_bilinear(&input, [4, 4], ResizeSamplingMode::UnalignCorners)?;
    let actual = evaluate(&graph, &[(input, &[0., 1., 2., 3.])], output)?;
    let axis = [0., 0.25, 0.75, 1.];
    let expected: Vec<f32> = (0..16).map(|i| 2. * axis[i / 4] + axis[i % 4]).collect();
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn additive_attention_mask_excludes_masked_positions() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let query = graph.placeholder([1, 1, 2, 4], DataType::Float32)?;
    let key = graph.placeholder([1, 1, 2, 4], DataType::Float32)?;
    let value = graph.placeholder([1, 1, 2, 4], DataType::Float32)?;
    let mask = graph.placeholder([1, 1, 2, 2], DataType::Float32)?;
    let output = graph.scaled_dot_product_attention(&query, &key, &value, Some(&mask))?;
    let actual = evaluate(
        &graph,
        &[
            (query, &[1.; 8]),
            (key, &[1.; 8]),
            (value, &[0., 0., 0., 0., 10., 10., 10., 10.]),
            (mask, &[0., f32::NEG_INFINITY, 0., 0.]),
        ],
        output,
    )?;
    assert_eq!(actual, [0., 0., 0., 0., 5., 5., 5., 5.]);
    Ok(())
}

#[test]
fn pointwise_convolution_adds_bias() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([1, 2, 1, 2], DataType::Float32)?;
    let weights = graph.constant(&[1., 0., 1., 1.], [2, 2, 1, 1])?;
    let bias = graph.constant(&[10., 20.], [2])?;
    let output = graph.convolution_2d(
        &input,
        &weights,
        Some(&bias),
        &Convolution2dDescriptor::default(),
    )?;
    let actual = evaluate(&graph, &[(input, &[1., 2., 3., 4.])], output)?;
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
fn layer_norm_folds_epsilon_into_reciprocal_square_root() -> Result<(), ane::Error> {
    let values = [1., 2., 3., 4., -1., 0., 1., 6.];
    let graph = Graph::new();
    let input = graph.placeholder([2, 4], DataType::Float32)?;
    let scale = graph.constant(&[1., 2., 1., 0.5], [4])?;
    let output = graph.layer_norm(&input, &[-1], Some(&scale), None, 1e-3)?;
    let actual = evaluate(&graph, &[(input, &values)], output)?;
    for (row, actual) in values.chunks(4).zip(actual.chunks(4)) {
        let mean = row.iter().sum::<f32>() / 4.;
        let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.;
        for ((value, actual), scale) in row.iter().zip(actual).zip([1., 2., 1., 0.5]) {
            let expected = (value - mean) / (variance + 1e-3).sqrt() * scale;
            assert!((actual - expected).abs() < 1e-2, "{actual} vs {expected}");
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
            PadFillMode::Replicate,
            3,
            Some(vec![1., 1., 1., 1., 2., 3., 4., 4., 4., 4.]),
        ),
    ] {
        let graph = Graph::new();
        let input = graph.placeholder([1, 1, 1, 4], DataType::Float32)?;
        let output = graph.pad(&input, [0, 0, width, width], mode, 0.0);
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
fn slice_update_places_on_ane() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([1, 2, 4, 4], DataType::Float32)?;
    let update = graph.placeholder([1, 1, 4, 4], DataType::Float32)?;
    let output = graph.slice_update(&input, &update, &[0, 1, 0, 0])?;
    let patch: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let actual = evaluate(&graph, &[(input, &[0.; 32]), (update, &patch)], output)?;
    let mut expected = [0.; 32];
    expected[16..].copy_from_slice(&patch);
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
fn logical_and_crosses_the_ane_boundary_as_bytes() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let x = graph.placeholder([4], DataType::Bool)?;
    let y = graph.placeholder([4], DataType::Bool)?;
    let and = graph.logical_and(&x, &y)?;
    let executable = graph.compile(&[and], &[], None)?;
    let left = executable.input(x)?.allocate()?;
    left.write(&[false, false, true, true])?;
    let right = executable.input(y)?.allocate()?;
    right.write(&[false, true, false, true])?;
    let result = executable.run(&[&left, &right], None, None)?;
    assert_eq!(&*result[0].read::<bool>()?, [false, false, false, true]);
    Ok(())
}

#[test]
fn multi_axis_reductions_lower_to_one_operation() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([2, 3, 4], DataType::Float32)?;
    let mean = graph.reduction_mean(&input, &[1, 2])?;
    let sum = graph.reduction_sum(&input, &[0, -1])?;
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
    let weights = graph.palettized_weights(
        &codes,
        [outputs, inputs, 1, 1],
        &Palettization::new(4, &palette),
    )?;
    let output =
        graph.convolution_2d(&input, &weights, None, &Convolution2dDescriptor::default())?;
    let mil = graph.program(&[output], &[DataType::Float32])?.mil()?;
    assert!(mil.text.contains("indices = tensor<uint4, [32, 64, 1, 1]>"));
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
    let mil = graph.program(&[output], &[DataType::Float32])?.mil()?;
    assert!(mil.text.contains("tensor<int8, [32, 64, 1, 1]>"));
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

#[test]
fn per_channel_constant_dequantization() -> Result<(), ane::Error> {
    let (rows, columns) = (8, 64);
    let codes: Vec<i8> = (0..rows * columns)
        .map(|i| ((i * 37 % 255) as i32 - 127) as i8)
        .collect();
    let scales: Vec<f32> = (0..rows)
        .map(|row| 1. / (1 << (row % 3 + 4)) as f32)
        .collect();
    let zeros: Vec<i32> = (0..rows as i32).map(|row| row - 4).collect();
    let graph = Graph::new();
    let input = graph.placeholder([rows, columns], DataType::Float32)?;
    let bytes: Vec<u8> = codes.iter().map(|&code| code as u8).collect();
    let weights = graph.constant_with_bytes(&bytes, [rows, columns], DataType::Int8)?;
    let weights = graph.dequantize(&weights, &scales, Some(&zeros), Some(0))?;
    let output = graph.multiplication(&weights, &input)?;
    let actual = evaluate(&graph, &[(input, &[1.; 512])], output)?;
    for (index, actual) in actual.iter().enumerate() {
        let row = index / columns;
        let expected = (codes[index] as i32 - zeros[row]) as f32 * scales[row];
        assert_eq!(*actual, expected, "{index}");
    }
    Ok(())
}

#[test]
fn attention_matches_reference_and_guards_float32_inputs() -> Result<(), ane::Error> {
    let (heads, key_heads, rows, depth) = (4, 4, 8, 16);
    let wave = |count: usize, step: usize| -> Vec<f32> {
        (0..count)
            .map(|i| ((i * step % 13) as f32 - 6.) * 0.05)
            .collect()
    };
    let query = wave(heads * rows * depth, 7);
    let key = wave(key_heads * rows * depth, 5);
    let value = wave(key_heads * rows * depth, 3);
    let graph = Graph::new();
    let q = graph.placeholder([1, heads, rows, depth], DataType::Float32)?;
    let k = graph.placeholder([1, key_heads, rows, depth], DataType::Float32)?;
    let v = graph.placeholder([1, key_heads, rows, depth], DataType::Float32)?;
    let output = graph.scaled_dot_product_attention(&q, &k, &v, None)?;
    let half = CompilationDescriptor {
        output_types: Some(vec![DataType::Float16]),
        ..Default::default()
    };
    assert!(matches!(
        graph.compile(&[output], &[], Some(&half)),
        Err(ane::Error::Graph(ane::GraphError::UnsupportedComposition(
            _
        )))
    ));
    let graph = Graph::new();
    let q = graph.placeholder([1, heads, rows, depth], DataType::Float16)?;
    let k = graph.placeholder([1, key_heads, rows, depth], DataType::Float16)?;
    let v = graph.placeholder([1, key_heads, rows, depth], DataType::Float16)?;
    let output = graph.scaled_dot_product_attention(&q, &k, &v, None)?;
    let executable = graph.compile(&[output], &[], Some(&half))?;
    let data = executable
        .feed_tensors()
        .iter()
        .map(|tensor| {
            let data = executable.input(*tensor)?.allocate()?;
            let values = [(q, &query), (k, &key), (v, &value)]
                .into_iter()
                .find(|(t, _)| t == tensor)
                .unwrap()
                .1;
            data.copy_from_f32(values)?;
            Ok(data)
        })
        .collect::<Result<Vec<_>, ane::Error>>()?;
    let inputs: Vec<_> = data.iter().collect();
    let actual = executable.run(&inputs, None, None)?[0].read_f32()?;
    for head in 0..heads {
        let shared = head / (heads / key_heads);
        for row in 0..rows {
            let scores: Vec<f32> = (0..rows)
                .map(|column| {
                    (0..depth)
                        .map(|c| {
                            query[(head * rows + row) * depth + c]
                                * key[(shared * rows + column) * depth + c]
                        })
                        .sum::<f32>()
                        / (depth as f32).sqrt()
                })
                .collect();
            let peak = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
            let weights: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
            let total: f32 = weights.iter().sum();
            for c in 0..depth {
                let expected: f32 = (0..rows)
                    .map(|column| {
                        weights[column] / total * value[(shared * rows + column) * depth + c]
                    })
                    .sum();
                let actual = actual[(head * rows + row) * depth + c];
                assert!(
                    (actual - expected).abs() < 2e-3,
                    "{head},{row},{c}: {actual} vs {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn pooling_and_index_lookups_match_reference_semantics() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([1, 1, 5, 5], DataType::Float32)?;
    let mut descriptor = Pooling2dDescriptor::new([2, 2], [2, 2]);
    descriptor.padding = [1, 1, 1, 1];
    let pooled = graph.max_pooling_2d(&input, &descriptor)?;
    let image: Vec<f32> = (0..25).map(|v| v as f32).collect();
    let expected: Vec<f32> = (0..3)
        .flat_map(|i| (0..3).map(move |j| ((2 * i).min(4) * 5 + (2 * j).min(4)) as f32))
        .collect();
    assert_eq!(evaluate(&graph, &[(input, &image)], pooled)?, expected);

    let graph = Graph::new();
    let input = graph.placeholder([1, 1, 4, 4], DataType::Float32)?;
    let mut corner = [0.; 9];
    corner[0] = 1.;
    let weights = graph.constant(&corner, [1, 1, 3, 3])?;
    let descriptor = Convolution2dDescriptor {
        strides: [2, 2],
        pad_mode: PadMode::Same,
        ..Default::default()
    };
    let same = graph.convolution_2d(&input, &weights, None, &descriptor)?;
    let image: Vec<f32> = (0..16).map(|v| v as f32).collect();
    assert_eq!(
        evaluate(&graph, &[(input, &image)], same)?,
        [0., 2., 8., 10.]
    );

    let graph = Graph::new();
    let input = graph.placeholder([2, 4], DataType::Float32)?;
    let positions = graph.placeholder([3], DataType::Float32)?;
    let indices = graph.cast(&positions, DataType::UInt16)?;
    let lookup = graph.gather(&input, &indices, -1)?;
    let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
    assert_eq!(
        evaluate(
            &graph,
            &[(input, &values), (positions, &[3., 0., 2.])],
            lookup
        )?,
        [3., 0., 2., 7., 4., 6.]
    );
    Ok(())
}

#[test]
fn submissions_keep_their_own_results() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float32)?;
    let one = graph.constant(&[1.], [1])?;
    let output = graph.addition(&input, &one)?;
    let executable = graph.compile(&[output], &[], None)?;
    let data = executable.input(input)?.allocate()?;
    data.copy_from_f32(&[1.; 4096])?;
    let outputs = executable.allocate_outputs()?;
    let results: Vec<_> = outputs.iter().collect();
    let first = executable.run_async(&[&data], Some(&results), None)?;
    while !first.wait_timeout(std::time::Duration::from_secs(1)) {}
    executable.run(&[&data], Some(&results), None)?;
    assert!(first.is_finished());
    first.wait()?;
    let first = executable.run_async(&[&data], Some(&results), None)?;
    while !first.is_finished() {
        std::thread::yield_now();
    }
    data.copy_from_f32(&[2.; 4096])?;
    let mut second = executable.run_async(&[&data], Some(&results), None)?;
    first.wait()?;
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    let values = loop {
        if let std::task::Poll::Ready(values) = std::pin::Pin::new(&mut second).poll(&mut context) {
            break values?;
        }
        std::thread::yield_now();
    };
    assert!(values[0].read_f32()?.iter().all(|&v| v == 3.));
    Ok(())
}

#[test]
fn sparse_compressed_weights_decode_on_ane() -> Result<(), ane::Error> {
    let (rows, columns) = (8, 16);
    let count = rows * columns;
    let mask: Vec<u8> = (0..count / 8).map(|i| [0x5a, 0xc3, 0x0f][i % 3]).collect();
    let set: Vec<usize> = (0..count)
        .filter(|&i| mask[i / 8] & (1 << (i % 8)) != 0)
        .collect();
    let palette: Vec<f32> = (0..16).map(|v| (v as f32 - 8.) * 0.25).collect();
    let codes: Vec<u8> = (0..set.len()).map(|i| (i * 7 % 16) as u8).collect();
    let packed: Vec<u8> = codes.chunks(2).map(|p| p[0] | (p[1] << 4)).collect();
    let bytes: Vec<i8> = (0..set.len()).map(|i| (i * 37 % 255) as i8).collect();
    let scales: Vec<f32> = (0..rows)
        .map(|row| 1. / (1 << (row % 3 + 3)) as f32)
        .collect();
    let ones = vec![1.; count];
    for palettized in [true, false] {
        let graph = Graph::new();
        let input = graph.placeholder([rows, columns], DataType::Float32)?;
        let weights = if palettized {
            graph.sparse_palettized_weights(&packed, &mask, 4, [rows, columns], &palette)?
        } else {
            let data: Vec<u8> = bytes.iter().map(|&v| v as u8).collect();
            graph.sparse_blockwise_weights(
                &data,
                &mask,
                [rows, columns],
                &ane::BlockwiseQuantization {
                    data_type: WeightDataType::Int8,
                    scales: &scales,
                    scale_shape: [rows, 1],
                    offsets: None,
                    zero_points: None,
                },
            )?
        };
        let output = graph.multiplication(&weights, &input)?;
        let actual = evaluate(&graph, &[(input, &ones)], output)?;
        let mut expected = vec![0.; count];
        for (slot, &index) in set.iter().enumerate() {
            expected[index] = if palettized {
                palette[codes[slot] as usize]
            } else {
                bytes[slot] as f32 * scales[index / columns]
            };
        }
        assert_eq!(actual, expected, "palettized {palettized}");
    }
    Ok(())
}

#[test]
fn shared_executables_share_one_model_weights_and_variable() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let half_identity: Vec<f32> = (0..4096)
        .map(|i| if i / 64 == i % 64 { 0.5 } else { 0. })
        .collect();
    let weights = graph.constant(&half_identity, [64, 64])?;
    let state = graph.variable_with_data(&[0.; 4096], [64, 64])?;
    let input = graph.placeholder([64, 64], DataType::Float16)?;
    let stored = graph.read_variable(&state)?;
    let product = graph.matrix_multiplication(&input, &weights, false, false)?;
    let accumulate = graph.assign_variable(&state, &graph.addition(&stored, &product)?)?;
    let accumulated = graph.read_variable(&state)?;
    let current = graph.read_variable(&state)?;
    let projected = graph.matrix_multiplication(&current, &weights, false, false)?;
    let one = graph.constant(&[1.], [1])?;
    let increment = graph.assign_variable(&state, &graph.addition(&current, &one)?)?;
    let incremented = graph.read_variable(&state)?;
    let mut shared = graph.compile_shared(
        &[
            (&[accumulated], &[&accumulate]),
            (&[projected], &[]),
            (&[incremented], &[&increment]),
        ],
        None,
    )?;
    let (increment, project, accumulate) = (
        shared.pop().unwrap(),
        shared.pop().unwrap(),
        shared.pop().unwrap(),
    );
    assert!(accumulate.report().mil_bytes < 2 * 4096 * 2);
    assert!(project.feed_tensors().is_empty());
    let ones = accumulate.input(input)?.allocate()?;
    ones.copy_from_f32(&[1.; 4096])?;
    let all = |data: &TensorData, expected: f32| -> Result<bool, ane::Error> {
        Ok(data.read_f32()?.iter().all(|&v| v == expected))
    };
    assert!(all(&project.run(&[], None, None)?[0], 0.)?);
    for expected in [0.5, 1.] {
        assert!(all(&accumulate.run(&[&ones], None, None)?[0], expected)?);
    }
    assert!(all(&increment.run(&[], None, None)?[0], 2.)?);
    assert!(all(&project.run(&[], None, None)?[0], 1.)?);
    assert!(all(project.variable_data(&state)?, 2.)?);
    Ok(())
}

fn fusion_weights(transposed: bool) -> Vec<f32> {
    (0..256 * 256)
        .map(|i| {
            if transposed {
                i % 256 * 256 + i / 256
            } else {
                i
            }
        })
        .map(|i| ((i * 31 % 17) as f32 - 8.) / 256.)
        .collect()
}

#[test]
fn compiler_fuses_composed_ops_into_native_results() -> Result<(), ane::Error> {
    type Build = fn(&Graph, &Tensor) -> Result<Tensor, ane::GraphError>;
    let cases: [(&str, Build, Build); 5] = [
        (
            "layer_norm",
            |g, x| {
                let mean = g.reduction_mean(x, &[-1])?;
                let centered = g.subtraction(x, &mean)?;
                let squared = g.multiplication(&centered, &centered)?;
                let variance = g.reduction_mean(&squared, &[-1])?;
                g.multiplication(&centered, &g.reciprocal_square_root(&variance, 1e-5)?)
            },
            |g, x| g.layer_norm(x, &[-1], None, None, 1e-5),
        ),
        (
            "silu",
            |g, x| g.multiplication(x, &g.sigmoid(x)?),
            |g, x| g.silu(x),
        ),
        (
            "leaky_relu",
            |g, x| g.maximum(x, &g.multiplication(x, &g.constant(&[0.1], [1])?)?),
            |g, x| g.leaky_relu(x, 0.1),
        ),
        (
            "linear",
            |g, x| {
                let w = g.constant(&fusion_weights(true), [256, 256])?;
                let bias = g.constant(&[0.25; 256], [256])?;
                g.addition(&g.matrix_multiplication(x, &w, false, false)?, &bias)
            },
            |g, x| {
                let w = g.constant(&fusion_weights(false), [256, 256])?;
                let bias = g.constant(&[0.25; 256], [256])?;
                g.linear(x, &w, Some(&bias))
            },
        ),
        (
            "transposed_matmul",
            |g, x| {
                let w = g.constant(&fusion_weights(false), [256, 256])?;
                g.matrix_multiplication(x, &g.transpose(&w, [1, 0])?, false, false)
            },
            |g, x| {
                let w = g.constant(&fusion_weights(false), [256, 256])?;
                g.matrix_multiplication(x, &w, false, true)
            },
        ),
    ];
    let values: Vec<f32> = (0..64 * 256)
        .map(|i| ((i * 7919 % 1000) as f32 / 500. - 1.) * 4. + (i / 256) as f32 * 0.05)
        .collect();
    for (name, composed, native) in cases {
        let [composed, native] = [composed, native].map(|build| {
            let graph = Graph::new();
            let x = graph.placeholder([64, 256], DataType::Float32)?;
            let y = build(&graph, &x)?;
            evaluate(&graph, &[(x, &values)], y)
        });
        let (composed, native) = (composed?, native?);
        assert!(
            composed
                .iter()
                .zip(&native)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: the ANE compiler no longer fuses the composed form"
        );
    }
    Ok(())
}

fn pack(codes: &[i32], bits: usize) -> Vec<u8> {
    let mut packed = vec![0u8; (codes.len() * bits).div_ceil(8)];
    for (index, &code) in codes.iter().enumerate() {
        let bit = index * bits;
        let code = (code as u32 & ((1 << bits) - 1)) << (bit % 8);
        packed[bit / 8] |= code as u8;
        if bit % 8 + bits > 8 {
            packed[bit / 8 + 1] |= (code >> 8) as u8;
        }
    }
    packed
}

#[test]
fn integer_zero_points_dequantize_int4_and_int8_weights() -> Result<(), ane::Error> {
    let (rows, columns, block) = (8, 64, 32);
    for (data_type, bits, signed) in [
        (WeightDataType::Int8, 8, true),
        (WeightDataType::UInt8, 8, false),
        (WeightDataType::Int4, 4, true),
        (WeightDataType::UInt4, 4, false),
    ] {
        let low = if signed { -(1 << (bits - 1)) } else { 0 };
        let code = |i: usize| low + (i * 37 % (1 << bits)) as i32;
        let codes: Vec<i32> = (0..rows * columns).map(code).collect();
        let blocks = rows * columns / block;
        let zero_points: Vec<i32> = (0..blocks).map(|i| code(i * 11 + 5)).collect();
        let scales: Vec<f32> = (0..blocks)
            .map(|i| 1. / (1 << (i % 3 + 3)) as f32)
            .collect();
        let graph = Graph::new();
        let input = graph.placeholder([rows, columns], DataType::Float32)?;
        let weights = graph.blockwise_weights(
            &pack(&codes, bits),
            [rows, columns],
            &BlockwiseQuantization {
                zero_points: Some(&pack(&zero_points, bits)),
                ..BlockwiseQuantization::new(data_type, &scales, [rows, columns / block])
            },
        )?;
        let output = graph.multiplication(&weights, &input)?;
        let actual = evaluate(&graph, &[(input, &vec![1.; rows * columns])], output)?;
        let expected: Vec<f32> = (0..rows * columns)
            .map(|i| {
                let block = i / block;
                (codes[i] - zero_points[block]) as f32 * scales[block]
            })
            .collect();
        assert_eq!(actual, expected, "{data_type:?}");
    }
    Ok(())
}

#[test]
fn quantized_and_vector_palettes_decode_on_ane() -> Result<(), ane::Error> {
    let (rows, columns) = (4, 32);
    let ones = vec![1.; rows * columns];
    let indices: Vec<i32> = (0..rows * columns).map(|i| (i * 7 % 16) as i32).collect();

    let codes: Vec<i32> = (0..32).map(|i| i * 9 % 256 - 128).collect();
    let (scales, zero_points) = ([0.25f32, 0.125], [3, -5]);
    let graph = Graph::new();
    let input = graph.placeholder([rows, columns], DataType::Float32)?;
    let weights = graph.palettized_weights(
        &pack(&indices, 4),
        [rows, columns],
        &Palettization {
            group_shape: [2, 1],
            palette: ane::Palette::Quantized {
                data_type: WeightDataType::Int8,
                codes: &pack(&codes, 8),
                scales: &scales,
                zero_points: Some(&pack(&zero_points, 8)),
            },
            ..Palettization::new(4, &[])
        },
    )?;
    let output = graph.multiplication(&weights, &input)?;
    let actual = evaluate(&graph, &[(input, &ones)], output)?;
    let expected: Vec<f32> = (0..rows * columns)
        .map(|i| {
            let group = i / columns / 2;
            let code = codes[group * 16 + indices[i] as usize];
            (code - zero_points[group]) as f32 * scales[group]
        })
        .collect();
    assert_eq!(actual, expected, "int8 palette");

    let palette: Vec<f32> = (0..32).map(|i| i as f32 * 0.5 - 4.).collect();
    let vectors: Vec<i32> = indices[..rows * columns / 2].to_vec();
    let graph = Graph::new();
    let input = graph.placeholder([rows, columns], DataType::Float32)?;
    let weights = graph.palettized_weights(
        &pack(&vectors, 4),
        [rows, columns],
        &Palettization {
            vector_axis: Some(1),
            ..Palettization::new(4, &palette)
        },
    )?;
    let output = graph.multiplication(&weights, &input)?;
    let actual = evaluate(&graph, &[(input, &ones)], output)?;
    let expected: Vec<f32> = (0..rows * columns)
        .map(|i| {
            let (row, column) = (i / columns, i % columns);
            palette[vectors[row * columns / 2 + column / 2] as usize * 2 + column % 2]
        })
        .collect();
    assert_eq!(actual, expected, "vector palette");
    Ok(())
}

#[test]
fn int8_activations_move_through_layout_ops() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([16, 64], DataType::Float32)?;
    let codes = graph.quantize(&input, &[0.125], None, None, DataType::Int8)?;
    let transposed = graph.transpose(&codes, [1, 0])?;
    let reshaped = graph.reshape(&transposed, [32, 32])?;
    let sliced = graph.slice(&reshaped, [8, 0], [16, 32])?;
    let output = graph.dequantize(&sliced, &[0.125], None, None)?;
    let values: Vec<f32> = (0..16 * 64)
        .map(|i| (i % 255) as f32 * 0.125 - 15.875)
        .collect();
    let actual = evaluate(&graph, &[(input, &values)], output)?;
    let expected: Vec<f32> = (8 * 32..24 * 32)
        .map(|i| {
            let (row, column) = (i / 16, i % 16);
            values[column * 64 + row]
        })
        .collect();
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn purged_models_keep_running_and_recompile() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([64, 64], DataType::Float32)?;
    let offset = graph.constant(&[1.25], [1])?;
    let output = graph.addition(&input, &offset)?;
    let executable = graph.compile(&[output], &[], None)?;
    let data = executable.input(input)?.allocate()?;
    data.copy_from_f32(&[2.; 4096])?;
    executable.purge_compiled_model()?;
    let result = executable.run(&[&data], None, None)?;
    assert!(result[0].read_f32()?.iter().all(|&v| v == 3.25));
    let recompiled = graph.compile(&[output], &[], None)?;
    let result = recompiled.run(&[&data], None, None)?;
    assert!(result[0].read_f32()?.iter().all(|&v| v == 3.25));
    Ok(())
}

#[test]
fn compile_errors_carry_the_compiler_diagnostic() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let input = graph.placeholder([4, 8], DataType::Float16)?;
    let indices = graph.placeholder([3], DataType::UInt16)?;
    let rows = graph.gather(&input, &indices, 0)?;
    let message = graph.compile(&[rows], &[], None).err().unwrap().to_string();
    let detail = message
        .split_once("ANECCompile() FAILED: ")
        .map(|(_, detail)| detail);
    assert!(
        detail.is_some_and(|detail| detail.contains("Layer")),
        "{message}"
    );
    Ok(())
}
