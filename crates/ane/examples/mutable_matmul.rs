use ane::{DataType, Error, Graph, IOSurface, IOSurfaceExt, TensorData};
use half::f16;
use std::io::IsTerminal;

fn draw_matrices(input: &[f32], weights: &[f16], output: &[f32], stage: &str, runs: usize) {
    let weights: Vec<_> = weights.iter().map(|value| value.to_f32()).collect();
    let matrices = [input, weights.as_slice(), output];
    let styles = [
        "38;2;125;211;252;48;2;20;41;54",
        "38;2;216;180;254;48;2;41;30;57",
        "38;2;134;239;172;48;2;20;46;36",
    ];
    let border = "─".repeat(24);
    println!("\x1b[H\x1b[J\x1b[1;38;2;198;214;170mANE  /  LIVE MATRIX MULTIPLICATION\x1b[0m");
    println!("\x1b[38;2;166;173;200mChange the weights. Keep the compiled model.\x1b[0m\n");
    println!("\x1b[1m{stage}\x1b[0m\n");
    println!(
        "{:^26}     {:^26}     {:^26}",
        "INPUT X", "WEIGHTS W", "RESULT Y"
    );
    println!(
        "\x1b[2m{:^26}     {:^26}     {:^26}\x1b[0m",
        "FP32", "FP16", "FP32"
    );
    println!("\x1b[2m╭{border}╮     ╭{border}╮     ╭{border}╮\x1b[0m");
    for row in 0..4 {
        for (matrix_index, matrix) in matrices.iter().enumerate() {
            print!("\x1b[2m│\x1b[0m");
            for column in 0..4 {
                let value = matrix[row * 64 + column];
                let style = if runs == 2 && matrix_index > 0 && row == 1 && column == 1 {
                    "38;2;253;230;138;48;2;76;48;20"
                } else if value == 0.0 {
                    "38;2;77;86;106"
                } else {
                    styles[matrix_index]
                };
                print!("\x1b[{style}m{value:>6.0}\x1b[0m");
            }
            print!("\x1b[2m│\x1b[0m");
            match (matrix_index, row) {
                (0, 1) => print!("  ×  "),
                (1, 1) => print!("  =  "),
                (0 | 1, _) => print!("     "),
                _ => {}
            }
        }
        println!();
    }
    println!("\x1b[2m╰{border}╯     ╰{border}╯     ╰{border}╯\x1b[0m");
    println!("\x1b[2mTop-left 4×4 shown. Full 64×64 result checked.\x1b[0m\n");
    println!("\x1b[1;38;2;134;239;172m✓ All 4096 outputs match\x1b[0m");
    println!(
        "1 compile  ·  {runs} executions  ·  {} exact checks",
        runs * 4096
    );
}

fn main() -> Result<(), Error> {
    let visual = std::io::stdout().is_terminal();
    let activation_surface = IOSurface::with_byte_count(64 * 64 * 4)?;
    let weight_surface = IOSurface::with_byte_count(64 * 64 * 2)?;
    let decoy_surface = IOSurface::with_byte_count(64 * 64 * 2)?;
    let activations = unsafe {
        TensorData::from_surface(activation_surface.clone(), [64, 64], DataType::Float32)?
    };
    let weights =
        unsafe { TensorData::from_surface(weight_surface.clone(), [64, 64], DataType::Float16)? };

    let graph = Graph::new();
    let x = graph.placeholder([64, 64], DataType::Float32)?;
    let w = graph.variable_with_tensor_data(&weights)?;
    let weight_value = graph.read_variable(&w)?;
    let y = graph.matrix_multiplication(&x, &weight_value, false, false)?;
    let executable = graph.compile(&[y], &[], None)?;
    assert_eq!(executable.report().constant_bytes, 0);
    drop(graph);
    let output = executable.output(y)?.allocate()?;

    assert_eq!(
        executable.variable_data(&w)?.surface().surfaceID(),
        weight_surface.surfaceID(),
    );
    drop(weights);
    if !visual {
        println!(
            "activation IOSurface {}, weight IOSurface {}, decoy IOSurface {}",
            activation_surface.surfaceID(),
            weight_surface.surfaceID(),
            decoy_surface.surfaceID(),
        );
        println!("compiled once with zero-filled surfaces and 0 constant weight bytes");
    }

    let set_activation = |shift: usize| -> Result<(), Error> {
        let mut values = activations.as_f32_slice_mut()?;
        values.fill(0.0);
        for row in 0..64 {
            values[row * 64 + (row + shift) % 64] = 1.0;
        }
        Ok(())
    };
    let mut observed = vec![0u8; 64 * 64 * 2];
    let mut runs = 0;
    let mut check = |label: &str, shift: usize, expected: &[f16]| -> Result<(), Error> {
        output.as_f32_slice_mut()?.fill(f32::NAN);
        executable.run(&[&activations], Some(&[&output]), None)?;
        let result = output.read_f32()?;
        assert_eq!(result.len(), 64 * 64);
        for (index, actual) in result.iter().enumerate() {
            let source = ((index / 64 + shift) % 64) * 64 + index % 64;
            assert_eq!(
                actual.to_bits(),
                expected[source].to_f32().to_bits(),
                "{label}: output {index}, weight {source}",
            );
        }
        unsafe { weight_surface.read_bytes(&mut observed)? };
        assert_eq!(
            observed,
            bytemuck::cast_slice::<_, u8>(expected),
            "weights changed",
        );
        runs += 1;
        if visual && runs <= 4 {
            draw_matrices(&activations.read_f32()?, expected, &result, label, runs);
            std::thread::sleep(std::time::Duration::from_millis(1400));
        }
        Ok(())
    };

    if visual {
        let mut expected: Vec<_> = (0..4096)
            .map(|index| f16::from_f32(((index * 7 + index / 64 * 3) % 19) as f32 - 9.0))
            .collect();
        set_activation(0)?;
        unsafe { weight_surface.write_bytes(bytemuck::cast_slice(&expected))? };
        check("01 / 04  Load weights after compilation", 0, &expected)?;

        expected[65] = -expected[65];
        unsafe { weight_surface.write_bytes(bytemuck::cast_slice(&expected))? };
        check(
            "02 / 04  Flip one weight → the result changes",
            0,
            &expected,
        )?;

        unsafe { decoy_surface.write_bytes(&[0xff; 64 * 64 * 2])? };
        check(
            "03 / 04  Change an unbound buffer → the result stays",
            0,
            &expected,
        )?;

        set_activation(1)?;
        check(
            "04 / 04  Shift the input → the result rows move",
            1,
            &expected,
        )?;
    }

    for round in 0..16 {
        let salt = rand::random::<u16>() & 0x0fff;
        let mut expected: Vec<_> = (0..4096)
            .map(|i| f16::from_bits(0x3000 | (i ^ salt) | ((round % 2) << 15)))
            .collect();
        set_activation(0)?;
        unsafe { weight_surface.write_bytes(bytemuck::cast_slice(&expected))? };
        check("identity", 0, &expected)?;

        let index = usize::from(round) * 257;
        expected[index] = -expected[index];
        unsafe { weight_surface.write_bytes(bytemuck::cast_slice(&expected))? };
        check("single weight changed", 0, &expected)?;

        unsafe { decoy_surface.write_bytes(&[0xff; 64 * 64 * 2])? };
        check("unbound surface changed", 0, &expected)?;

        let shift = usize::from(round) + 1;
        set_activation(shift)?;
        check("activation permuted", shift, &expected)?;
    }
    if visual {
        println!(
            "\n\x1b[1;38;2;134;239;172m✓ {runs} executions · {} exact checks passed\x1b[0m",
            runs * 4096
        );
        println!("\x1b[2mMutable weights · edit isolation · input permutations verified\x1b[0m");
    } else {
        println!(
            "{runs} executions, {} exact output checks passed",
            runs * 4096
        );
        println!(
            "post-compile weights, single-weight edits, decoy isolation and input permutations passed"
        );
    }
    Ok(())
}
