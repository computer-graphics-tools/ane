use ane::{DataType, Error, Graph, IOSurface, IOSurfaceExt, TensorData};
use half::f16;

fn main() -> Result<(), Error> {
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
    println!(
        "activation IOSurface {}, weight IOSurface {}, decoy IOSurface {}",
        activation_surface.surfaceID(),
        weight_surface.surfaceID(),
        decoy_surface.surfaceID(),
    );
    println!("compiled once with zero-filled surfaces and 0 constant weight bytes");

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
        Ok(())
    };

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
    println!(
        "{runs} executions, {} exact output checks passed",
        runs * 4096
    );
    println!(
        "post-compile weights, single-weight edits, decoy isolation and input permutations passed"
    );
    Ok(())
}
