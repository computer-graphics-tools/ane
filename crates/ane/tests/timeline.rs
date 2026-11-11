mod common;

use std::time::Duration;

use ane::{DataType, Graph, SharedEvent, Timeline};

const TIMEOUT: Duration = Duration::from_secs(10);

#[test]
fn trips_take_consecutive_values() -> Result<(), ane::Error> {
    let timeline = Timeline::new(SharedEvent::new()?);
    let (first, second) = (timeline.reserve(), timeline.reserve());
    assert_eq!(
        [first.ready, first.done, second.ready, second.done],
        [1, 2, 3, 4]
    );
    Ok(())
}

#[test]
fn the_ane_runs_a_trip_between_ready_and_done() -> Result<(), ane::Error> {
    require_ane!();
    let graph = Graph::new();
    let x = graph.placeholder([64, 64], DataType::Float32)?;
    let y = graph.addition(&x, &graph.constant(&[1.], [1])?)?;
    let executable = graph.compile(&[y], &[], None)?;
    let (input, output) = (
        executable.input(x)?.allocate()?,
        executable.output(y)?.allocate()?,
    );
    let bound = executable.bind(&[&input], &[&output])?;

    let event = SharedEvent::new()?;
    let timeline = Timeline::new(event.clone());
    for step in 0..3 {
        let trip = timeline.reserve();
        input.copy_from_f32(&[step as f32; 4096])?;
        // SAFETY: the CPU writes the input before `ready` and reads the output after `done`.
        unsafe { timeline.submit(trip, &bound, |error| panic!("{error}"))? };
        event.set_signaled_value(trip.ready);
        assert!(
            event.wait_until_signaled_value(trip.done, TIMEOUT),
            "step {step}"
        );
        assert!(
            output
                .read_f32()?
                .iter()
                .all(|&value| value == step as f32 + 1.)
        );
    }
    Ok(())
}
