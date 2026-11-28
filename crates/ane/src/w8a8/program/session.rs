use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use super::super::{FFN_SHAPE_ALIGNMENT, INPUT_CHANNELS};
use super::{Host, Layer, Program, Surfaces, fits_ffn, fits_linear, valid_weight_layer};
use crate::{
    AneError, Error, SharedEventError,
    w8a8::{OutputBasis, WeightLayer},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Shape {
    Ffn {
        hidden: usize,
        intermediate: usize,
        up_output_basis: OutputBasis,
    },
    Linear {
        output: usize,
    },
}

#[derive(Default)]
struct State {
    host: Option<Host>,
    /// None remembers a failed shape without blocking other shapes.
    programs: Vec<(Shape, Option<Arc<Program>>)>,
    linear: Option<[Arc<Surfaces>; 2]>,
}

/// Lazily built programs; an unavailable ANE retires the session permanently.
#[derive(Default)]
pub struct Session {
    retired: Arc<AtomicBool>,
    state: Mutex<State>,
}

impl Session {
    /// Binds an FFN layer when its shape is supported.
    ///
    /// # Safety
    /// Compact weight regions and rotations must be correctly sized, immutable, resident on the
    /// host device, and owned by `owner`. The layer retains that owner through every call.
    pub unsafe fn bind_ffn(
        &self,
        up: &WeightLayer<'_>,
        down: &WeightLayer<'_>,
        owner: Arc<dyn Send + Sync>,
        host_factory: impl FnOnce() -> Result<Host, Error>,
    ) -> Result<Option<Layer>, Error> {
        let hidden = up.columns as usize;
        let intermediate = up.rows as usize / 2;
        if up.rows as usize != intermediate.saturating_mul(2)
            || down.rows as usize != hidden
            || down.columns as usize != intermediate
            || !fits_ffn(hidden, intermediate)
            || !valid_weight_layer(up)
            || !valid_weight_layer(down)
        {
            return Ok(None);
        }

        let up_output_basis = output_basis(up);
        let layer_output_basis = output_basis(down);

        let shape = Shape::Ffn {
            hidden,
            intermediate,
            up_output_basis,
        };
        let up_target_block = if hidden == INPUT_CHANNELS {
            INPUT_CHANNELS as u32
        } else {
            FFN_SHAPE_ALIGNMENT as u32
        };
        let Some(program) = self.program(shape, host_factory, |host| {
            host.kernels
                .prepare_weight_stage(
                    up.source,
                    up.columns,
                    up.input_rotation.block_size,
                    up_target_block,
                    false,
                )
                .map_err(Error::Metal)?;
            host.kernels
                .prepare_weight_stage(
                    down.source,
                    down.columns,
                    down.input_rotation.block_size,
                    FFN_SHAPE_ALIGNMENT as u32,
                    true,
                )
                .map_err(Error::Metal)
        })?
        else {
            return Ok(None);
        };
        program
            .bind_layer(up, Some(down), layer_output_basis, owner)
            .map(Some)
    }

    /// Binds a linear layer when its shape is supported.
    ///
    /// # Safety
    /// Compact weight regions and rotations must be correctly sized, immutable, resident on the
    /// host device, and owned by `owner`. The layer retains that owner through every call.
    pub unsafe fn bind_linear(
        &self,
        weights: &WeightLayer<'_>,
        owner: Arc<dyn Send + Sync>,
        host_factory: impl FnOnce() -> Result<Host, Error>,
    ) -> Result<Option<Layer>, Error> {
        let (input, output) = (weights.columns as usize, weights.rows as usize);
        if !fits_linear(input, output) || !valid_weight_layer(weights) {
            return Ok(None);
        }

        let layer_output_basis = output_basis(weights);
        let Some(program) = self.program(Shape::Linear { output }, host_factory, |host| {
            host.kernels
                .prepare_weight_stage(
                    weights.source,
                    weights.columns,
                    weights.input_rotation.block_size,
                    INPUT_CHANNELS as u32,
                    false,
                )
                .map_err(Error::Metal)
        })?
        else {
            return Ok(None);
        };
        program
            .bind_layer(weights, None, layer_output_basis, owner)
            .map(Some)
    }

    /// Permanently disables further ANE program lookups for this session.
    pub fn retire(&self) {
        self.retired.store(true, Ordering::Release);
    }

    /// Whether this session has been disabled or the ANE became unavailable.
    pub fn is_retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }

    /// Whether a usable FFN program has been built and the session remains healthy.
    pub fn has_ffn_program(&self) -> bool {
        if self.is_retired() {
            return false;
        }
        let state = lock(&self.state);
        !self.is_retired()
            && state
                .programs
                .iter()
                .any(|(shape, program)| matches!(shape, Shape::Ffn { .. }) && program.is_some())
    }

    fn program(
        &self,
        shape: Shape,
        host_factory: impl FnOnce() -> Result<Host, Error>,
        prepare: impl FnOnce(&Host) -> Result<(), Error>,
    ) -> Result<Option<Arc<Program>>, Error> {
        if self.is_retired() {
            return Ok(None);
        }

        let mut state = lock(&self.state);
        if self.is_retired() {
            return Ok(None);
        }
        if let Some((_, program)) = state.programs.iter().find(|(known, _)| *known == shape) {
            let Some(program) = program.clone() else {
                return Ok(None);
            };
            let host = state
                .host
                .as_ref()
                .expect("a cached program keeps its host in the session")
                .clone();
            prepare(&host)?;
            return Ok(Some(program));
        }

        if state.host.is_none() {
            match host_factory() {
                Ok(host) => state.host = Some(host),
                Err(error) => {
                    state.programs.push((shape, None));
                    if is_unavailable(&error) {
                        self.retire();
                    }
                    return Err(error);
                }
            }
        }
        let host = state
            .host
            .as_ref()
            .expect("the host was initialized above")
            .clone();
        prepare(&host)?;
        match self.build(shape, &host, &mut state) {
            Ok(program) => {
                let program = Arc::new(program);
                state.programs.push((shape, Some(program.clone())));
                Ok(Some(program))
            }
            Err(error) => {
                state.programs.push((shape, None));
                if is_unavailable(&error) {
                    self.retire();
                }
                Err(error)
            }
        }
    }

    fn build(&self, shape: Shape, host: &Host, state: &mut State) -> Result<Program, Error> {
        match shape {
            Shape::Ffn {
                hidden,
                intermediate,
                up_output_basis,
            } => Program::ffn(
                hidden,
                intermediate,
                up_output_basis,
                host,
                self.retired.clone(),
            ),
            Shape::Linear { output } => {
                Program::linear(output, &mut state.linear, host, self.retired.clone())
            }
        }
    }
}

fn output_basis(layer: &WeightLayer<'_>) -> OutputBasis {
    match layer.output_rotation {
        Some(rotation) if rotation.block_size == 32 => OutputBasis::Hadamard32,
        _ => OutputBasis::Identity,
    }
}

fn is_unavailable(error: &Error) -> bool {
    matches!(
        error,
        Error::Ane(AneError::FrameworkNotFound | AneError::Unavailable(_))
            | Error::Device(_)
            | Error::SharedEvent(SharedEventError::Unsupported(_))
    )
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}
