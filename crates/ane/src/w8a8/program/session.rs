use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use super::{Host, Layer, Program, Surfaces, fits_ffn, fits_linear, valid_w4};
use crate::w8a8::RhtW4;
use crate::{AneError, Error, SharedEventError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Shape {
    Ffn { hidden: usize, intermediate: usize },
    Linear { output: usize },
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
    /// W4 regions must be correctly sized, immutable, resident on the host device, and owned
    /// by `owner`. The layer retains that owner through every call.
    pub unsafe fn bind_ffn(
        &self,
        up: &RhtW4,
        down: &RhtW4,
        owner: Arc<dyn Send + Sync>,
        host_factory: impl FnOnce() -> Result<Host, Error>,
    ) -> Result<Option<Layer>, Error> {
        let hidden = up.input_signs.len();
        let intermediate = up.output_signs.len() / 2;
        if up.output_signs.len() != intermediate * 2
            || down.output_signs.len() != hidden
            || down.input_signs.len() != intermediate
            || !fits_ffn(hidden, intermediate)
            || !valid_w4(up.w4, up.output_signs.len(), hidden)
            || !valid_w4(down.w4, hidden, intermediate)
        {
            return Ok(None);
        }

        let shape = Shape::Ffn {
            hidden,
            intermediate,
        };
        let Some(program) = self.program(shape, host_factory)? else {
            return Ok(None);
        };
        program.bind_layer(up, Some(down), owner).map(Some)
    }

    /// Binds a linear layer when its shape is supported.
    ///
    /// # Safety
    /// W4 regions must be correctly sized, immutable, resident on the host device, and owned
    /// by `owner`. The layer retains that owner through every call.
    pub unsafe fn bind_linear(
        &self,
        weights: &RhtW4,
        owner: Arc<dyn Send + Sync>,
        host_factory: impl FnOnce() -> Result<Host, Error>,
    ) -> Result<Option<Layer>, Error> {
        let (input, output) = (weights.input_signs.len(), weights.output_signs.len());
        if !fits_linear(input, output) || !valid_w4(weights.w4, output, input) {
            return Ok(None);
        }

        let Some(program) = self.program(Shape::Linear { output }, host_factory)? else {
            return Ok(None);
        };
        program.bind_layer(weights, None, owner).map(Some)
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
    ) -> Result<Option<Arc<Program>>, Error> {
        if self.is_retired() {
            return Ok(None);
        }

        let mut state = lock(&self.state);
        if self.is_retired() {
            return Ok(None);
        }
        if let Some((_, program)) = state.programs.iter().find(|(known, _)| *known == shape) {
            return Ok(program.clone());
        }

        match self.build(shape, host_factory, &mut state) {
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

    fn build(
        &self,
        shape: Shape,
        host_factory: impl FnOnce() -> Result<Host, Error>,
        state: &mut State,
    ) -> Result<Program, Error> {
        if state.host.is_none() {
            state.host = Some(host_factory()?);
        }
        let host = state.host.as_ref().expect("the host was created above");
        match shape {
            Shape::Ffn {
                hidden,
                intermediate,
            } => Program::ffn(hidden, intermediate, host, self.retired.clone()),
            Shape::Linear { output } => {
                Program::linear(output, &mut state.linear, host, self.retired.clone())
            }
        }
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
