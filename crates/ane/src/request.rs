use objc2::{Message, rc::Retained};
use objc2_io_surface::IOSurface;

use crate::apple_neural_engine::ANERequest;
use crate::io_surface::SurfaceLease;
use crate::shared_event::io_surface_shared_event;
use crate::{Completion, Error, ExecutionDescriptor, IOSurfaceError, LoadedProgram, Outcome};
use std::sync::{Arc, Mutex, OnceLock};

pub struct Request {
    inner: Retained<ANERequest>,
    program: Arc<LoadedProgram>,
    completion: OnceLock<Arc<Completion>>,
    access: Box<[(Retained<IOSurface>, bool)]>,
    bindings: Box<[u32]>,
}

// SAFETY: the Objective-C request is only evaluated or mapped through `LoadedProgram`, which
// serializes those calls, and its surfaces are guarded by surface leases.
unsafe impl Send for Request {}
unsafe impl Sync for Request {}

impl Request {
    pub fn new(
        program: Arc<LoadedProgram>,
        procedure: u32,
        inputs: &[(&IOSurface, u32, bool)],
        outputs: &[(&IOSurface, u32)],
        bindings: Box<[u32]>,
    ) -> Result<Arc<Self>, Error> {
        let mut access: Vec<_> = inputs
            .iter()
            .map(|&(surface, _, write)| (surface.retain(), write))
            .collect();
        for (index, (surface, write)) in access.iter().enumerate() {
            if access[..index].iter().any(|(other, other_write)| {
                surface.surfaceID() == other.surfaceID() && (*write || *other_write)
            }) {
                return Err(Error::Alias);
            }
        }
        for (surface, _) in outputs {
            if access
                .iter()
                .any(|(s, _)| s.surfaceID() == surface.surfaceID())
            {
                return Err(Error::Alias);
            }
            access.push((surface.retain(), true));
        }
        access.sort_by_key(|(s, _)| s.surfaceID());
        access.dedup_by(|a, b| {
            if a.0.surfaceID() != b.0.surfaceID() {
                return false;
            }
            b.1 |= a.1;
            true
        });
        let (input_surfaces, input_indices): (Vec<_>, Vec<_>) = inputs
            .iter()
            .map(|&(surface, index, _)| (surface, index))
            .unzip();
        let (output_surfaces, output_indices): (Vec<_>, Vec<_>) = outputs.iter().copied().unzip();
        let inner = ANERequest::new(
            &input_surfaces,
            &input_indices,
            &output_surfaces,
            &output_indices,
            procedure,
        )?;
        program.map(&inner)?;
        Ok(Arc::new(Self {
            inner,
            program,
            completion: OnceLock::new(),
            access: access.into(),
            bindings,
        }))
    }

    /// IOSurface IDs of the caller-provided inputs and outputs.
    pub fn bindings(&self) -> &[u32] {
        &self.bindings
    }

    pub fn run(self: &Arc<Self>) -> Result<(), Error> {
        if self.completion.get().is_some() {
            return self.submit(None)?.wait();
        }
        let _leases = self.acquire()?;
        self.program.evaluate(&self.inner)
    }

    pub fn submit(
        self: &Arc<Self>,
        descriptor: Option<&ExecutionDescriptor<'_>>,
    ) -> Result<Arc<Outcome>, Error> {
        let descriptor = descriptor.copied().unwrap_or_default();
        let completion = self.completion()?;
        let (value, outcome) = completion.begin(self.clone(), self.acquire()?)?;
        let wait: Vec<_> = descriptor
            .wait_events
            .iter()
            .map(|(event, value)| (io_surface_shared_event(event), *value))
            .collect();
        let signal: Vec<_> = descriptor
            .signal_events
            .iter()
            .map(|(event, value)| (io_surface_shared_event(event), *value))
            .chain([(io_surface_shared_event(&completion.event), value)])
            .collect();
        let configured = self
            .inner
            .set_shared_events(&wait, &signal)
            .map_err(Error::from)
            .and_then(|()| self.program.evaluate(&self.inner));
        if let Err(error) = configured {
            completion.abort();
            return Err(error);
        }
        Ok(outcome)
    }

    fn acquire(&self) -> Result<Vec<SurfaceLease>, Error> {
        self.access
            .iter()
            .map(|(surface, write)| {
                SurfaceLease::acquire(surface, *write).map_err(|error| match error {
                    IOSurfaceError::Busy => Error::BufferBusy,
                    error => error.into(),
                })
            })
            .collect()
    }

    fn completion(&self) -> Result<Arc<Completion>, Error> {
        if let Some(completion) = self.completion.get() {
            return Ok(completion.clone());
        }
        static INITIALIZATION: Mutex<()> = Mutex::new(());
        let _init = INITIALIZATION.lock().map_err(|_| Error::Synchronization)?;
        if let Some(completion) = self.completion.get() {
            return Ok(completion.clone());
        }
        let completion = Arc::new(Completion::new()?);
        let weak = Arc::downgrade(&completion);
        self.inner.set_completion_handler(move |result| {
            if let Some(completion) = weak.upgrade() {
                completion.complete(result.map_err(Error::from));
            }
        })?;
        self.completion
            .set(completion.clone())
            .map_err(|_| Error::Synchronization)?;
        Ok(completion)
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        self.program.unmap(&self.inner);
    }
}
