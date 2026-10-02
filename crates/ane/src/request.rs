use objc2::{Message, rc::Retained};
use objc2_io_surface::IOSurface;

use crate::apple_neural_engine::ANERequest;
use crate::io_surface::SurfaceLease;
use crate::shared_event::io_surface_shared_event;
use crate::{Error, LoadedProgram, NativeOutputs, SharedEvent, completion::Completion};
use std::sync::{Arc, Mutex, OnceLock};

pub struct Request {
    inner: Retained<ANERequest>,
    program: Arc<LoadedProgram>,
    mapped: bool,
    completion: OnceLock<Arc<Completion>>,
    access: Box<[(Retained<IOSurface>, bool)]>,
    bindings: Box<[u32]>,
}

unsafe impl Send for Request {}
unsafe impl Sync for Request {}

impl Request {
    pub fn bindings(&self) -> &[u32] {
        &self.bindings
    }

    pub fn acquire_into(&self, leases: &mut Vec<SurfaceLease>) -> Result<(), Error> {
        for (surface, write) in &self.access {
            match SurfaceLease::acquire(surface, *write) {
                Ok(lease) => leases.push(lease),
                Err(error) => {
                    leases.clear();
                    return Err(error.into());
                }
            }
        }
        Ok(())
    }
    pub fn map(&mut self) -> Result<(), Error> {
        self.program.map(&self.inner)?;
        self.mapped = true;
        Ok(())
    }
    pub fn run(self: &Arc<Self>) -> Result<(), Error> {
        if self.completion.get().is_some() {
            return self.submit(&[], &[])?.wait();
        }
        let mut leases = Vec::with_capacity(self.access.len());
        self.acquire_into(&mut leases)?;
        self.program.evaluate(&self.inner)
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
    pub fn submit(
        self: &Arc<Self>,
        wait: &[(&SharedEvent, u64)],
        signal: &[(&SharedEvent, u64)],
    ) -> Result<Arc<Completion>, Error> {
        let completion = self.completion()?;
        let value = completion.begin(self.clone())?;
        let wait: Vec<_> = wait
            .iter()
            .map(|(event, value)| (io_surface_shared_event(event), *value))
            .collect();
        let signal: Vec<_> = signal
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
        Ok(completion)
    }

    pub fn new(
        program: Arc<LoadedProgram>,
        inputs: &[&IOSurface],
        outputs: &[&IOSurface],
        input_indices: &[u32],
        output_indices: &[u32],
        input_writes: &[bool],
        extra_outputs: &NativeOutputs,
    ) -> Result<Self, Error> {
        let bindings = inputs
            .iter()
            .chain(outputs)
            .map(|surface| surface.surfaceID())
            .collect();
        let state_outputs = &extra_outputs.states;
        let discarded_outputs = &extra_outputs.discarded;
        let mut access: Vec<_> = inputs
            .iter()
            .zip(input_writes)
            .map(|(s, w)| (s.retain(), *w))
            .collect();
        for (index, (surface, write)) in access.iter().enumerate() {
            if access[..index].iter().any(|(other, other_write)| {
                surface.surfaceID() == other.surfaceID() && (*write || *other_write)
            }) {
                return Err(Error::Alias);
            }
        }
        for surface in outputs {
            if access
                .iter()
                .any(|(s, _)| s.surfaceID() == surface.surfaceID())
            {
                return Err(Error::Alias);
            }
            access.push((surface.retain(), true));
        }
        let discarded = discarded_outputs
            .iter()
            .map(|(_, spec)| spec.allocate())
            .collect::<Result<Vec<_>, _>>()?;
        access.extend(discarded.iter().map(|data| (data.surface().retain(), true)));
        access.sort_by_key(|(s, _)| s.surfaceID());
        access.dedup_by(|a, b| {
            if a.0.surfaceID() != b.0.surfaceID() {
                return false;
            }
            b.1 |= a.1;
            true
        });
        let mut native_outputs = outputs.to_vec();
        let mut native_indices = output_indices.to_vec();
        for &(index, input) in state_outputs {
            if !input_writes.get(input).copied().unwrap_or(false) {
                return Err(Error::Metadata("state output without writable input"));
            }
            native_outputs.push(inputs[input]);
            native_indices.push(index);
        }
        native_outputs.extend(discarded.iter().map(|data| data.surface()));
        native_indices.extend(discarded_outputs.iter().map(|(index, _)| *index));
        let inner = ANERequest::new(inputs, input_indices, &native_outputs, &native_indices)?;
        Ok(Self {
            inner,
            program,
            mapped: false,
            completion: OnceLock::new(),
            access: access.into(),
            bindings,
        })
    }
}

impl Drop for Request {
    fn drop(&mut self) {
        if self.mapped {
            self.program.unmap(&self.inner);
        }
    }
}
