use std::path::Path;
use std::sync::{Mutex, PoisonError};

use objc2::rc::Retained;
use objc2_foundation::{NSData, NSQualityOfService};

use crate::Error;
use crate::apple_neural_engine::{
    ANEClient, ANEInMemoryModel, ANEInMemoryModelDescriptor, ANEModel, ANERequest, ModelAttributes,
};
use crate::ir::{MilProgram, catch_native};

static MODEL_LIFECYCLE: Mutex<()> = Mutex::new(());

pub struct LoadedProgram {
    program: Retained<ANEInMemoryModel>,
    model: Retained<ANEModel>,
    client: Retained<ANEClient>,
    qos: NSQualityOfService,
    dispatch: Mutex<()>,
}
// SAFETY: model calls are serialized by `dispatch` and `MODEL_LIFECYCLE`; the Objective-C objects
// are never mutated outside those locks.
unsafe impl Send for LoadedProgram {}
unsafe impl Sync for LoadedProgram {}

impl LoadedProgram {
    pub fn new(mil: MilProgram, qos: NSQualityOfService) -> Result<Self, Error> {
        let _lifecycle = MODEL_LIFECYCLE.lock().map_err(|_| Error::Synchronization)?;
        let (program, model) =
            objc2::rc::autoreleasepool(|_| catch_native(|| Self::load(mil, qos)))??;
        Ok(Self {
            program,
            model,
            client: ANEClient::shared()?,
            qos,
            dispatch: Mutex::new(()),
        })
    }
    fn load(
        mil: MilProgram,
        qos: NSQualityOfService,
    ) -> Result<(Retained<ANEInMemoryModel>, Retained<ANEModel>), Error> {
        let text = NSData::from_vec(mil.text.into_bytes());
        let weights = NSData::from_vec(mil.weights);
        let program = ANEInMemoryModel::new(&*ANEInMemoryModelDescriptor::new(&text, &weights)?)?;
        let directory = std::env::temp_dir().join(program.identifier()?);
        let loaded = stage(
            &directory,
            &[("model.mil", &text), ("weights/weight.bin", &weights)],
        )
        .and_then(|()| {
            if !program.is_compiled() {
                program.compile(qos)?;
            }
            Ok(program.load(qos)?)
        });
        let _ = std::fs::remove_dir_all(&directory);
        loaded?;
        let model = program.model()?;
        Ok((program, model))
    }
    pub fn purge(&self) -> Result<(), Error> {
        let _lifecycle = MODEL_LIFECYCLE.lock().map_err(|_| Error::Synchronization)?;
        self.program.purge_compiled_model();
        Ok(())
    }
    pub fn model_attributes(&self) -> Result<ModelAttributes, Error> {
        Ok(self.model.model_attributes()?)
    }
    pub fn evaluate(&self, request: &ANERequest) -> Result<(), Error> {
        let _dispatch = self.dispatch.lock().map_err(|_| Error::Synchronization)?;
        Ok(self.client.evaluate(&self.model, self.qos, request)?)
    }
    pub fn map(&self, request: &ANERequest) -> Result<(), Error> {
        let _dispatch = self.dispatch.lock().map_err(|_| Error::Synchronization)?;
        Ok(self.client.map(&self.model, request)?)
    }
    pub fn unmap(&self, request: &ANERequest) {
        let _dispatch = self.dispatch.lock().unwrap_or_else(PoisonError::into_inner);
        self.client.unmap(&self.model, request);
    }
}

fn stage(directory: &Path, files: &[(&str, &NSData)]) -> Result<(), Error> {
    std::fs::create_dir_all(directory.join("weights"))?;
    for (name, data) in files.iter().filter(|(_, data)| !data.is_empty()) {
        std::fs::write(directory.join(name), unsafe { data.as_bytes_unchecked() })?;
    }
    Ok(())
}

impl Drop for LoadedProgram {
    fn drop(&mut self) {
        let _lifecycle = MODEL_LIFECYCLE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let _ = self.program.unload(self.qos);
    }
}
