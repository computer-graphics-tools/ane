use super::Executable;
use crate::{Error, request::Request};
use objc2_io_surface::IOSurface;

pub struct PreparedRequest<'model> {
    model: &'model Executable,
    request: Request,
}

impl<'model> PreparedRequest<'model> {
    pub fn new(
        model: &'model Executable,
        inputs: &[&IOSurface],
        outputs: &[&IOSurface],
    ) -> Result<Self, Error> {
        let request = model.make_request(inputs, outputs, &[], &[])?;
        model.inner.map(request.inner())?;
        Ok(Self { model, request })
    }

    pub fn run(&mut self) -> Result<(), Error> {
        self.model
            .inner
            .evaluate(self.model.qos, self.request.inner())?;
        Ok(())
    }
}

impl Drop for PreparedRequest<'_> {
    fn drop(&mut self) {
        self.model.inner.unmap(self.request.inner());
    }
}
