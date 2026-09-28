use std::sync::Arc;

use block2::RcBlock;
use objc2::{
    msg_send,
    rc::Retained,
    runtime::{AnyObject, Bool},
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSQualityOfService, NSString};
use objc2_io_surface::IOSurface;

use crate::ane_in_memory_model::ANEInMemoryModel;
use crate::completion::Completion;
use crate::request::Request;
use crate::{DataType, Error, MilProgram, Submission, TensorData};

#[path = "prepared_request.rs"]
mod prepared_request;
pub use prepared_request::PreparedRequest;

pub struct Executable {
    inner: Retained<ANEInMemoryModel>,
    qos: NSQualityOfService,
    inputs: Box<[(String, [usize; 4], DataType)]>,
    outputs: Box<[(String, [usize; 4], DataType)]>,
    input_indices: Vec<u32>,
    output_indices: Vec<u32>,
    input_sizes: Vec<usize>,
    output_sizes: Vec<usize>,
}

unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

fn dictionary(
    object: &AnyObject,
    key: &str,
) -> Result<Retained<NSDictionary<NSString, AnyObject>>, Error> {
    let value: Option<Retained<NSDictionary<NSString, AnyObject>>> =
        unsafe { msg_send![object, objectForKey: &*NSString::from_str(key)] };
    value.ok_or_else(|| Error::Binding(format!("missing metadata dictionary {key}")))
}
fn array(object: &AnyObject, key: &str) -> Result<Retained<NSArray<AnyObject>>, Error> {
    let value: Option<Retained<NSArray<AnyObject>>> =
        unsafe { msg_send![object, objectForKey: &*NSString::from_str(key)] };
    value.ok_or_else(|| Error::Binding(format!("missing metadata array {key}")))
}
fn string(object: &AnyObject, key: &str) -> Result<String, Error> {
    let value: Option<Retained<NSString>> =
        unsafe { msg_send![object, objectForKey: &*NSString::from_str(key)] };
    value
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Binding(format!("missing metadata string {key}")))
}
fn integer(object: &AnyObject, key: &str) -> Result<usize, Error> {
    let value: Option<Retained<NSNumber>> =
        unsafe { msg_send![object, objectForKey: &*NSString::from_str(key)] };
    let value = value.ok_or_else(|| Error::Binding(format!("missing metadata integer {key}")))?;
    Ok(unsafe { msg_send![&*value, unsignedIntegerValue] })
}

fn bindings(
    specs: &[(String, [usize; 4], DataType)],
    names: &NSArray<AnyObject>,
    layouts: &NSArray<AnyObject>,
    output: bool,
) -> Result<(Vec<u32>, Vec<usize>), Error> {
    if names.count() != specs.len() || layouts.count() != specs.len() {
        return Err(Error::Binding(
            "compiled tensor count differs from MIL".into(),
        ));
    }
    let names: Vec<String> = names
        .iter()
        .map(|s| {
            let name: Retained<NSString> = unsafe { msg_send![&*s, description] };
            name.to_string()
        })
        .collect();
    let mut indices = Vec::new();
    let mut sizes = Vec::new();
    for (name, shape, dtype) in specs {
        let symbol = if output {
            format!("{name}@output")
        } else {
            name.clone()
        };
        let index = names
            .iter()
            .position(|n| n == &symbol)
            .ok_or_else(|| Error::Binding(format!("missing symbol {symbol}")))?;
        let layout = layouts
            .iter()
            .find(|l| string(l, "Name").ok().as_deref() == Some(symbol.as_str()))
            .ok_or_else(|| Error::Binding(format!("missing layout {symbol}")))?;
        let expected_type = match dtype {
            DataType::Float32 => "Float32",
            DataType::Float16 => "Float16",
            DataType::Int8 => "Int8",
            DataType::UInt8 => "UInt8",
            DataType::Int32 => "Int32",
        };
        if *dtype == DataType::Int32 {
            if shape.iter().product::<usize>() != 1 || string(&layout, "Type")? != expected_type {
                return Err(Error::Binding("invalid scalar parameter layout".into()));
            }
            indices.push(index as u32);
            sizes.push(4);
            continue;
        }
        let row = shape[3] * dtype.byte_width();
        if integer(&layout, "Width")? != shape[3]
            || integer(&layout, "Height")? != shape[2]
            || integer(&layout, "Channels")? != shape[1]
            || integer(&layout, "Batches")? != shape[0]
            || integer(&layout, "Depth")? != 1
            || string(&layout, "Type")? != expected_type
            || (shape[2] > 1 && integer(&layout, "RowStride")? != row)
            || (shape[1] > 1 && integer(&layout, "PlaneStride")? != row * shape[2])
            || (shape[0] > 1 && integer(&layout, "BatchStride")? != row * shape[2] * shape[1])
        {
            return Err(Error::Binding(format!(
                "{symbol} requires a padded or different layout; dense zero-copy binding is unavailable"
            )));
        }
        indices.push(index as u32);
        sizes.push(
            integer(&layout, "BatchStride")?
                .checked_mul(integer(&layout, "Batches")?)
                .ok_or_else(|| Error::Binding("compiled allocation size overflow".into()))?,
        );
    }
    Ok((indices, sizes))
}

impl Executable {
    pub fn compile(program: MilProgram, qos: NSQualityOfService) -> Result<Self, Error> {
        let inner = crate::compile_model(&program, qos)?;
        let mut executable = Self {
            inner,
            qos,
            inputs: program.inputs,
            outputs: program.outputs,
            input_indices: Vec::new(),
            output_indices: Vec::new(),
            input_sizes: Vec::new(),
            output_sizes: Vec::new(),
        };
        let attributes = executable.inner.model_attributes();
        let description = dictionary(&attributes, "ANEFModelDescription")?;
        let networks = array(&attributes, "NetworkStatusList")?;
        if networks.count() == 0 {
            return Err(Error::Binding("no compiled network metadata".into()));
        }
        let network = networks.objectAtIndex(0);
        let inputs = array(&description, "kANEFModelInputSymbolsArrayKey")?;
        let outputs = array(&description, "kANEFModelOutputSymbolsArrayKey")?;
        let mut input_layouts = Vec::new();
        for key in ["LiveInputList", "LiveStateList", "LiveInputParamList"] {
            let list: Option<Retained<NSArray<AnyObject>>> =
                unsafe { msg_send![&*network, objectForKey: &*NSString::from_str(key)] };
            if let Some(list) = list {
                input_layouts.extend(list.iter());
            }
        }
        (executable.input_indices, executable.input_sizes) = bindings(
            &executable.inputs,
            &inputs,
            &NSArray::from_retained_slice(&input_layouts),
            false,
        )?;
        (executable.output_indices, executable.output_sizes) = bindings(
            &executable.outputs,
            &outputs,
            &*array(&network, "LiveOutputList")?,
            true,
        )?;
        Ok(executable)
    }

    pub fn run(&self, inputs: &[&TensorData], outputs: &[&TensorData]) -> Result<(), Error> {
        self.validate_tensors(inputs, outputs)?;
        self.run_surfaces(
            &inputs.iter().map(|x| x.surface()).collect::<Vec<_>>(),
            &outputs.iter().map(|x| x.surface()).collect::<Vec<_>>(),
        )
    }

    pub fn prepare(
        &self,
        inputs: &[&TensorData],
        outputs: &[&TensorData],
    ) -> Result<PreparedRequest<'_>, Error> {
        self.validate_tensors(inputs, outputs)?;
        self.prepare_surfaces(
            &inputs.iter().map(|x| x.surface()).collect::<Vec<_>>(),
            &outputs.iter().map(|x| x.surface()).collect::<Vec<_>>(),
        )
    }

    fn validate_tensors(
        &self,
        inputs: &[&TensorData],
        outputs: &[&TensorData],
    ) -> Result<(), Error> {
        for (data, specs) in [(inputs, &*self.inputs), (outputs, &*self.outputs)] {
            if data.len() != specs.len()
                || data
                    .iter()
                    .zip(specs)
                    .any(|(d, (_, s, t))| d.shape() != s.as_slice() || *t != DataType::Float32)
            {
                return Err(Error::Binding(
                    "TensorData requires matching FP32 shapes; use run_surfaces for typed I/O"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    pub fn run_surfaces(&self, inputs: &[&IOSurface], outputs: &[&IOSurface]) -> Result<(), Error> {
        let request = self.make_request(inputs, outputs, &[], &[])?;
        self.inner.evaluate(self.qos, request.inner())?;
        Ok(())
    }

    pub fn submit(
        self: &Arc<Self>,
        inputs: &[&IOSurface],
        outputs: &[&IOSurface],
        wait: &[(&AnyObject, u64)],
        signal: &[(&AnyObject, u64)],
    ) -> Result<Submission, Error> {
        if signal.is_empty() {
            return Err(Error::Binding(
                "queued ANE work must signal at least one event".into(),
            ));
        }
        let request = self.make_request(inputs, outputs, wait, signal)?;
        let state = Arc::new(Completion::default());
        let slot = state.clone();
        let handler = RcBlock::new(move |success: Bool, error: *mut NSError| {
            let result = if success.as_bool() {
                Ok(())
            } else {
                Err(unsafe { Retained::retain(error) }
                    .map(Error::from)
                    .unwrap_or(Error::Evaluation))
            };
            *slot.0.lock().unwrap() = Some(result);
            slot.1.notify_all();
        });
        request.inner().set_completion_handler(&handler);
        self.inner.evaluate(self.qos, request.inner())?;
        Ok(Submission::new(self.clone(), state, request))
    }

    pub fn prepare_surfaces(
        &self,
        inputs: &[&IOSurface],
        outputs: &[&IOSurface],
    ) -> Result<PreparedRequest<'_>, Error> {
        PreparedRequest::new(self, inputs, outputs)
    }

    fn make_request(
        &self,
        inputs: &[&IOSurface],
        outputs: &[&IOSurface],
        wait: &[(&AnyObject, u64)],
        signal: &[(&AnyObject, u64)],
    ) -> Result<Request, Error> {
        for (surfaces, sizes) in [(inputs, &self.input_sizes), (outputs, &self.output_sizes)] {
            if surfaces.len() != sizes.len()
                || surfaces
                    .iter()
                    .zip(sizes)
                    .any(|(s, n)| (s.allocationSize() as usize) < *n)
            {
                return Err(Error::Binding(
                    "surface count or allocation size differs from compiled metadata".into(),
                ));
            }
        }
        Request::new(
            inputs,
            outputs,
            &self.input_indices,
            &self.output_indices,
            wait,
            signal,
        )
    }
}
impl Drop for Executable {
    fn drop(&mut self) {
        self.inner.unload(self.qos);
    }
}
