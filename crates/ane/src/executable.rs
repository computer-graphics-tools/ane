use std::collections::HashMap;
use std::sync::Arc;

use objc2_foundation::NSQualityOfService;

use crate::apple_neural_engine::SurfaceLayout;
use crate::request::Request;
use crate::{
    CompilationReport, DataType, Error, ExecutionDescriptor, LoadedProgram, NativeOutputs, Program,
    RequestCache, Submission, Tensor, TensorData, TensorSpec, padded_shape, require,
};

#[derive(Clone)]
pub struct Executable {
    inner: Arc<LoadedProgram>,
    inputs: Box<[TensorSpec]>,
    outputs: Box<[TensorSpec]>,
    input_indices: Vec<u32>,
    output_indices: Vec<u32>,
    input_writes: Vec<bool>,
    extra_outputs: NativeOutputs,
    operations: Arc<[String]>,
    anec_bytes: usize,
    constant_bytes: usize,
    feed_tensors: Box<[Tensor]>,
    target_tensors: Box<[Tensor]>,
    variables: HashMap<Tensor, TensorData>,
    feeds: Box<[Tensor]>,
    requests: Arc<RequestCache>,
}

fn bindings(
    specs: &[(String, [usize; 4], DataType)],
    names: &[String],
    layouts: &[&SurfaceLayout],
    symbols: &[String],
) -> Result<(Vec<u32>, Box<[TensorSpec]>), Error> {
    require(
        symbols.len() == specs.len(),
        Error::Count {
            kind: "compiled tensors",
            expected: specs.len(),
            actual: symbols.len(),
        },
    )?;
    let mut indices = Vec::new();
    let mut sizes = Vec::new();
    for ((name, shape, dtype), symbol) in specs.iter().zip(symbols) {
        let mismatch = |field| Error::Layout {
            symbol: symbol.clone(),
            field,
        };
        let index = names
            .iter()
            .position(|n| n == symbol)
            .ok_or_else(|| mismatch("symbol"))?;
        let layout = layouts
            .iter()
            .find(|layout| &layout.name == symbol)
            .ok_or_else(|| mismatch("layout"))?;
        if layout.storage_type != dtype.storage_name() {
            return Err(mismatch("storage type"));
        }
        if *dtype == DataType::Int32 {
            if shape.iter().product::<usize>() != 1 {
                return Err(mismatch("scalar layout"));
            }
            indices.push(index as u32);
            sizes.push(TensorSpec::new(name, shape, *dtype)?);
            continue;
        }
        let integer = |value: Option<usize>, key| value.ok_or_else(|| mismatch(key));
        if integer(layout.width, "Width")? != shape[3]
            || integer(layout.height, "Height")? != shape[2]
            || integer(layout.channels, "Channels")? != shape[1]
            || integer(layout.batches, "Batches")? != shape[0]
            || integer(layout.depth, "Depth")? != 1
        {
            return Err(mismatch("dimensions"));
        }
        indices.push(index as u32);
        let strides = [
            integer(layout.batch_stride, "BatchStride")?,
            integer(layout.plane_stride, "PlaneStride")?,
            integer(layout.row_stride, "RowStride")?,
            dtype.byte_width(),
        ];
        let bytes = strides[0]
            .checked_mul(shape[0])
            .ok_or_else(|| mismatch("allocation size"))?;
        sizes.push(TensorSpec::new(name, shape, *dtype)?.with_layout(strides, bytes)?);
    }
    Ok((indices, sizes.into()))
}

impl Executable {
    pub fn report(&self) -> CompilationReport<'_> {
        CompilationReport {
            inputs: &self.inputs,
            outputs: &self.outputs,
            operations: &self.operations,
            anec_bytes: self.anec_bytes,
            constant_bytes: self.constant_bytes,
        }
    }
    pub fn with_graph_tensors(
        mut self,
        inputs: &[Tensor],
        outputs: &[Tensor],
        variables: HashMap<Tensor, TensorData>,
    ) -> Result<Self, Error> {
        for (tensors, specs, kind) in [
            (inputs, &*self.inputs, "graph inputs"),
            (outputs, &*self.outputs, "graph outputs"),
        ] {
            require(
                tensors.len() == specs.len(),
                Error::Count {
                    kind,
                    expected: specs.len(),
                    actual: tensors.len(),
                },
            )?;
            if let Some(spec) = tensors.iter().zip(specs).find_map(|(t, s)| {
                (s.name() != t.symbol() && s.name() != format!("{}_result", t.symbol())
                    || padded_shape(t.shape()) != padded_shape(s.shape()))
                .then_some(s)
            }) {
                return Err(Error::Layout {
                    symbol: spec.name().into(),
                    field: "graph tensor",
                });
            }
        }
        for (tensor, data) in &variables {
            let index = inputs
                .iter()
                .position(|t| t == tensor)
                .ok_or(Error::Unbound("input"))?;
            self.inputs[index].validate(data)?;
        }
        self.feeds = inputs
            .iter()
            .filter(|tensor| !variables.contains_key(tensor))
            .copied()
            .collect();
        self.feed_tensors = inputs.into();
        self.target_tensors = outputs.into();
        self.variables = variables;
        self.requests = Arc::default();
        Ok(self)
    }
    pub fn input_tensors(&self) -> &[Tensor] {
        &self.feeds
    }
    pub fn output_tensors(&self) -> &[Tensor] {
        &self.target_tensors
    }
    pub fn input(&self, tensor: Tensor) -> Result<&TensorSpec, Error> {
        self.feed_tensors
            .iter()
            .position(|t| *t == tensor)
            .map(|i| &self.inputs[i])
            .ok_or(Error::Unbound("input"))
    }
    pub fn output(&self, tensor: Tensor) -> Result<&TensorSpec, Error> {
        self.target_tensors
            .iter()
            .position(|t| *t == tensor)
            .map(|i| &self.outputs[i])
            .ok_or(Error::Unbound("output"))
    }
    pub fn variable_data(&self, variable: &Tensor) -> Result<&TensorData, Error> {
        self.variables
            .get(variable)
            .ok_or(Error::Unbound("variable"))
    }
    pub fn inputs(&self) -> &[TensorSpec] {
        &self.inputs
    }
    pub fn outputs(&self) -> &[TensorSpec] {
        &self.outputs
    }
    pub fn allocate_inputs(&self) -> Result<Vec<TensorData>, Error> {
        self.feeds
            .iter()
            .map(|tensor| Ok(self.input(*tensor)?.allocate()?))
            .collect()
    }
    pub fn allocate_outputs(&self) -> Result<Vec<TensorData>, Error> {
        Ok(self
            .outputs
            .iter()
            .map(TensorSpec::allocate)
            .collect::<Result<_, _>>()?)
    }
    pub fn with_tensor_shapes(
        mut self,
        inputs: &[&[usize]],
        outputs: &[&[usize]],
    ) -> Result<Self, Error> {
        for (shapes, specs) in [(inputs, &self.inputs), (outputs, &self.outputs)] {
            require(
                shapes.len() == specs.len(),
                Error::Count {
                    kind: "tensor shapes",
                    expected: specs.len(),
                    actual: shapes.len(),
                },
            )?;
        }
        self.inputs = self
            .inputs
            .iter()
            .zip(inputs)
            .map(|(s, shape)| s.clone().with_shape(shape))
            .collect::<Result<_, _>>()?;
        self.outputs = self
            .outputs
            .iter()
            .zip(outputs)
            .map(|(s, shape)| s.clone().with_shape(shape))
            .collect::<Result<_, _>>()?;
        Ok(self)
    }

    pub fn compile(program: Program, qos: NSQualityOfService) -> Result<Self, Error> {
        let operations: Arc<[String]> = program.operations().into();
        let inner = Arc::new(LoadedProgram::new(&program, qos).map_err(|source| {
            Error::Compilation {
                operations: operations
                    .iter()
                    .map(String::as_str)
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(", "),
                source: Box::new(source),
            }
        })?);
        let anec_bytes = inner.anec_bytes;
        let constant_bytes = program.constant_bytes();
        let input_specs = program.inputs().to_vec();
        let output_specs = program.outputs().to_vec();
        let mut executable = Self {
            inner,
            inputs: Box::new([]),
            outputs: Box::new([]),
            input_indices: Vec::new(),
            output_indices: Vec::new(),
            input_writes: Vec::new(),
            extra_outputs: NativeOutputs::default(),
            operations,
            anec_bytes,
            constant_bytes,
            feed_tensors: Box::new([]),
            target_tensors: Box::new([]),
            variables: HashMap::new(),
            feeds: Box::new([]),
            requests: Arc::default(),
        };
        let attributes = executable.inner.model_attributes()?;
        let network = attributes
            .networks
            .first()
            .ok_or(Error::Metadata("network"))?;
        executable.input_writes = input_specs
            .iter()
            .enumerate()
            .map(|(i, _)| {
                network
                    .states
                    .iter()
                    .any(|state| state.name == executable.inner.input_symbols[i])
                    || executable
                        .inner
                        .state_outputs
                        .iter()
                        .any(|(_, index)| *index == i)
            })
            .collect();
        let input_layouts: Vec<_> = network
            .inputs
            .iter()
            .chain(&network.states)
            .chain(&network.parameters)
            .collect();
        let output_layouts: Vec<_> = network.outputs.iter().collect();
        (executable.input_indices, executable.inputs) = bindings(
            &input_specs,
            &attributes.description.input_symbols,
            &input_layouts,
            &executable.inner.input_symbols,
        )?;
        (executable.output_indices, executable.outputs) = bindings(
            &output_specs,
            &attributes.description.output_symbols,
            &output_layouts,
            &executable.inner.output_symbols,
        )?;
        for (symbol, input) in &executable.inner.state_outputs {
            let (indices, specs) = bindings(
                &input_specs[*input..*input + 1],
                &attributes.description.output_symbols,
                &output_layouts,
                std::slice::from_ref(symbol),
            )?;
            if specs[0].strides() != executable.inputs[*input].strides()
                || specs[0].allocation_size() != executable.inputs[*input].allocation_size()
            {
                return Err(Error::Layout {
                    symbol: symbol.clone(),
                    field: "state layout",
                });
            }
            executable.extra_outputs.states.push((indices[0], *input));
        }
        let symbols = executable
            .inner
            .discarded_outputs
            .iter()
            .map(|(s, _, _)| s.clone())
            .collect::<Vec<_>>();
        let (indices, specs) = bindings(
            &executable.inner.discarded_outputs,
            &attributes.description.output_symbols,
            &output_layouts,
            &symbols,
        )?;
        executable.extra_outputs.discarded = indices.into_iter().zip(Vec::from(specs)).collect();
        Ok(executable)
    }

    pub fn run(
        &self,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        descriptor: Option<&ExecutionDescriptor<'_>>,
    ) -> Result<Box<[TensorData]>, Error> {
        let (request, results) = self.request(inputs, results)?;
        match descriptor {
            Some(descriptor)
                if !descriptor.wait_events.is_empty() || !descriptor.signal_events.is_empty() =>
            {
                request
                    .submit(descriptor.wait_events, descriptor.signal_events)?
                    .wait()?
            }
            _ => request.run()?,
        }
        Ok(results)
    }

    pub fn run_async(
        &self,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        descriptor: Option<&ExecutionDescriptor<'_>>,
    ) -> Result<Submission, Error> {
        let (request, results) = self.request(inputs, results)?;
        let descriptor = descriptor.copied().unwrap_or_default();
        let state = request.submit(descriptor.wait_events, descriptor.signal_events)?;
        Ok(Submission::new(state, results))
    }

    fn request(
        &self,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
    ) -> Result<(Arc<Request>, Box<[TensorData]>), Error> {
        let input_count = Error::Count {
            kind: "inputs",
            expected: self.feeds.len(),
            actual: inputs.len(),
        };
        if inputs.len() != self.feeds.len() {
            return Err(input_count);
        }
        let allocated;
        let results = match results {
            Some(results) => results.to_vec(),
            None => {
                allocated = self.allocate_outputs()?;
                allocated.iter().collect()
            }
        };
        require(
            results.len() == self.outputs.len(),
            Error::Count {
                kind: "results",
                expected: self.outputs.len(),
                actual: results.len(),
            },
        )?;
        let mut provided = inputs.iter().copied();
        let bound: Vec<&TensorData> = self
            .feed_tensors
            .iter()
            .map(|tensor| self.variables.get(tensor).or_else(|| provided.next()))
            .collect::<Option<_>>()
            .ok_or(input_count)?;
        for (spec, data) in self.inputs.iter().zip(&bound) {
            spec.validate(data)?;
        }
        for (spec, data) in self.outputs.iter().zip(&results) {
            spec.validate(data)?;
        }
        let bindings: Vec<_> = bound
            .iter()
            .chain(&results)
            .map(|data| data.surface().surfaceID())
            .collect();
        let request = self.requests.get_or_insert(&bindings, || {
            let inputs: Vec<_> = bound.iter().map(|data| data.surface()).collect();
            let outputs: Vec<_> = results.iter().map(|data| data.surface()).collect();
            let mut request = Request::new(
                self.inner.clone(),
                &inputs,
                &outputs,
                &self.input_indices,
                &self.output_indices,
                &self.input_writes,
                &self.extra_outputs,
            )?;
            request.map()?;
            Ok(request)
        })?;
        Ok((request, results.into_iter().cloned().collect()))
    }
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Executable>();
};
