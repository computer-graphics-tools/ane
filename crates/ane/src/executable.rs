#![deny(missing_docs)]

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use objc2_foundation::NSQualityOfService;

use crate::ir::MilProgram;
use crate::{
    CompilationReport, Error, ExecutionDescriptor, LoadedProgram, Procedure, Program, Request,
    RequestCache, StateData, Submission, Tensor, TensorData, TensorSpec,
};

/// A compiled ANE program for one set of targets, run with `run` or `run_async`. Executables from
/// one `compile_shared` call share a model, its constants and its variables.
#[derive(Clone)]
pub struct Executable {
    program: Arc<LoadedProgram>,
    procedure: Procedure,
    variables: HashMap<Tensor, TensorData>,
    operations: Arc<[String]>,
    mil_bytes: usize,
    constant_bytes: usize,
    requests: Arc<RequestCache>,
}

impl Executable {
    /// Loads `functions` as one ANE model with one executable per function.
    pub fn compile(
        functions: &[Program],
        graph_states: &HashMap<Tensor, StateData>,
        qos: NSQualityOfService,
    ) -> Result<Vec<Self>, Error> {
        let operations: Arc<[String]> = functions.iter().flat_map(Program::operations).collect();
        let mil = MilProgram::new(functions)?;
        let mil_bytes = mil.text.len() + mil.weights.len();
        let outputs = mil.outputs.clone();
        let program = Arc::new(LoadedProgram::new(mil, qos).map_err(|source| {
            Error::Compilation {
                operations: operations
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(", "),
                source: Box::new(source),
            }
        })?);
        let attributes = program.model_attributes()?;
        let mut variables = HashMap::new();
        let procedures = functions
            .iter()
            .zip(&outputs)
            .map(|(function, outputs)| {
                Procedure::new(
                    function,
                    outputs,
                    &attributes,
                    program.clone(),
                    graph_states,
                    &mut variables,
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let constant_bytes = functions.iter().map(Program::constant_bytes).sum();
        Ok(procedures
            .into_iter()
            .map(|procedure| Self {
                program: program.clone(),
                procedure,
                variables: variables.clone(),
                operations: operations.clone(),
                mil_bytes,
                constant_bytes,
                requests: Arc::default(),
            })
            .collect())
    }
    /// Deletes this model from the ANE compiler cache; the loaded model keeps running, and the
    /// next compile of the same graph compiles from scratch.
    pub fn purge_compiled_model(&self) -> Result<(), Error> {
        self.program.purge()
    }
    /// Operation names, MIL size and constant bytes of the compiled model.
    pub fn report(&self) -> CompilationReport<'_> {
        CompilationReport {
            operations: &self.operations,
            mil_bytes: self.mil_bytes,
            constant_bytes: self.constant_bytes,
        }
    }
    /// Inputs that `run` takes, in order; graph-owned variables are bound automatically.
    pub fn feed_tensors(&self) -> &[Tensor] {
        self.procedure.input_tensors()
    }
    /// Target tensors in result order.
    pub fn target_tensors(&self) -> &[Tensor] {
        self.procedure.output_tensors()
    }
    /// Layout of an input; `allocate` makes a matching IOSurface.
    pub fn input(&self, tensor: Tensor) -> Result<&TensorSpec, Error> {
        self.procedure.input(tensor)
    }
    /// Layout of a result; `allocate` makes a matching IOSurface.
    pub fn output(&self, tensor: Tensor) -> Result<&TensorSpec, Error> {
        self.procedure.output(tensor)
    }
    /// New result buffers in target order.
    pub fn allocate_outputs(&self) -> Result<Vec<TensorData>, Error> {
        self.procedure.allocate_outputs()
    }
    /// IOSurface storage of a variable the graph initialised.
    pub fn variable_data(&self, variable: &Tensor) -> Result<&TensorData, Error> {
        self.variables
            .get(variable)
            .ok_or(Error::Unbound("variable"))
    }

    /// Runs with `inputs` in `feed_tensors()` order and returns results in target order. Passing
    /// `results` reuses their IOSurfaces and the request mapped for this exact binding set.
    pub fn run(
        &self,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        descriptor: Option<&ExecutionDescriptor<'_>>,
    ) -> Result<Box<[TensorData]>, Error> {
        let (request, results) = self.request(inputs, results)?;
        match descriptor {
            Some(descriptor) => request.submit(Some(descriptor))?.wait()?,
            None => request.run()?,
        }
        Ok(results)
    }

    /// Like `run`, but returns once the request is queued; results are valid when the submission
    /// finishes.
    pub fn run_async(
        &self,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
        descriptor: Option<&ExecutionDescriptor<'_>>,
    ) -> Result<Submission, Error> {
        let (request, results) = self.request(inputs, results)?;
        Ok(Submission::new(request.submit(descriptor)?, results))
    }

    fn request(
        &self,
        inputs: &[&TensorData],
        results: Option<&[&TensorData]>,
    ) -> Result<(Arc<Request>, Box<[TensorData]>), Error> {
        let Some(results) = results else {
            let results = self.procedure.allocate_outputs()?;
            let bound: Vec<_> = results.iter().collect();
            return Ok((self.procedure.request(inputs, &bound)?, results.into()));
        };
        let bindings: Vec<_> = inputs
            .iter()
            .chain(results)
            .map(|data| data.surface().surfaceID())
            .collect();
        let request = self
            .requests
            .get_or_insert(&bindings, || self.procedure.request(inputs, results))?;
        Ok((
            request,
            results.iter().map(|data| (*data).clone()).collect(),
        ))
    }
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Executable>();
};
