use std::collections::HashMap;
use std::sync::Arc;

use objc2_io_surface::IOSurface;

use crate::apple_neural_engine::{ModelAttributes, SurfaceLayout};
use crate::{
    DataType, Error, LoadedProgram, Program, Request, StateData, Tensor, TensorData, TensorSpec,
    require,
};

#[derive(Clone)]
pub struct Procedure {
    index: u32,
    program: Arc<LoadedProgram>,
    inputs: Box<[TensorSpec]>,
    outputs: Box<[TensorSpec]>,
    input_indices: Box<[u32]>,
    output_indices: Box<[u32]>,
    input_writes: Box<[bool]>,
    feed_tensors: Box<[Tensor]>,
    target_tensors: Box<[Tensor]>,
    feeds: Box<[Tensor]>,
    states: Box<[(Tensor, TensorData)]>,
}

impl Procedure {
    pub fn new(
        function: &Program,
        outputs: &[String],
        attributes: &ModelAttributes,
        program: Arc<LoadedProgram>,
        graph_states: &HashMap<Tensor, StateData>,
        states: &mut HashMap<Tensor, TensorData>,
    ) -> Result<Self, Error> {
        let description = &attributes.description;
        let index = *description
            .procedure_ids
            .get(function.name())
            .ok_or(Error::Metadata("procedure"))?;
        let procedure = description
            .procedures
            .iter()
            .find(|procedure| procedure.id == index)
            .ok_or(Error::Metadata("procedure"))?;
        let network = attributes
            .networks
            .iter()
            .find(|network| network.name == function.name())
            .ok_or(Error::Metadata("network"))?;
        let input_symbols: Vec<_> = function
            .inputs()
            .iter()
            .map(|(name, ..)| name.clone())
            .collect();
        let output_symbols: Vec<_> = outputs
            .iter()
            .map(|output| format!("{output}@output"))
            .collect();
        let input_layouts: Vec<_> = network
            .inputs
            .iter()
            .chain(&network.states)
            .chain(&network.parameters)
            .collect();
        let (input_indices, inputs) = bindings(
            function.inputs(),
            function.feed_tensors(),
            &input_symbols,
            &description.input_symbols,
            &procedure.inputs,
            &input_layouts,
        )?;
        let (output_indices, outputs) = bindings(
            function.outputs(),
            function.target_tensors(),
            &output_symbols,
            &description.output_symbols,
            &procedure.outputs,
            &network.outputs.iter().collect::<Vec<_>>(),
        )?;
        let mut bound = Vec::new();
        for (tensor, spec) in function.feed_tensors().iter().zip(&inputs) {
            let Some(data) = graph_states.get(tensor) else {
                continue;
            };
            let data = match states.get(tensor) {
                Some(data) => data.clone(),
                None => data.initialize(spec)?,
            };
            spec.validate(&data)?;
            states.insert(*tensor, data.clone());
            bound.push((*tensor, data));
        }
        Ok(Self {
            index,
            program,
            input_writes: input_symbols
                .iter()
                .map(|symbol| network.states.iter().any(|state| &state.name == symbol))
                .collect(),
            inputs,
            outputs,
            input_indices: input_indices.into(),
            output_indices: output_indices.into(),
            feeds: function
                .feed_tensors()
                .iter()
                .filter(|tensor| !graph_states.contains_key(tensor))
                .copied()
                .collect(),
            feed_tensors: function.feed_tensors().into(),
            target_tensors: function.target_tensors().into(),
            states: bound.into(),
        })
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
    pub fn allocate_outputs(&self) -> Result<Vec<TensorData>, Error> {
        Ok(self
            .outputs
            .iter()
            .map(TensorSpec::allocate)
            .collect::<Result<_, _>>()?)
    }

    pub fn request(
        &self,
        inputs: &[&TensorData],
        outputs: &[&TensorData],
    ) -> Result<Arc<Request>, Error> {
        for (kind, expected, actual) in [
            ("inputs", self.feeds.len(), inputs.len()),
            ("outputs", self.outputs.len(), outputs.len()),
        ] {
            require(
                expected == actual,
                Error::Count {
                    kind,
                    expected,
                    actual,
                },
            )?;
        }
        let mut provided = inputs.iter().copied();
        let bound: Vec<&TensorData> = self
            .feed_tensors
            .iter()
            .map(|tensor| {
                self.states
                    .iter()
                    .find_map(|(state, data)| (state == tensor).then_some(data))
                    .or_else(|| provided.next())
            })
            .collect::<Option<_>>()
            .ok_or(Error::Unbound("input"))?;
        let bindings = inputs
            .iter()
            .chain(outputs)
            .map(|data| data.surface().surfaceID())
            .collect();
        for (spec, data) in self.inputs.iter().zip(&bound) {
            spec.validate(data)?;
        }
        for (spec, data) in self.outputs.iter().zip(outputs) {
            spec.validate(data)?;
        }
        let inputs: Vec<(&IOSurface, u32, bool)> = bound
            .iter()
            .zip(&self.input_indices)
            .zip(&self.input_writes)
            .map(|((data, index), write)| (data.surface(), *index, *write))
            .collect();
        let outputs: Vec<(&IOSurface, u32)> = outputs
            .iter()
            .zip(&self.output_indices)
            .map(|(data, index)| (data.surface(), *index))
            .collect();
        Request::new(
            self.program.clone(),
            self.index,
            &inputs,
            &outputs,
            bindings,
        )
    }
}

fn bindings(
    specs: &[(String, [usize; 4], DataType)],
    tensors: &[Tensor],
    symbols: &[String],
    names: &[String],
    indices: &[u32],
    layouts: &[&SurfaceLayout],
) -> Result<(Vec<u32>, Box<[TensorSpec]>), Error> {
    let mut bound = Vec::new();
    let mut sizes = Vec::new();
    for (((name, shape, dtype), tensor), symbol) in specs.iter().zip(tensors).zip(symbols) {
        let mismatch = |field| Error::Layout {
            symbol: symbol.clone(),
            field,
        };
        let index = *indices
            .iter()
            .find(|&&index| names.get(index as usize) == Some(symbol))
            .ok_or_else(|| mismatch("symbol"))?;
        let layout = layouts
            .iter()
            .find(|layout| &layout.name == symbol)
            .ok_or_else(|| mismatch("layout"))?;
        if layout.storage_type != dtype.storage_name() {
            return Err(mismatch("storage type"));
        }
        bound.push(index);
        let spec = TensorSpec::new(name, shape, *dtype)?;
        if *dtype == DataType::Int32 {
            if shape.iter().product::<usize>() != 1 {
                return Err(mismatch("scalar layout"));
            }
            sizes.push(spec.with_shape(tensor.shape())?);
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
        let strides = [
            integer(layout.batch_stride, "BatchStride")?,
            integer(layout.plane_stride, "PlaneStride")?,
            integer(layout.row_stride, "RowStride")?,
            dtype.byte_width(),
        ];
        let bytes = strides[0]
            .checked_mul(shape[0])
            .ok_or_else(|| mismatch("allocation size"))?;
        sizes.push(
            spec.with_layout(strides, bytes)?
                .with_shape(tensor.shape())?,
        );
    }
    Ok((bound, sizes.into()))
}
