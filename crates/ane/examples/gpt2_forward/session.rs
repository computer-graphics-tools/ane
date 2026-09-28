use ane::{IOSurfaceExt, LockedSlice, PreparedRequest, TensorData};
use objc2::rc::Retained;
use objc2_io_surface::IOSurface;

use crate::compiled_model::CompiledModel;
use crate::error::Error;

pub struct Session<'model> {
    model: &'model CompiledModel,
    prefill_request: PreparedRequest<'model>,
    decode_request: PreparedRequest<'model>,
    prefill_input: TensorData,
    decode_input: TensorData,
    selector: TensorData,
    decode_mask: TensorData,
    cache_position: Retained<IOSurface>,
    output: TensorData,
    position: usize,
}

impl<'model> Session<'model> {
    pub fn new(model: &'model CompiledModel) -> Result<Self, Error> {
        let sequence = model.padded_prompt_length;
        let context = model.max_sequence_length;
        let prefill_input = TensorData::new(&[1, model.config.n_embd, 1, sequence]);
        let decode_input = TensorData::new(&[1, 1, 1, model.config.n_embd]);
        let prefill_mask = TensorData::with_f32(
            &(0..sequence * context)
                .map(|i| {
                    if i % context <= i / context {
                        0.0
                    } else {
                        -40_000.0
                    }
                })
                .collect::<Vec<_>>(),
            &[1, 1, sequence, context],
        );
        let decode_mask = TensorData::new(&[1, 1, 1, context]);
        let selector = TensorData::new(&[1, 1, 1, sequence]);
        let cache_position = IOSurface::with_byte_count(4);
        let cache_bytes = vec![0; model.config.n_layer * model.config.n_embd * context * 2];
        let cache = [(); 2].map(|_| {
            let surface = IOSurface::with_byte_count(cache_bytes.len());
            surface.write_bytes(&cache_bytes);
            surface
        });
        let output = TensorData::new(&[1, 1, 1, model.config.vocab_size]);
        let mut inputs = vec![
            prefill_input.surface(),
            prefill_mask.surface(),
            &*cache_position,
            selector.surface(),
        ];
        inputs.extend(cache.iter().map(|s| &**s));
        let prefill_request = model
            .executables
            .prefill
            .prepare_surfaces(&inputs, &[output.surface()])?;
        let mut inputs = vec![
            decode_input.surface(),
            decode_mask.surface(),
            &*cache_position,
        ];
        inputs.extend(cache.iter().map(|s| &**s));
        let decode_request = model
            .executables
            .decode
            .prepare_surfaces(&inputs, &[output.surface()])?;
        Ok(Self {
            model,
            prefill_request,
            decode_request,
            prefill_input,
            decode_input,
            selector,
            decode_mask,
            cache_position,
            output,
            position: 0,
        })
    }

    pub fn prefill(
        &mut self,
        token_ids: &[u32],
        real_prompt_length: usize,
    ) -> Result<LockedSlice<'_>, Error> {
        let e = self.model.config.n_embd;
        let sequence = self.model.padded_prompt_length;
        if token_ids.len() != sequence
            || real_prompt_length == 0
            || real_prompt_length > sequence
            || token_ids
                .iter()
                .any(|&t| t as usize >= self.model.config.vocab_size)
        {
            return Err(Error::Input("prompt shape, length or token ID is invalid"));
        }
        {
            let mut input = self.prefill_input.as_f32_slice_mut();
            for (position, &token) in token_ids.iter().enumerate() {
                for channel in 0..e {
                    input[channel * sequence + position] = self.model.token_embeddings
                        [token as usize * e + channel]
                        + self.model.position_embeddings[position * e + channel];
                }
            }
            let mut selector = self.selector.as_f32_slice_mut();
            selector.fill(0.0);
            selector[real_prompt_length - 1] = 1.0;
        }
        self.cache_position.write_bytes(&0i32.to_le_bytes());
        self.position = 0;
        self.prefill_request.run()?;
        self.position = real_prompt_length;
        {
            let mut mask = self.decode_mask.as_f32_slice_mut();
            mask.fill(-40_000.0);
            mask[..self.position].fill(0.0);
        }
        Ok(self.output.as_f32_slice())
    }

    pub fn decode_step(&mut self, token: u32) -> Result<LockedSlice<'_>, Error> {
        if self.position == 0
            || self.position >= self.model.max_sequence_length
            || token as usize >= self.model.config.vocab_size
        {
            return Err(Error::Input(
                "prefill is required, context is full, or token ID is invalid",
            ));
        }
        let e = self.model.config.n_embd;
        {
            let mut input = self.decode_input.as_f32_slice_mut();
            for channel in 0..e {
                input[channel] = self.model.token_embeddings[token as usize * e + channel]
                    + self.model.position_embeddings[self.position * e + channel];
            }
        }
        self.decode_mask.as_f32_slice_mut()[self.position] = 0.0;
        self.cache_position
            .write_bytes(&(self.position as i32).to_le_bytes());
        self.decode_request
            .run()
            .inspect_err(|_| self.position = 0)?;
        self.position += 1;
        Ok(self.output.as_f32_slice())
    }
}
