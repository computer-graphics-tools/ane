mod compiled_executables;
mod compiled_model;
mod config;
mod error;
mod sampler;
mod session;
mod spinner;
mod weights;

use std::io::{self, Write};
use std::iter::repeat_n;
use std::time::Instant;

use tokenizers::Tokenizer;

use compiled_model::CompiledModel;
use session::Session;

const REPO_ID: &str = "openai-community/gpt2";
const PROMPT: &str = "The meaning of life is";
const MAX_NEW_TOKENS: usize = 256;
const MAX_SEQUENCE_LENGTH: usize = 1024;
const MIN_SPATIAL_WIDTH: usize = 64;
const TEMPERATURE: f32 = 0.8;
const TOP_P: f32 = 0.95;
const REPETITION_PENALTY: f32 = 1.2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args_os().nth(1).is_some() {
        return Err("usage: gpt2_forward".into());
    }
    let start = Instant::now();
    let model_files = weights::ModelFiles::download(REPO_ID)?;
    let config = model_files.config;

    let tokenizer = Tokenizer::from_file(&model_files.tokenizer_path)
        .map_err(|error| format!("tokenizer: {error}"))?;

    let encoding = tokenizer
        .encode(PROMPT, false)
        .map_err(|error| format!("encode: {error}"))?;
    let prompt_token_ids = encoding.get_ids();
    let prompt_length = prompt_token_ids.len();

    let padding_token = config.vocab_size as u32 - 1;
    let padded_length = prompt_length.div_ceil(MIN_SPATIAL_WIDTH) * MIN_SPATIAL_WIDTH;
    let padded_token_ids: Box<[u32]> = prompt_token_ids
        .iter()
        .copied()
        .chain(repeat_n(padding_token, padded_length - prompt_length))
        .collect();

    let model = CompiledModel::from_safetensors(
        config,
        model_files.safetensors_bytes,
        padded_length,
        MAX_SEQUENCE_LENGTH,
    )?;

    let mut session = Session::new(&model)?;
    let mut rng = rand::rng();
    let mut sampler = sampler::Sampler::new();

    let mut previous_text = tokenizer
        .decode(prompt_token_ids, true)
        .map_err(|error| format!("decode: {error}"))?;
    print!("{previous_text}");
    io::stdout().flush()?;
    let mut generated_tokens = prompt_token_ids.to_vec();
    let mut generation_start = Instant::now();
    for step in 0..MAX_NEW_TOKENS {
        let logits = if step == 0 {
            session.prefill(&padded_token_ids, prompt_length)?
        } else {
            session.decode_step(*generated_tokens.last().unwrap())?
        };
        let next_token = sampler.sample(
            &logits,
            TEMPERATURE,
            TOP_P,
            REPETITION_PENALTY,
            &generated_tokens,
            &mut rng,
        );
        generated_tokens.push(next_token);

        let current_text = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|error| format!("decode: {error}"))?;
        if let Some(delta) = current_text.strip_prefix(&previous_text) {
            print!("{delta}");
        }
        io::stdout().flush()?;
        previous_text = current_text;
        if step == 0 {
            generation_start = Instant::now();
        }
    }
    let decode_tokens = MAX_NEW_TOKENS - 1;
    let generation_elapsed = generation_start.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "\n\x1b[2m[{decode_tokens} decode steps in {generation_elapsed:.1}s ({:.1} tok/s) | total {:.1}s]\x1b[0m",
        decode_tokens as f64 / generation_elapsed,
        start.elapsed().as_secs_f64(),
    );
    Ok(())
}
