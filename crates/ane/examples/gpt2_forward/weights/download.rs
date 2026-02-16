use std::fs;
use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;

use crate::config::Gpt2Config;
use crate::spinner::Spinner;

pub struct ModelFiles {
    pub config: Gpt2Config,
    pub tokenizer_path: PathBuf,
    pub safetensors_bytes: Vec<u8>,
}

pub fn download_model(repo_id: &str) -> Result<ModelFiles, Box<dyn std::error::Error>> {
    let api = ApiBuilder::new().with_progress(true).build()?;
    let repo = api.model(repo_id.to_string());

    let mut spinner = Spinner::new("Downloading config.json");
    let config_path = repo.get("config.json")?;
    let config: Gpt2Config = serde_json::from_reader(fs::File::open(&config_path)?)?;

    spinner.update("Downloading tokenizer.json");
    let tokenizer_path = repo.get("tokenizer.json")?;

    spinner.update("Downloading model.safetensors");
    let safetensors_path = repo.get("model.safetensors")?;
    let safetensors_bytes = fs::read(&safetensors_path)?;
    spinner.finish("Downloaded model files");

    Ok(ModelFiles {
        config,
        tokenizer_path,
        safetensors_bytes,
    })
}
