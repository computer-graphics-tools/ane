use std::fs;
use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;

use crate::config::Gpt2Config;
use crate::error::Error;
use crate::spinner::Spinner;

pub struct ModelFiles {
    pub config: Gpt2Config,
    pub tokenizer_path: PathBuf,
    pub safetensors_bytes: Vec<u8>,
}

impl ModelFiles {
    pub fn download(repo_id: &str) -> Result<Self, Error> {
        let builder = ApiBuilder::new().with_progress(true);
        #[cfg(target_os = "ios")]
        let builder = {
            use objc2_foundation::{
                NSSearchPathDirectory, NSSearchPathDomainMask, NSSearchPathForDirectoriesInDomains,
            };
            let paths = NSSearchPathForDirectoriesInDomains(
                NSSearchPathDirectory::CachesDirectory,
                NSSearchPathDomainMask::UserDomainMask,
                true,
            );
            let cache = paths.firstObject().ok_or(Error::CacheDirectory)?;
            builder.with_cache_dir(PathBuf::from(cache.to_string()).join("huggingface"))
        };
        let api = builder.build()?;
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
}
