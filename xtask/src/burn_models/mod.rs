//! Burn model asset maintenance tasks.
//!
//! Each submodule owns its full lifecycle (build + push) and targets a
//! dedicated Hugging Face repository. New model families are added by
//! dropping a new sibling module here and wiring its `build`/`push` into
//! the top-level delegates below.

pub(crate) mod pp_ocrv6;

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

const MODEL_ROOT: &str = "target/models";
const HF_MODELS_OWNER_ENV: &str = "HF_OWNER";
const HF_TOKEN_ENV: &str = "HF_TOKEN";
/// Hugging Face URL prefix for upstream repo links.
const HF_BASE_URL: &str = "https://huggingface.co/";

/// Builds all burn model artifacts.
pub(crate) fn build() -> Result<()> {
    pp_ocrv6::build()
}

/// Pushes all burn model artifacts to their dedicated HF repos.
pub(crate) fn push() -> Result<()> {
    pp_ocrv6::push()
}

/// Resolves required HF token for pushing to burn model repos.
///
/// # Errors
///
/// Returns an error when `HF_TOKEN` is unset or non-UTF8.
fn hf_token() -> Result<String> {
    std::env::var(HF_TOKEN_ENV).with_context(|| {
        format!("{HF_TOKEN_ENV} must be set to push burn model assets")
    })
}

/// Resolves required HF owner/organization for burn model repos.
///
/// # Errors
///
/// Returns an error when `AKUNA_BURN_MODELS_OWNER` is unset or non-UTF8.
fn hf_owner() -> Result<String> {
    std::env::var(HF_MODELS_OWNER_ENV).with_context(|| {
        format!("{HF_MODELS_OWNER_ENV} must be set to push burn model assets")
    })
}

/// Resolves repository output root for generated model artifacts.
fn model_root() -> Result<std::path::PathBuf> {
    crate::repo_root().map(|repo| repo.join(MODEL_ROOT))
}

/// Prints a deterministic, repository-relative path for each emitted file.
///
/// Falls back to absolute path only when caller did not provide a path under
/// the model root.
fn print_relative(base_dir: &Path, path: &Path) {
    if let Ok(relative) = path.strip_prefix(base_dir) {
        println!("models/{relative}", relative = relative.display());
        return;
    }

    println!("{}", path.display());
}

/// Writes `parent_folder/child_path` under `target/models` and logs location.
///
/// Centralize writes here so build pipelines always emit the same output
/// format and consistent path tracing in logs.
pub(super) fn write_file(
    parent_folder: &str,
    child_path: impl AsRef<Path>,
    bytes: impl AsRef<[u8]>,
) -> Result<()> {
    let root = model_root()?;
    let output = root.join(parent_folder).join(child_path.as_ref());

    if let Some(parent_dir) = output.parent() {
        fs::create_dir_all(parent_dir).with_context(|| {
            format!("failed to create {}", parent_dir.display())
        })?;
    }

    fs::write(&output, bytes)
        .with_context(|| format!("failed to write {}", output.display()))?;
    print_relative(&root, &output);

    Ok(())
}
