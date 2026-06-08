//! Schema CLI commands.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Subcommand;

use crate::config::{AppConfig, config_schema_file_name};

const DEFAULT_SCHEMA_OUT_DIR: &str = "target";

/// Schema CLI commands.
#[derive(Subcommand)]
pub(crate) enum SchemasCommand {
    /// Generate app-owned schemas.
    Generate {
        /// Directory to write generated artifacts into.
        #[arg(short, long)]
        path: Option<PathBuf>,
    },
}

impl SchemasCommand {
    /// Runs selected schema command.
    pub(crate) async fn run(self) -> Result<()> {
        match self {
            Self::Generate { path: out } => generate_schemas(out),
        }
    }
}

/// Generates schema artifacts into target directory.
fn generate_schemas(out_dir: Option<PathBuf>) -> Result<()> {
    let out_dir = match out_dir {
        Some(out_dir) => out_dir,
        None => default_schema_out_dir()?,
    };

    std::fs::create_dir_all(&out_dir)
        .with_context(|| format!("Failed to create {}", out_dir.display()))?;

    let config_schema_path =
        AppConfig::generate_schema(out_dir.join(config_schema_file_name()))?;
    let openapi_schema_path = crate::api::server::generate_schema(&out_dir)?;

    crate::print_json(&serde_json::json!({
        "config_schema": config_schema_path,
        "openapi_schema": openapi_schema_path,
    }))
}

/// Returns default schema output directory.
fn default_schema_out_dir() -> Result<PathBuf> {
    let current_dir =
        std::env::current_dir().context("Failed to read current directory")?;
    let workspace_dir = workspace_dir();

    if current_dir.starts_with(&workspace_dir) {
        return Ok(workspace_dir.join(DEFAULT_SCHEMA_OUT_DIR));
    }

    Ok(current_dir)
}

/// Returns compile-time workspace directory.
fn workspace_dir() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.to_path_buf())
}
