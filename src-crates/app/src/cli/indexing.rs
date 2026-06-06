use std::path::PathBuf;

use akuna_core::{
    config::{AppConfig, default_config_file},
    indexing::{IndexLocation, is_valid_name},
};
use anyhow::{Result, bail};
use clap::{Args, Subcommand};

/// Indexing CLI commands.
#[derive(Subcommand)]
pub(crate) enum IndexCommand {
    /// Add or update a path for indexing.
    Add(IndexAddCommand),
}

impl IndexCommand {
    /// Runs the selected indexing command.
    pub(crate) async fn run(self) -> Result<()> {
        match self {
            Self::Add(command) => command.run().await,
        }
    }
}

#[derive(Args)]
pub(crate) struct IndexAddCommand {
    /// Filesystem path to index. Defaults to current working directory.
    #[arg(short, long)]
    path: Option<PathBuf>,

    /// Name for new indexing location.
    #[arg(short, long)]
    name: String,

    /// Description for indexing location.
    #[arg(short, long)]
    description: Option<String>,
}

impl IndexAddCommand {
    /// Adds or updates a configured indexing location.
    async fn run(self) -> Result<()> {
        let path = match self.path {
            Some(path) => path,
            None => std::env::current_dir()?,
        }
        .canonicalize()?;
        let config_file = default_config_file();
        let mut config = AppConfig::load_from_file(&config_file)?;
        if !is_valid_name(&self.name) {
            bail!("name must only contain ASCII letters, numbers, '-' or '_'");
        }
        let location = IndexLocation {
            path,
            name: self.name,
            description: self.description,
        };

        if let Some(existing) = config
            .indexing
            .locations
            .iter_mut()
            .find(|existing| existing.path == location.path)
        {
            *existing = location.clone();
        } else {
            config.indexing.locations.push(location.clone());
        }

        config.save_to_file(&config_file)?;

        crate::print_json(&serde_json::json!({
            "config_file": config_file,
            "location": location,
        }))
    }
}
