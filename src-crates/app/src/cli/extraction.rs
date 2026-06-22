use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::print_json;

/// CLI arguments for the `extract` command.
#[derive(Args)]
pub(crate) struct ExtractCommand {
    /// Path to the file to extract.
    file: PathBuf,
    /// Include detected file metadata in the result.
    #[arg(long)]
    metadata: bool,
    /// Include extracted text content in the result.
    #[arg(long)]
    content: bool,
    /// Include extracted text chunks in the result.
    #[arg(long)]
    chunks: bool,
}

impl ExtractCommand {
    /// Runs file extraction and prints the result.
    pub(crate) async fn run(self) -> Result<()> {
        tracing::info!("extracting data from {}", self.file.display());

        let full = !self.metadata && !self.content && !self.chunks;

        let extraction = akuna_core::extraction::extract_file(
            self.file,
            &akuna_core::extraction::ExtractionConfig {
                return_metadata: full || self.metadata,
                return_content: full || self.content,
                return_chunking: full || self.chunks,
                text: Some(
                    akuna_core::extraction::TextExtractionConfig::default(),
                ),
                chunking: Some(akuna_core::chunking::ChunkingConfig::default()),
            },
        )
        .await?;

        print_json(&extraction)
    }
}
