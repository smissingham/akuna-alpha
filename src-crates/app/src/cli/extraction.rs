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
    /// Print extracted text.
    #[arg(long)]
    text: bool,
    /// Include structured content parts in the result.
    #[arg(long)]
    parts: bool,
}

impl ExtractCommand {
    /// Runs file extraction and prints the result.
    pub(crate) async fn run(self) -> Result<()> {
        if self.text && !self.metadata && !self.parts {
            let text =
                akuna_core::extraction::extract_file_text(self.file).await?;
            print!("{text}");
            return Ok(());
        }

        tracing::info!("extracting data from {}", self.file.display());

        let full = !self.metadata && !self.text && !self.parts;

        let extraction = akuna_core::extraction::extract_file(
            self.file,
            &akuna_core::extraction::ExtractionConfig {
                return_metadata: full || self.metadata,
                return_content: full || self.text,
                return_part_chunks: false,
                return_parts: full || self.parts,
                text: Some(
                    akuna_core::extraction::TextExtractionConfig::default(),
                ),
                chunking: None,
            },
        )
        .await?;

        print_json(&extraction)
    }
}
