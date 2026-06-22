use std::fmt::Debug;

use crate::extraction::ExtractionMetadata;

type ExtractionErrorSource = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Errors that can occur during file extraction.
#[derive(Debug, thiserror::Error)]
pub enum FileExtractionError {
    /// I/O error.
    #[error("I/O error: {:?}", source.to_string())]
    Io {
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// Error from an engine determining file type.
    #[error("Detection engine '{engine}' failed")]
    DetectionEngine {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The underlying detection engine error.
        source: ExtractionErrorSource,
    },

    /// Error from an engine extracting file contents.
    #[error("Extraction engine '{engine}' failed")]
    ExtractionEngine {
        /// The name of the engine that failed.
        engine: &'static str,
        /// The underlying extraction engine error.
        source: ExtractionErrorSource,
    },

    /// Extraction succeeded but returned no text content.
    #[error("Extraction engine '{engine}' returned no text content")]
    MissingTextContent {
        /// The name of the engine that returned non-text content.
        engine: &'static str,
    },

    /// File at the given path has no contents.
    #[error("File has no contents")]
    NoContents,

    /// File at the given path has no available extractor for its detected type.
    #[error("No extractor available for this file type:\n{metadata:?}")]
    UnsupportedFileType {
        /// The detected file metadata.
        metadata: Box<ExtractionMetadata>,
    },
}

impl From<std::io::Error> for FileExtractionError {
    fn from(source: std::io::Error) -> Self {
        Self::Io { source }
    }
}
