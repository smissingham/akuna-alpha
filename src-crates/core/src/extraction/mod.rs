//! File content and metadata extraction with type-driven dispatch.
//!
//! File type is detected via [`crate::detection`] (Magika), then routed to a
//! format-specific extractor: PDF, office documents, EPUB, or a generic text and
//! markup fallback through omniparse.
//!
//! The main entry point is [`extract_file`], which returns an [`ExtractionResult`]
//! populated with metadata, content, and chunks based on the [`ExtractionConfig`] flags.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::extraction::{extract_file, ExtractionConfig};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let config = ExtractionConfig {
//!     return_metadata: true,
//!     return_content: true,
//!     ..ExtractionConfig::default()
//! };
//! let result = extract_file("path/to/file.pdf", &config).await?;
//! if let Some(content) = result.content.and_then(|c| c.text) {
//!     println!("{}", content);
//! }
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use serde::Serialize;

mod content;
mod errors;
mod metadata;

use content::extract_text;
use metadata::extract_metadata;

use crate::chunking::{ChunkingConfig, chunk_text};

pub use errors::FileExtractionError;

/// Top-level extraction configuration.
///
/// File contents are only read when `return_content` or `return_chunking` is set.
/// Metadata inference never requires reading file contents.
pub struct ExtractionConfig {
    /// Include inferred file metadata in the result.
    pub return_metadata: bool,
    /// Include extracted content in the result.
    ///
    /// When `false` with `return_chunking` enabled, content is still extracted but not returned.
    pub return_content: bool,
    /// Include extracted text chunks in the result.
    ///
    /// When `false`, `chunking` has no effect.
    pub return_chunking: bool,
    /// Optional text extraction preferences.
    ///
    /// Only applied when `return_content` or `return_chunking` is enabled.
    pub text: Option<TextExtractionConfig>,
    /// Optional chunking preferences.
    ///
    /// Only applied when `return_chunking` is enabled.
    pub chunking: Option<ChunkingConfig>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            return_metadata: true,
            return_content: false,
            return_chunking: false,
            text: Some(TextExtractionConfig::default()),
            chunking: Some(ChunkingConfig::default()),
        }
    }
}

/// Preferences for how text is extracted.
#[derive(Default)]
pub struct TextExtractionConfig {
    /// Whether supported extractors should prefer Markdown output.
    pub prefer_markdown: bool,
}

/// Structured extraction output.
#[derive(Debug, Serialize)]
pub struct ExtractionResult {
    /// Detected file metadata, when requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ExtractionMetadata>,
    /// Extracted text content, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ExtractionContent>,
    /// Extracted text chunks, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<Vec<ExtractionChunk>>,
}

/// Extracted text content and related derived data.
#[derive(Debug, Serialize)]
pub struct ExtractionContent {
    /// Extracted text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Extracted text chunk and related derived data.
#[derive(Debug, Serialize)]
pub struct ExtractionChunk {
    /// Zero-based chunk index.
    pub index: usize,
    /// Chunk text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Metadata detected during extraction.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionMetadata {
    /// File stem from the path, without the extension, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stem: Option<String>,
    /// File extension from the path, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extension: Option<String>,
    /// Unique label identifying the detected content type.
    pub label: String,
    /// Detected MIME type.
    pub mime_type: String,
    /// Detected file type description.
    pub description: String,
    /// Whether the file can be treated as text.
    pub is_text: bool,
}

impl std::fmt::Display for ExtractionMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let json =
            serde_json::to_string_pretty(self).map_err(|_| std::fmt::Error)?;
        write!(f, "{json}")
    }
}

/// Configurable main entry point to extract content and metadata from a file on disk.
///
/// Detection, content extraction, and chunking run only as required by `config`.
///
/// # Errors
///
/// Returns [`FileExtractionError`] if the path is invalid, detection fails,
/// or no extractor is available for the detected file type.
pub async fn extract_file(
    file_path: impl AsRef<Path>,
    config: &ExtractionConfig,
) -> Result<ExtractionResult, FileExtractionError> {
    let file_path = file_path.as_ref();
    validate_file(file_path)?;

    let need_metadata = config.return_metadata
        || config.return_content
        || config.return_chunking;
    let need_content = config.return_content || config.return_chunking;
    let need_chunks = config.return_chunking;

    let metadata = if need_metadata {
        Some(extract_metadata(file_path)?)
    } else {
        None
    };

    let content =
        if let (Some(metadata), true) = (metadata.as_ref(), need_content) {
            Some(extract_text(config.text.as_ref(), file_path, metadata).await?)
        } else {
            None
        };

    let chunk_texts = if let (Some(metadata), Some(content), true) =
        (metadata.as_ref(), content.as_ref(), need_chunks)
    {
        Some(chunk_text(
            config.chunking.as_ref(),
            content,
            metadata.extension.as_deref(),
        ))
    } else {
        None
    };

    let returned_metadata = if config.return_metadata {
        metadata
    } else {
        None
    };
    let chunks = chunk_texts.and_then(|chunk_texts| {
        config.return_chunking.then(|| {
            chunk_texts
                .into_iter()
                .enumerate()
                .map(|(index, text)| ExtractionChunk {
                    index,
                    text: Some(text.to_owned()),
                })
                .collect()
        })
    });
    let content = config
        .return_content
        .then_some(ExtractionContent { text: content });

    Ok(ExtractionResult {
        metadata: returned_metadata,
        content,
        chunks,
    })
}

/// Confirm the path exists, is a readable file, and has non-zero size.
///
/// # Errors
///
/// Returns [`FileExtractionError`] if any check fails.
fn validate_file(file_path: &Path) -> Result<(), FileExtractionError> {
    // confirm file exists
    if !file_path.exists() {
        return Err(FileExtractionError::Io {
            source: std::io::Error::other("Given path does not exist"),
        });
    }

    // confirm file is readable
    if !file_path.is_file() {
        return Err(FileExtractionError::Io {
            source: std::io::Error::other("Given path is not a file"),
        });
    }

    // confirm file has bytes
    if file_path.metadata()?.len() == 0 {
        return Err(FileExtractionError::NoContents);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_extraction_fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../test-corpus/content/fixtures")
            .join(name)
    }

    async fn assert_extracts_content(
        file_name: &str,
        expected_content: &str,
    ) -> Result<(), FileExtractionError> {
        let file_path = get_extraction_fixture(file_name);

        let extraction = extract_file(
            &file_path,
            &ExtractionConfig {
                return_content: true,
                ..Default::default()
            },
        )
        .await?;
        let Some(content) = extraction.content else {
            panic!("File {} did not return content", file_name);
        };
        let Some(text) = content.text else {
            panic!("File {} did not return content text", file_name);
        };
        let preview = text.chars().take(100).collect::<String>();

        assert!(
            text.contains(expected_content),
            "File {} does not contain expected text in content: {}",
            file_name,
            preview
        );

        Ok(())
    }

    macro_rules! extract_from_files {
        ($expected_content:expr; $($test_name:ident => $file_name:expr),+ $(,)?) => {
            $(
            #[tokio::test]
            async fn $test_name() -> Result<(), FileExtractionError> {
                assert_extracts_content($file_name, $expected_content).await
            }
            )+
        };
    }

    macro_rules! unsupported_format_test {
        ($test_name:ident, $file_name:expr) => {
            #[tokio::test]
            async fn $test_name() -> Result<(), FileExtractionError> {
                let file_path = get_extraction_fixture($file_name);
                let error = match extract_file(
                    &file_path,
                    &ExtractionConfig {
                        return_content: true,
                        ..Default::default()
                    },
                )
                .await
                {
                    Ok(_) => {
                        panic!("File {} should be unsupported", $file_name)
                    }
                    Err(error) => error,
                };

                assert!(
                    matches!(
                        error,
                        FileExtractionError::UnsupportedFileType { .. }
                    ),
                    "File {} returned unexpected error: {}",
                    $file_name,
                    error
                );

                Ok(())
            }
        };
    }

    const SAMPLE_TEXT: &str = "life is but an instant; his substance";
    const SAMPLE_CODE: &str = "extraction fixture marker: shared code sample";

    // Supported document formats
    extract_from_files!(SAMPLE_TEXT;
        supported_doc => "text.doc",
        supported_docx => "text.docx",
        supported_epub => "text.epub",
        supported_md => "text.md",
        supported_pdf => "text.pdf",
        supported_pptx => "text.pptx",
        supported_rss => "text.rss",
        supported_rtf => "text.rtf",
        supported_txt => "text.txt",
        supported_xhtml => "text.xhtml",
        supported_xml => "text.xml",
    );

    // Supported code formats
    extract_from_files!(SAMPLE_CODE;
        supported_c => "code.c",
        supported_cpp => "code.cpp",
        supported_cs => "code.cs",
        supported_css => "code.css",
        supported_go => "code.go",
        supported_html => "code.html",
        supported_java => "code.java",
        supported_js => "code.js",
        supported_php => "code.php",
        supported_py => "code.py",
        supported_rb => "code.rb",
        supported_rs => "code.rs",
        supported_sh => "code.sh",
        supported_sql => "code.sql",
        supported_toml => "code.toml",
        supported_ts => "code.ts",
        supported_yaml => "code.yaml",
    );

    // Currently known faulty formats
    unsupported_format_test!(unsupported_webp, "image-excel-table.webp");
    unsupported_format_test!(unsupported_odt, "text.odt");
    unsupported_format_test!(unsupported_zip, "text.txt.zip");

    #[tokio::test]
    async fn extracts_content_with_metadata() -> Result<(), FileExtractionError>
    {
        let file_path = get_extraction_fixture("text.txt");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_content: true,
                return_chunking: true,
                ..Default::default()
            },
        )
        .await?;
        let Some(metadata) = extraction.metadata else {
            panic!("File text.txt did not return metadata");
        };

        assert_eq!(metadata.extension.as_deref(), Some("txt"));
        assert_eq!(metadata.stem.as_deref(), Some("text"));
        assert!(extraction.content.is_some());
        assert!(extraction.chunks.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn extracts_metadata_without_content()
    -> Result<(), FileExtractionError> {
        let file_path = get_extraction_fixture("text.pptx");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_metadata: true,
                return_content: false,
                return_chunking: false,
                ..Default::default()
            },
        )
        .await?;
        let Some(metadata) = extraction.metadata else {
            panic!("File text.pptx did not return metadata");
        };

        assert_eq!(metadata.extension.as_deref(), Some("pptx"));
        assert!(extraction.content.is_none());
        assert!(extraction.chunks.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn extracts_chunks_without_content() -> Result<(), FileExtractionError>
    {
        let file_path = get_extraction_fixture("text.txt");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_content: false,
                return_chunking: true,
                ..Default::default()
            },
        )
        .await?;

        assert!(extraction.content.is_none());
        assert!(extraction.chunks.is_some_and(|chunks| !chunks.is_empty()));

        Ok(())
    }
}
