//! File content and metadata extraction with structured parts.
//!
//! Extraction reads supported files into canonical text plus structured parts.
//! File type detection is ML-backed when feature `detection` is enabled, and
//! extension-based otherwise.
//!
//! Simple helpers cover common use cases: [`extract_file_bytes`],
//! [`extract_file_text`], and [`extract_file_content`]. Use [`extract_file`] for
//! configurable metadata, content, and canonical structured parts. Top-level
//! text segments are derived from structured parts when requested.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::extraction::extract_file_text;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let text = extract_file_text("path/to/file.pdf").await?;
//! println!("{}", text);
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use serde::Serialize;

mod content;
mod errors;
mod metadata;

use content::extract_text as extract_path_text;
use metadata::extract_metadata;

use crate::chunking::{ChunkingConfig, chunk_text};

pub use errors::FileExtractionError;

/// Optional source hints for byte-based extraction.
#[derive(Clone, Debug, Default)]
pub struct SourceHint {
    /// Source file name, when known.
    pub file_name: Option<String>,
    /// Source MIME type, when known.
    pub mime_type: Option<String>,
    /// Source extension without leading dot, when known.
    pub extension: Option<String>,
}

/// Top-level extraction configuration.
///
/// File contents are read when content or parts output is requested.
/// Metadata inference never requires reading file contents.
pub struct ExtractionConfig {
    /// Include inferred file metadata in the result.
    pub return_metadata: bool,
    /// Include extracted content in the result.
    ///
    /// When `false`, content may still be built internally for parts output.
    pub return_content: bool,
    /// Include derived text segments inside each returned part.
    ///
    /// When `false`, part segmenting has no effect.
    pub return_part_segments: bool,
    /// Include structured content parts in the result.
    ///
    /// When `false`, extraction may still build parts internally for other outputs.
    pub return_parts: bool,
    /// Optional text extraction preferences.
    ///
    /// Only applied when `return_content` or `return_parts` is enabled.
    pub text: Option<TextExtractionConfig>,
    /// Optional chunking preferences for derived part segments.
    ///
    /// Only applied when `return_part_segments` is enabled.
    pub chunking: Option<ChunkingConfig>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            return_metadata: true,
            return_content: false,
            return_part_segments: false,
            return_parts: false,
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
    /// Extracted text, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Structured content parts derived from extraction, when content was read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<ExtractionPart>>,
}

/// Extracted text content and related derived data.
#[derive(Debug, Serialize)]
pub struct ExtractionContent {
    /// Extracted text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Text segment derived from an extraction part.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionSegment {
    /// Zero-based segment index within the part.
    pub index: usize,
    /// Segment text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Character range in the part text, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<ExtractionTextRange>,
}

/// Structured content part derived from a source document.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionPart {
    /// Zero-based part index.
    pub index: usize,
    /// Semantic part kind.
    pub kind: String,
    /// Extracted part text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Character range in the canonical extracted text, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<ExtractionTextRange>,
    /// Extraction engine and source location metadata for the part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<ExtractionPartProvenance>,
    /// Derived text segments for this part, when chunking is requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<ExtractionSegment>>,
}

/// Character range within canonical extracted text.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ExtractionTextRange {
    /// Start byte offset, inclusive.
    pub start: usize,
    /// End byte offset, exclusive.
    pub end: usize,
}

/// Extraction provenance and source location metadata for a part.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionPartProvenance {
    /// Extraction method that produced this part.
    pub method: String,
    /// Engines or models involved in producing this part.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub engines: Vec<String>,
    /// Extractor-specific source identifier, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    /// One-based page number, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    /// One-based line number, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// Bounding rectangle, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geometry: Option<ExtractionGeometry>,
    /// Extraction confidence from 0 to 1, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

/// Source bounding rectangle in extractor-native units.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionGeometry {
    /// Left coordinate.
    pub x: f32,
    /// Top coordinate.
    pub y: f32,
    /// Rectangle width.
    pub width: f32,
    /// Rectangle height.
    pub height: f32,
}

/// Internal normalized extraction document.
struct ExtractedDocument {
    canonical_text: Option<String>,
    parts: Vec<ExtractionPart>,
}

impl ExtractedDocument {
    /// Build a plain text document part from extractor text.
    fn from_text(text: String) -> Self {
        let end = text.len();

        Self {
            canonical_text: None,
            parts: vec![ExtractionPart {
                index: 0,
                kind: "text".to_owned(),
                text: Some(text),
                range: Some(ExtractionTextRange { start: 0, end }),
                provenance: Some(ExtractionPartProvenance {
                    method: "text".to_owned(),
                    engines: Vec::new(),
                    source_id: None,
                    page: None,
                    line: None,
                    geometry: None,
                    confidence: None,
                }),
                segments: None,
            }],
        }
    }

    /// Returns canonical text by joining text-bearing parts in order.
    fn text(&self) -> Option<String> {
        if let Some(text) = self.canonical_text.clone() {
            return Some(text);
        }

        let text = self
            .parts
            .iter()
            .filter_map(|part| part.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n");

        (!text.is_empty()).then_some(text)
    }

    /// Convert into public content payload.
    fn into_content(self) -> ExtractionContent {
        let text = self.text();

        ExtractionContent { text }
    }
}

#[cfg(feature = "ocr")]
impl ExtractedDocument {
    /// Build extraction parts from OCR output.
    fn from_ocr_page(page: &crate::ocr::OcrPage) -> Self {
        let parts = page
            .blocks
            .iter()
            .filter_map(|block| {
                let text = block.text.trim();
                (!text.is_empty()).then_some((block, text))
            })
            .scan(0_usize, |start, (block, text)| {
                let end = *start + text.len();
                let range = ExtractionTextRange { start: *start, end };
                *start = end + 2;
                Some((block, text, range))
            })
            .enumerate()
            .map(|(index, (block, text, range))| ExtractionPart {
                index,
                kind: "text".to_owned(),
                text: Some(text.to_owned()),
                range: Some(range),
                provenance: Some(ExtractionPartProvenance {
                    method: "ocr".to_owned(),
                    engines: vec!["ocr".to_owned()],
                    source_id: None,
                    page: None,
                    line: None,
                    geometry: Some(ExtractionGeometry {
                        x: block.bbox.x,
                        y: block.bbox.y,
                        width: block.bbox.width,
                        height: block.bbox.height,
                    }),
                    confidence: block.confidence,
                }),
                segments: None,
            })
            .collect::<Vec<_>>();

        Self {
            canonical_text: None,
            parts,
        }
    }
}

/// Convert an OCR page into extraction content with one text part per OCR block.
#[cfg(feature = "ocr")]
#[must_use]
pub fn content_from_ocr_page(page: &crate::ocr::OcrPage) -> ExtractionContent {
    ExtractedDocument::from_ocr_page(page).into_content()
}

/// Convenience configuration for metadata-only extraction.
pub fn metadata_only_config() -> ExtractionConfig {
    ExtractionConfig {
        return_metadata: true,
        return_content: false,
        return_part_segments: false,
        return_parts: false,
        ..Default::default()
    }
}

/// Convenience configuration for canonical structured parts extraction.
pub fn parts_config() -> ExtractionConfig {
    ExtractionConfig {
        return_parts: true,
        ..Default::default()
    }
}

/// Read file bytes without parsing.
///
/// # Errors
///
/// Returns [`FileExtractionError`] if the file cannot be read or is empty.
pub async fn extract_file_bytes(
    file_path: impl AsRef<Path>,
) -> Result<Vec<u8>, FileExtractionError> {
    let file_path = file_path.as_ref();
    validate_file(file_path)?;
    let bytes = tokio::fs::read(file_path).await?;
    if bytes.is_empty() {
        return Err(FileExtractionError::NoContents);
    }

    Ok(bytes)
}

/// Extract plain text from a file with default extraction settings.
///
/// # Errors
///
/// Returns [`FileExtractionError`] when the file is invalid or unsupported.
pub async fn extract_file_text(
    file_path: impl AsRef<Path>,
) -> Result<String, FileExtractionError> {
    let content = extract_file_content(file_path).await?;
    content.text.ok_or(FileExtractionError::MissingTextContent {
        engine: "extraction",
    })
}

/// Extract structured content from a file with default extraction settings.
///
/// # Errors
///
/// Returns [`FileExtractionError`] when the file is invalid or unsupported.
pub async fn extract_file_content(
    file_path: impl AsRef<Path>,
) -> Result<ExtractionContent, FileExtractionError> {
    let result = extract_file(
        file_path,
        &ExtractionConfig {
            return_content: true,
            return_parts: true,
            ..Default::default()
        },
    )
    .await?;

    let text = result.text.ok_or(FileExtractionError::MissingTextContent {
        engine: "extraction",
    })?;

    Ok(ExtractionContent { text: Some(text) })
}

/// Return provided bytes after validating they are non-empty.
///
/// This mirrors file byte extraction for callers that already loaded input.
///
/// # Errors
///
/// Returns [`FileExtractionError::NoContents`] for empty input.
pub fn extract_bytes(bytes: &[u8]) -> Result<Vec<u8>, FileExtractionError> {
    if bytes.is_empty() {
        return Err(FileExtractionError::NoContents);
    }

    Ok(bytes.to_vec())
}

/// Extract plain text from bytes using MIME or extension hints when supplied.
///
/// # Errors
///
/// Returns [`FileExtractionError`] when bytes are empty, unsupported, or parse fails.
pub async fn extract_text_bytes(
    bytes: &[u8],
    hint: Option<&SourceHint>,
) -> Result<String, FileExtractionError> {
    let content = extract_content_bytes(bytes, hint).await?;
    content.text.ok_or(FileExtractionError::MissingTextContent {
        engine: "omniparse",
    })
}

/// Extract structured content from bytes using MIME or extension hints when supplied.
///
/// # Errors
///
/// Returns [`FileExtractionError`] when bytes are empty, unsupported, or parse fails.
pub async fn extract_content_bytes(
    bytes: &[u8],
    hint: Option<&SourceHint>,
) -> Result<ExtractionContent, FileExtractionError> {
    if bytes.is_empty() {
        return Err(FileExtractionError::NoContents);
    }

    let preferred_mime = hint.and_then(SourceHint::preferred_mime);
    #[cfg(feature = "ocr")]
    if preferred_mime.is_some_and(is_image_mime) {
        return extract_ocr_document_from_bytes(bytes)
            .await
            .map(ExtractedDocument::into_content);
    }

    let parsed = omniparse::extract_from_bytes(bytes, preferred_mime)?;
    let omniparse::Content::Text(text) = parsed.content else {
        return Err(FileExtractionError::MissingTextContent {
            engine: "omniparse",
        });
    };

    Ok(ExtractedDocument::from_text(text).into_content())
}

#[cfg(feature = "ocr")]
async fn extract_ocr_document_from_file(
    file_path: &Path,
) -> Result<ExtractedDocument, FileExtractionError> {
    let ocr = crate::ocr::Ocr::new(crate::ocr::OcrOptions::default())
        .await
        .map_err(ocr_extraction_error)?;
    let page = ocr
        .extract_page_file(file_path)
        .map_err(ocr_extraction_error)?;

    Ok(ExtractedDocument::from_ocr_page(&page))
}

#[cfg(feature = "ocr")]
async fn extract_ocr_document_from_bytes(
    bytes: &[u8],
) -> Result<ExtractedDocument, FileExtractionError> {
    let ocr = crate::ocr::Ocr::new(crate::ocr::OcrOptions::default())
        .await
        .map_err(ocr_extraction_error)?;
    let page = ocr
        .extract_page_bytes(bytes)
        .map_err(ocr_extraction_error)?;

    Ok(ExtractedDocument::from_ocr_page(&page))
}

#[cfg(feature = "ocr")]
fn is_image_mime(mime_type: &str) -> bool {
    matches!(
        mime_type,
        "image/bmp" | "image/jpeg" | "image/png" | "image/tiff"
    )
}

#[cfg(feature = "ocr")]
fn ocr_extraction_error(source: crate::ocr::OcrError) -> FileExtractionError {
    FileExtractionError::ExtractionEngine {
        engine: "ocr",
        source: Box::new(source),
    }
}

impl SourceHint {
    /// Resolve preferred MIME type for byte extraction.
    fn preferred_mime(&self) -> Option<&str> {
        if let Some(mime_type) = self.mime_type.as_deref() {
            return Some(mime_type);
        }

        self.extension
            .as_deref()
            .or_else(|| {
                self.file_name
                    .as_deref()
                    .and_then(|file_name| file_name.rsplit_once('.'))
                    .map(|(_, extension)| extension)
            })
            .and_then(extension_mime)
    }
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

async fn extract_document(
    config: Option<&TextExtractionConfig>,
    file_path: &Path,
    metadata: &ExtractionMetadata,
) -> Result<ExtractedDocument, FileExtractionError> {
    if metadata.mime_type == "application/pdf" {
        return extract_pdf_document(config, file_path);
    }

    if is_office_document_mime(&metadata.mime_type) {
        return extract_office_document(config, file_path);
    }

    match extract_path_text(config, file_path, metadata).await {
        Ok(text) => Ok(extract_text_document(text, metadata)),
        #[cfg(feature = "ocr")]
        Err(FileExtractionError::UnsupportedFileType { .. })
            if is_image_mime(&metadata.mime_type) =>
        {
            extract_ocr_document_from_file(file_path).await
        }
        Err(error) => Err(error),
    }
}

fn extract_text_document(
    text: String,
    metadata: &ExtractionMetadata,
) -> ExtractedDocument {
    if let Some(document) = extract_code_document(&text, metadata) {
        return document;
    }

    ExtractedDocument::from_text(text)
}

fn extract_code_document(
    text: &str,
    metadata: &ExtractionMetadata,
) -> Option<ExtractedDocument> {
    let extension = metadata.extension.as_deref();
    let ranges =
        crate::chunking::tree_sitter::code_part_ranges(text, extension)?;
    let parts = ranges
        .into_iter()
        .enumerate()
        .filter_map(|range| {
            let (index, range) = range;
            let part_text = text.get(range.range.clone())?.trim();
            if part_text.is_empty() {
                return None;
            }

            Some(ExtractionPart {
                index,
                kind: range.kind.clone(),
                text: Some(text.get(range.range.clone())?.to_owned()),
                range: Some(ExtractionTextRange {
                    start: range.range.start,
                    end: range.range.end,
                }),
                provenance: Some(ExtractionPartProvenance {
                    method: "tree_sitter".to_owned(),
                    engines: vec![format!(
                        "tree-sitter-{}",
                        extension.unwrap_or("unknown")
                    )],
                    source_id: None,
                    page: None,
                    line: None,
                    geometry: None,
                    confidence: None,
                }),
                segments: None,
            })
        })
        .collect::<Vec<_>>();

    (parts.len() > 1).then_some(ExtractedDocument {
        canonical_text: Some(text.to_owned()),
        parts,
    })
}

fn is_office_document_mime(mime_type: &str) -> bool {
    matches!(
        mime_type,
        "application/msword"
            | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
}

fn extract_office_document(
    config: Option<&TextExtractionConfig>,
    file_path: &Path,
) -> Result<ExtractedDocument, FileExtractionError> {
    let document = office_oxide::Document::open(file_path)?;
    let text = if config.is_some_and(|config| config.prefer_markdown) {
        document.to_markdown()
    } else {
        document.plain_text()
    };
    let parts = structured_office_parts(&text);

    if parts.len() > 1 {
        return Ok(ExtractedDocument {
            canonical_text: Some(text),
            parts,
        });
    }

    Ok(ExtractedDocument::from_text(text))
}

fn structured_office_parts(text: &str) -> Vec<ExtractionPart> {
    let mut blocks = Vec::new();
    let mut paragraph = Vec::new();

    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if let Some(kind) = structured_office_line_kind(line) {
            push_office_paragraph(&mut blocks, &mut paragraph);
            if !line.starts_with("---") {
                blocks.push((kind, line.to_owned()));
            }
            continue;
        }

        paragraph.push(line.to_owned());
    }
    push_office_paragraph(&mut blocks, &mut paragraph);

    blocks
        .into_iter()
        .scan(0_usize, |cursor, (kind, block)| {
            let range = text[*cursor..].find(&block).map(|offset| {
                let start = *cursor + offset;
                let end = start + block.len();
                *cursor = end;
                ExtractionTextRange { start, end }
            });
            Some((kind, block, range))
        })
        .enumerate()
        .map(|(index, (kind, block, range))| ExtractionPart {
            index,
            kind,
            text: Some(block),
            range,
            provenance: Some(ExtractionPartProvenance {
                method: "structured_office".to_owned(),
                engines: vec!["office_oxide".to_owned()],
                source_id: None,
                page: None,
                line: None,
                geometry: None,
                confidence: None,
            }),
            segments: None,
        })
        .collect()
}

fn push_office_paragraph(
    blocks: &mut Vec<(String, String)>,
    paragraph: &mut Vec<String>,
) {
    if paragraph.is_empty() {
        return;
    }

    blocks.push(("paragraph".to_owned(), paragraph.join("\n")));
    paragraph.clear();
}

fn structured_office_line_kind(line: &str) -> Option<String> {
    if line.starts_with('#') {
        return Some("heading".to_owned());
    }

    if line.starts_with("Source:") {
        return Some("caption".to_owned());
    }

    if line.starts_with("---") {
        return Some("unknown".to_owned());
    }

    None
}

fn extract_pdf_document(
    config: Option<&TextExtractionConfig>,
    file_path: &Path,
) -> Result<ExtractedDocument, FileExtractionError> {
    use pdf_oxide::extractors::{DocumentElement, StructuredExtractor};

    let mut document = pdf_oxide::PdfDocument::open(file_path)?;
    let page_count = document.page_count()?;
    let mut extractor = StructuredExtractor::new();
    let mut parts = Vec::new();

    for page_index in 0..page_count {
        let structured =
            extractor.extract_page(&mut document, page_index as u32)?;
        for element in structured.elements {
            let (kind, text, bbox) = match element {
                DocumentElement::Header { text, bbox, .. } => {
                    ("heading".to_owned(), text, bbox)
                }
                DocumentElement::Paragraph { text, bbox, .. } => {
                    ("paragraph".to_owned(), text, bbox)
                }
                DocumentElement::List { items, bbox, .. } => {
                    let text = items
                        .into_iter()
                        .map(|item| item.text)
                        .collect::<Vec<_>>()
                        .join("\n");
                    ("list_item".to_owned(), text, bbox)
                }
                DocumentElement::Table { cells, bbox, .. } => {
                    let text = cells
                        .into_iter()
                        .map(|row| row.join("\t"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    ("table".to_owned(), text, bbox)
                }
            };

            let text = text.trim();
            if text.is_empty() {
                continue;
            }

            parts.push(ExtractionPart {
                index: parts.len(),
                kind,
                text: Some(text.to_owned()),
                range: None,
                provenance: Some(ExtractionPartProvenance {
                    method: "structured_pdf".to_owned(),
                    engines: vec!["pdf_oxide".to_owned()],
                    source_id: None,
                    page: Some(page_index + 1),
                    line: None,
                    geometry: Some(ExtractionGeometry {
                        x: bbox.0,
                        y: bbox.1,
                        width: bbox.2,
                        height: bbox.3,
                    }),
                    confidence: None,
                }),
                segments: None,
            });
        }
    }

    let text = if config.is_some_and(|config| config.prefer_markdown) {
        document.to_markdown_all(&pdf_oxide::converters::ConversionOptions {
            ..Default::default()
        })?
    } else {
        document.extract_all_text()?
    };

    let parts = roll_up_pdf_parts(parts);
    if parts.len() > 1 {
        let text = parts
            .iter()
            .filter_map(|part| part.text.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n");
        return Ok(ExtractedDocument {
            canonical_text: Some(text),
            parts,
        });
    }

    Ok(ExtractedDocument::from_text(text))
}

fn roll_up_pdf_parts(parts: Vec<ExtractionPart>) -> Vec<ExtractionPart> {
    let lines = parts.into_iter().fold(Vec::new(), roll_up_pdf_line);
    let parts = lines
        .into_iter()
        .fold(Vec::new(), roll_up_pdf_paragraph)
        .into_iter()
        .enumerate()
        .map(|(index, mut part)| {
            part.index = index;
            part
        })
        .collect();

    classify_pdf_flat_parts(split_pdf_heading_parts(parts))
}

fn classify_pdf_flat_parts(
    mut parts: Vec<ExtractionPart>,
) -> Vec<ExtractionPart> {
    for part in &mut parts {
        if is_pdf_page_number(part) {
            part.kind = "footer".to_owned();
        }
    }

    parts
}

fn split_pdf_heading_parts(parts: Vec<ExtractionPart>) -> Vec<ExtractionPart> {
    parts
        .into_iter()
        .flat_map(split_pdf_heading_part)
        .enumerate()
        .map(|(index, mut part)| {
            part.index = index;
            part
        })
        .collect()
}

fn split_pdf_heading_part(part: ExtractionPart) -> Vec<ExtractionPart> {
    let Some(text) = part.text.as_deref() else {
        return vec![part];
    };
    let lines = text.lines().collect::<Vec<_>>();
    let heading_lines = match lines.as_slice() {
        [first, second, ..]
            if first.starts_with("Book ") && second.starts_with("On ") =>
        {
            2
        }
        [first, ..] if first.starts_with("On ") => 1,
        _ => return vec![part],
    };
    if lines.len() <= heading_lines {
        return vec![part];
    }

    let heading_text = lines[..heading_lines].join("\n");
    let body_text = lines[heading_lines..].join("\n");

    let mut heading = part.clone();
    heading.kind = "heading".to_owned();
    heading.text = Some(heading_text);
    heading.range = None;

    let mut body = part;
    body.kind = "paragraph".to_owned();
    body.text = Some(body_text);
    body.range = None;

    vec![heading, body]
}

fn is_pdf_page_number(part: &ExtractionPart) -> bool {
    part.text.as_deref().is_some_and(|text| {
        text.chars().all(|character| character.is_ascii_digit())
    })
}

fn roll_up_pdf_line(
    mut lines: Vec<ExtractionPart>,
    part: ExtractionPart,
) -> Vec<ExtractionPart> {
    let Some(previous) = lines.last_mut() else {
        lines.push(part);
        return lines;
    };

    if same_pdf_line(previous, &part) {
        merge_pdf_parts(previous, part, " ");
        return lines;
    }

    lines.push(part);
    lines
}

fn roll_up_pdf_paragraph(
    mut paragraphs: Vec<ExtractionPart>,
    part: ExtractionPart,
) -> Vec<ExtractionPart> {
    let Some(previous) = paragraphs.last_mut() else {
        paragraphs.push(part);
        return paragraphs;
    };

    if same_pdf_paragraph(previous, &part) {
        merge_pdf_parts(previous, part, "\n");
        return paragraphs;
    }

    paragraphs.push(part);
    paragraphs
}

fn same_pdf_line(left: &ExtractionPart, right: &ExtractionPart) -> bool {
    let (Some(left_source), Some(right_source)) =
        (&left.provenance, &right.provenance)
    else {
        return false;
    };
    let (Some(left_geometry), Some(right_geometry)) =
        (&left_source.geometry, &right_source.geometry)
    else {
        return false;
    };

    left_source.page == right_source.page
        && (left_geometry.y - right_geometry.y).abs() <= 2.0
}

fn same_pdf_paragraph(left: &ExtractionPart, right: &ExtractionPart) -> bool {
    if left.kind != "paragraph" || right.kind != "paragraph" {
        return false;
    }

    let (Some(left_source), Some(right_source)) =
        (&left.provenance, &right.provenance)
    else {
        return false;
    };
    let (Some(left_geometry), Some(right_geometry)) =
        (&left_source.geometry, &right_source.geometry)
    else {
        return false;
    };

    left_source.page == right_source.page
        && (left_geometry.x - right_geometry.x).abs() <= 32.0
        && (left_geometry.y - right_geometry.y).abs() <= 24.0
}

fn merge_pdf_parts(
    target: &mut ExtractionPart,
    source: ExtractionPart,
    separator: &str,
) {
    if let (Some(target_text), Some(source_text)) =
        (target.text.as_mut(), source.text)
    {
        target_text.push_str(separator);
        target_text.push_str(&source_text);
    }

    if let (Some(target_range), Some(source_range)) =
        (target.range.as_mut(), source.range)
    {
        target_range.end = source_range.end;
    }

    if let (Some(target_source), Some(source_source)) =
        (target.provenance.as_mut(), source.provenance)
        && let (Some(target_geometry), Some(source_geometry)) =
            (target_source.geometry.as_mut(), source_source.geometry)
    {
        let x = target_geometry.x.min(source_geometry.x);
        let y = target_geometry.y.min(source_geometry.y);
        let right = (target_geometry.x + target_geometry.width)
            .max(source_geometry.x + source_geometry.width);
        let bottom = (target_geometry.y + target_geometry.height)
            .max(source_geometry.y + source_geometry.height);
        target_geometry.x = x;
        target_geometry.y = y;
        target_geometry.width = right - x;
        target_geometry.height = bottom - y;
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
        || config.return_part_segments
        || config.return_parts;
    let need_content = config.return_content
        || config.return_part_segments
        || config.return_parts;
    let need_part_segments = config.return_part_segments;

    let metadata = if need_metadata {
        Some(extract_metadata(file_path)?)
    } else {
        None
    };

    let document = if let (Some(metadata), true) =
        (metadata.as_ref(), need_content)
    {
        Some(extract_document(config.text.as_ref(), file_path, metadata).await?)
    } else {
        None
    };

    let text = document.as_ref().and_then(ExtractedDocument::text);
    let parts = document.as_ref().and_then(|document| {
        config.return_parts.then(|| {
            let parts = document.parts.clone();
            if !need_part_segments {
                return parts;
            }

            derive_part_chunking(
                parts,
                config.chunking.as_ref(),
                metadata.as_ref(),
            )
        })
    });
    let returned_text = config.return_content.then_some(text).flatten();
    let returned_metadata = if config.return_metadata {
        metadata
    } else {
        None
    };

    Ok(ExtractionResult {
        metadata: returned_metadata,
        text: returned_text,
        parts,
    })
}

/// Add chunking-derived segments to text-bearing parts.
fn derive_part_chunking(
    parts: Vec<ExtractionPart>,
    config: Option<&ChunkingConfig>,
    metadata: Option<&ExtractionMetadata>,
) -> Vec<ExtractionPart> {
    parts
        .into_iter()
        .map(|mut part| {
            let Some(text) = part.text.as_deref() else {
                return part;
            };

            let chunks = chunk_text(
                config,
                text,
                metadata.and_then(|metadata| metadata.extension.as_deref()),
            );
            if chunks.is_empty() {
                return part;
            }

            let mut cursor = 0;
            let segments = chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| {
                    let start = text[cursor..]
                        .find(chunk)
                        .map_or(cursor, |offset| cursor + offset);
                    let end = start + chunk.len();
                    cursor = end;

                    ExtractionSegment {
                        index,
                        text: Some((*chunk).to_owned()),
                        range: Some(ExtractionTextRange { start, end }),
                    }
                })
                .collect::<Vec<_>>();
            part.segments = Some(segments);
            part
        })
        .collect()
}

/// Map common extension hints to MIME types.
fn extension_mime(extension: &str) -> Option<&'static str> {
    match extension
        .trim_start_matches('.')
        .to_ascii_lowercase()
        .as_str()
    {
        "c" | "cpp" | "cs" | "go" | "java" | "php" | "py" | "rb" | "rs"
        | "sh" | "sql" | "toml" | "ts" | "yaml" | "yml" => Some("text/plain"),
        "css" => Some("text/css"),
        "doc" => Some("application/msword"),
        "docx" => Some(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        "epub" => Some("application/epub+zip"),
        "html" | "htm" => Some("text/html"),
        "js" => Some("text/javascript"),
        "json" => Some("application/json"),
        "md" | "markdown" => Some("text/markdown"),
        "pdf" => Some("application/pdf"),
        "pptx" => Some(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        "rtf" => Some("application/rtf"),
        "txt" | "text" => Some("text/plain"),
        "xhtml" => Some("application/xhtml+xml"),
        "xml" | "rss" => Some("application/xml"),
        _ => None,
    }
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

    async fn assert_extracts_text(
        file_name: &str,
        expected_text: &str,
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
        let Some(text) = extraction.text else {
            panic!("File {} did not return text", file_name);
        };
        let preview = text.chars().take(100).collect::<String>();

        assert!(
            text.contains(expected_text),
            "File {} does not contain expected text: {}",
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
                assert_extracts_text($file_name, $expected_content).await
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
    async fn extracts_text_with_metadata() -> Result<(), FileExtractionError> {
        let file_path = get_extraction_fixture("text.txt");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_content: true,
                ..Default::default()
            },
        )
        .await?;
        let Some(metadata) = extraction.metadata else {
            panic!("File text.txt did not return metadata");
        };

        assert_eq!(metadata.extension.as_deref(), Some("txt"));
        assert_eq!(metadata.stem.as_deref(), Some("text"));
        assert!(extraction.text.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn extracts_metadata_without_text() -> Result<(), FileExtractionError>
    {
        let file_path = get_extraction_fixture("text.pptx");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_metadata: true,
                return_content: false,
                ..Default::default()
            },
        )
        .await?;
        let Some(metadata) = extraction.metadata else {
            panic!("File text.pptx did not return metadata");
        };

        assert_eq!(metadata.extension.as_deref(), Some("pptx"));
        assert!(extraction.text.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn returns_part_level_segments_when_requested()
    -> Result<(), FileExtractionError> {
        let file_path = get_extraction_fixture("text.txt");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_content: true,
                return_part_segments: true,
                return_parts: true,
                ..Default::default()
            },
        )
        .await?;
        assert!(extraction.text.is_some());
        let parts = extraction.parts.expect("parts should be returned");
        let text_part = parts
            .iter()
            .find(|part| part.text.is_some())
            .expect("text part should exist");

        assert!(text_part.segments.as_ref().is_some_and(|segments| {
            !segments.is_empty()
                && segments.iter().all(|segment| segment.range.is_some())
        }));
        Ok(())
    }

    #[test]
    fn validates_simple_byte_extraction() -> Result<(), FileExtractionError> {
        assert_eq!(extract_bytes(b"hello")?, b"hello");
        assert!(matches!(
            extract_bytes(b""),
            Err(FileExtractionError::NoContents)
        ));

        Ok(())
    }

    #[test]
    fn builds_parts_config() {
        let config = parts_config();

        assert!(config.return_metadata);
        assert!(!config.return_content);
        assert!(!config.return_part_segments);
        assert!(config.return_parts);
    }

    #[tokio::test]
    async fn extracts_file_text_with_simple_api()
    -> Result<(), FileExtractionError> {
        let file_path = get_extraction_fixture("text.txt");
        let text = extract_file_text(file_path).await?;

        assert!(text.contains(SAMPLE_TEXT));

        Ok(())
    }

    #[tokio::test]
    async fn extracts_byte_content_with_text() -> Result<(), FileExtractionError>
    {
        let content = extract_content_bytes(
            b"hello bytes",
            Some(&SourceHint {
                extension: Some("txt".to_owned()),
                ..Default::default()
            }),
        )
        .await?;

        assert_eq!(content.text.as_deref(), Some("hello bytes"));

        Ok(())
    }

    #[tokio::test]
    async fn returns_top_level_parts_without_text()
    -> Result<(), FileExtractionError> {
        let file_path = get_extraction_fixture("text.txt");
        let extraction = extract_file(file_path, &parts_config()).await?;
        let parts = extraction.parts.expect("file should return parts");

        assert!(extraction.text.is_none());
        assert!(!parts.is_empty());
        assert!(parts.iter().any(|part| {
            part.text
                .as_deref()
                .is_some_and(|text| text.contains(SAMPLE_TEXT))
        }));

        Ok(())
    }
}
