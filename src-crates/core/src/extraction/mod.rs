//! File content and metadata extraction with structured parts.
//!
//! Extraction reads supported files into structured parts and derived text.
//! File type detection is ML-backed when feature `detection` is enabled, and
//! extension-based otherwise.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::extraction::{extract_file, ExtractionConfig};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let result = extract_file("path/to/file.pdf", &ExtractionConfig::default()).await?;
//! # Ok(())
//! # }
//! ```

use std::path::Path;

use serde::Serialize;

mod code_parts;
mod content;
mod errors;
mod metadata;

use content::extract_text as extract_path_text;
use metadata::extract_metadata;

pub use errors::FileExtractionError;

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
    /// Include structured content parts in the result.
    ///
    /// When `false`, extraction may still build parts internally for other outputs.
    pub return_parts: bool,
    /// Include part source provenance in the result.
    pub return_provenance: bool,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            return_metadata: true,
            return_content: false,
            return_parts: false,
            return_provenance: false,
        }
    }
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
    /// Source location and extractor details for this part, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<ExtractionProvenance>,
}

/// Source details for an extracted part.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionProvenance {
    /// Extractor or source layer that produced this part.
    pub source: String,
    /// One-based page number, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page: Option<usize>,
    /// Bounding box in source coordinate space, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<ExtractionBbox>,
    /// Byte range in the source text, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_range: Option<ExtractionByteRange>,
    /// Recognition confidence from 0 to 1, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Source-local classification, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

/// Bounding box in source coordinates.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionBbox {
    /// Left coordinate.
    pub x: f32,
    /// Top coordinate.
    pub y: f32,
    /// Box width.
    pub width: f32,
    /// Box height.
    pub height: f32,
}

/// Byte range in source content.
#[derive(Clone, Debug, Serialize)]
pub struct ExtractionByteRange {
    /// Inclusive start byte offset.
    pub start: usize,
    /// Exclusive end byte offset.
    pub end: usize,
}

/// Internal normalized extraction document.
struct ExtractedDocument {
    canonical_text: Option<String>,
    parts: Vec<ExtractionPart>,
}

impl ExtractedDocument {
    /// Build a plain text document part from extractor text.
    fn from_text(text: String) -> Self {
        let parts = structured_text_parts(&text);
        if parts.len() > 1 {
            return Self {
                canonical_text: Some(text),
                parts,
            };
        }

        Self {
            canonical_text: None,
            parts: vec![ExtractionPart {
                index: 0,
                kind: "text".to_owned(),
                text: Some(text),
                provenance: None,
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
}

fn structured_text_parts(text: &str) -> Vec<ExtractionPart> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
        .map(|(index, line)| ExtractionPart {
            index,
            kind: text_line_kind(line).to_owned(),
            text: Some(line.to_owned()),
            provenance: None,
        })
        .collect()
}

fn text_line_kind(line: &str) -> &'static str {
    if line.starts_with("Source:") {
        return "caption";
    }

    if line.starts_with('<') {
        return markup_line_kind(line);
    }

    if line.starts_with('#') || line.starts_with("Book ") {
        return "heading";
    }

    "paragraph"
}

fn markup_line_kind(line: &str) -> &'static str {
    if line.starts_with("<h") || line.starts_with("<title") {
        return "heading";
    }

    if line.starts_with("<p") {
        return "paragraph";
    }

    if line.starts_with("<pre") || line.starts_with("<code") {
        return "code";
    }

    "markup"
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
            .enumerate()
            .map(|(index, (block, text))| ExtractionPart {
                index,
                kind: text_line_kind(text).to_owned(),
                text: Some(text.to_owned()),
                provenance: Some(ExtractionProvenance {
                    source: "ocr".to_owned(),
                    page: None,
                    bbox: Some(ExtractionBbox {
                        x: block.bbox.x,
                        y: block.bbox.y,
                        width: block.bbox.width,
                        height: block.bbox.height,
                    }),
                    byte_range: None,
                    confidence: block.confidence,
                    kind: Some(format!("{:?}", block.kind)),
                }),
            })
            .collect::<Vec<_>>();

        Self {
            canonical_text: None,
            parts,
        }
    }
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
    file_path: &Path,
    metadata: &ExtractionMetadata,
) -> Result<ExtractedDocument, FileExtractionError> {
    if metadata.mime_type == "application/pdf" {
        return extract_pdf_document(file_path);
    }

    if is_office_document_mime(&metadata.mime_type) {
        return extract_office_document(file_path);
    }

    match extract_path_text(file_path, metadata).await {
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
        crate::extraction::code_parts::code_part_ranges(text, extension)?;
    let parts = ranges
        .into_iter()
        .enumerate()
        .filter_map(|(index, range)| {
            let part_text = text.get(range.range.clone())?.trim();
            if part_text.is_empty() {
                return None;
            }

            Some(ExtractionPart {
                index,
                kind: range.kind.clone(),
                text: Some(text.get(range.range.clone())?.to_owned()),
                provenance: Some(ExtractionProvenance {
                    source: "text".to_owned(),
                    page: None,
                    bbox: None,
                    byte_range: Some(ExtractionByteRange {
                        start: range.range.start,
                        end: range.range.end,
                    }),
                    confidence: None,
                    kind: Some(range.kind),
                }),
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
    file_path: &Path,
) -> Result<ExtractedDocument, FileExtractionError> {
    let document = office_oxide::Document::open(file_path)?;
    let text = document.plain_text();
    let parts = structured_office_parts(&text);

    if parts.len() > 1 {
        return Ok(ExtractedDocument {
            canonical_text: None,
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
        .enumerate()
        .map(|(index, (kind, block))| ExtractionPart {
            index,
            kind,
            text: Some(block),
            provenance: None,
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

#[derive(Clone)]
struct PdfPart {
    kind: String,
    text: String,
    page: usize,
    bbox: (f32, f32, f32, f32),
}

impl PdfPart {
    fn into_extraction_part(self, index: usize) -> ExtractionPart {
        ExtractionPart {
            index,
            kind: self.kind,
            text: Some(self.text),
            provenance: Some(ExtractionProvenance {
                source: "pdf".to_owned(),
                page: Some(self.page),
                bbox: Some(ExtractionBbox {
                    x: self.bbox.0,
                    y: self.bbox.1,
                    width: self.bbox.2,
                    height: self.bbox.3,
                }),
                byte_range: None,
                confidence: None,
                kind: None,
            }),
        }
    }
}

fn extract_pdf_document(
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

            parts.push(PdfPart {
                kind,
                text: text.to_owned(),
                page: page_index + 1,
                bbox,
            });
        }
    }

    let text = document.extract_all_text()?;

    let parts = roll_up_pdf_parts(parts)
        .into_iter()
        .enumerate()
        .map(|(index, part)| part.into_extraction_part(index))
        .collect::<Vec<_>>();

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

fn roll_up_pdf_parts(parts: Vec<PdfPart>) -> Vec<PdfPart> {
    let lines = parts.into_iter().fold(Vec::new(), roll_up_pdf_line);
    let parts = lines.into_iter().fold(Vec::new(), roll_up_pdf_paragraph);

    classify_pdf_flat_parts(split_pdf_heading_parts(parts))
}

fn classify_pdf_flat_parts(mut parts: Vec<PdfPart>) -> Vec<PdfPart> {
    for part in &mut parts {
        if part
            .text
            .chars()
            .all(|character| character.is_ascii_digit())
        {
            part.kind = "footer".to_owned();
        }
    }

    parts
}

fn split_pdf_heading_parts(parts: Vec<PdfPart>) -> Vec<PdfPart> {
    parts.into_iter().flat_map(split_pdf_heading_part).collect()
}

fn split_pdf_heading_part(part: PdfPart) -> Vec<PdfPart> {
    let lines = part.text.lines().collect::<Vec<_>>();
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
    heading.text = heading_text;

    let mut body = part;
    body.kind = "paragraph".to_owned();
    body.text = body_text;

    vec![heading, body]
}

fn roll_up_pdf_line(mut lines: Vec<PdfPart>, part: PdfPart) -> Vec<PdfPart> {
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
    mut paragraphs: Vec<PdfPart>,
    part: PdfPart,
) -> Vec<PdfPart> {
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

fn same_pdf_line(left: &PdfPart, right: &PdfPart) -> bool {
    left.page == right.page && (left.bbox.1 - right.bbox.1).abs() <= 2.0
}

fn same_pdf_paragraph(left: &PdfPart, right: &PdfPart) -> bool {
    left.kind == "paragraph"
        && right.kind == "paragraph"
        && left.page == right.page
        && (left.bbox.0 - right.bbox.0).abs() <= 32.0
        && (left.bbox.1 - right.bbox.1).abs() <= 24.0
}

fn merge_pdf_parts(target: &mut PdfPart, source: PdfPart, separator: &str) {
    target.text.push_str(separator);
    target.text.push_str(&source.text);

    let x = target.bbox.0.min(source.bbox.0);
    let y = target.bbox.1.min(source.bbox.1);
    let right =
        (target.bbox.0 + target.bbox.2).max(source.bbox.0 + source.bbox.2);
    let bottom =
        (target.bbox.1 + target.bbox.3).max(source.bbox.1 + source.bbox.3);
    target.bbox = (x, y, right - x, bottom - y);
}

/// Configurable main entry point to extract content and metadata from a file on disk.
///
/// Detection and content extraction run only as required by `config`.
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

    let need_metadata =
        config.return_metadata || config.return_content || config.return_parts;
    let need_content = config.return_content || config.return_parts;

    let metadata = if need_metadata {
        Some(extract_metadata(file_path)?)
    } else {
        None
    };

    let document =
        if let (Some(metadata), true) = (metadata.as_ref(), need_content) {
            Some(extract_document(file_path, metadata).await?)
        } else {
            None
        };

    let text = document.as_ref().and_then(ExtractedDocument::text);
    let parts = document.as_ref().and_then(|document| {
        config.return_parts.then(|| {
            let mut parts = document.parts.clone();
            if !config.return_provenance {
                for part in &mut parts {
                    part.provenance = None;
                }
            }
            parts
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
    async fn returns_top_level_parts_without_text()
    -> Result<(), FileExtractionError> {
        let file_path = get_extraction_fixture("text.txt");
        let extraction = extract_file(
            file_path,
            &ExtractionConfig {
                return_parts: true,
                ..Default::default()
            },
        )
        .await?;
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

    #[tokio::test]
    async fn returns_parts_for_syntax_text_fixtures()
    -> Result<(), FileExtractionError> {
        for file_name in [
            "text.epub",
            "code.html",
            "text.rss",
            "text.xhtml",
            "text.xml",
        ] {
            let file_path = get_extraction_fixture(file_name);
            let extraction = extract_file(
                file_path,
                &ExtractionConfig {
                    return_parts: true,
                    ..Default::default()
                },
            )
            .await?;
            let parts = extraction.parts.expect("file should return parts");

            assert!(
                parts.len() > 1,
                "{file_name} should produce structured parts"
            );
        }

        Ok(())
    }

    #[cfg(feature = "ocr")]
    #[test]
    fn extracts_png_with_ocr() {
        let handle = std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(|| {
                let runtime = tokio::runtime::Runtime::new()
                    .expect("tokio runtime should start");
                runtime.block_on(async {
                    let file_path = get_extraction_fixture("text-hidpi.png");
                    let extraction = extract_file(
                        file_path,
                        &ExtractionConfig {
                            return_content: true,
                            return_parts: true,
                            ..Default::default()
                        },
                    )
                    .await
                    .expect("OCR extraction should succeed");

                    assert!(extraction.text.is_some_and(|text| {
                        text.contains("On Looking Inward")
                    }));
                    let parts =
                        extraction.parts.expect("OCR should return parts");
                    assert!(parts.len() > 1);
                    assert!(parts.iter().any(|part| part.kind == "heading"));
                    assert!(parts.iter().any(|part| part.kind == "caption"));
                });
            })
            .expect("OCR test thread should start");

        handle.join().expect("OCR test thread should finish");
    }
}
