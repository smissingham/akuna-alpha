use std::path::Path;

use crate::extraction::{ExtractionMetadata, FileExtractionError};

/// Detect or infer file type and assemble path-derived metadata.
///
/// # Errors
///
/// Returns [`FileExtractionError`] if enabled detection fails.
pub(super) fn extract_metadata(
    file_path: &Path,
) -> Result<ExtractionMetadata, FileExtractionError> {
    let detected = detect_file_type(file_path)?;
    let extension = file_path
        .extension()
        .map(|extension| extension.to_string_lossy().into_owned());
    let stem = file_path
        .file_stem()
        .map(|file_stem| file_stem.to_string_lossy().into_owned());

    Ok(ExtractionMetadata {
        stem,
        extension,
        label: detected.label,
        mime_type: detected.mime_type,
        description: detected.description,
        is_text: detected.is_text,
    })
}

/// Detection result used internally before assembling full metadata.
struct DetectionResult {
    mime_type: String,
    label: String,
    description: String,
    is_text: bool,
}

/// Detect file type from path using the Magika ML model.
///
/// Builds a fresh [`Session`](crate::detection::Session) per call. Callers
/// performing many detections should construct a `Session` once and call
/// [`Session::identify_file_sync`](crate::detection::Session::identify_file_sync)
/// directly.
///
/// # Errors
///
/// Returns [`FileExtractionError`] if the Magika session fails to load or infer.
#[cfg(feature = "detection")]
fn detect_file_type(
    file_path: &Path,
) -> Result<DetectionResult, FileExtractionError> {
    let mut magika = crate::detection::Session::new_default()?;
    let type_info = magika.identify_file_sync(file_path)?.info();

    Ok(DetectionResult {
        mime_type: type_info.mime_type.to_string(),
        label: type_info.label.to_string(),
        description: type_info.description.to_string(),
        is_text: type_info.is_text,
    })
}

/// Infer file type from path when ML detection is disabled.
#[cfg(not(feature = "detection"))]
fn detect_file_type(
    file_path: &Path,
) -> Result<DetectionResult, FileExtractionError> {
    let extension = file_path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default();
    let mime_type =
        extension_mime(extension).unwrap_or("application/octet-stream");

    Ok(DetectionResult {
        mime_type: mime_type.to_string(),
        label: extension.to_string(),
        description: format!("{} file", extension.to_ascii_uppercase()),
        is_text: mime_type.starts_with("text/")
            || matches!(
                mime_type,
                "application/json"
                    | "application/rtf"
                    | "application/xhtml+xml"
                    | "application/xml"
            ),
    })
}

/// Map common extension hints to MIME types.
#[cfg(not(feature = "detection"))]
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
