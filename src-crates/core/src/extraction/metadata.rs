use std::path::Path;

use crate::extraction::{ExtractionMetadata, FileExtractionError};

/// Detect file type via Magika and assemble path-derived metadata.
///
/// # Errors
///
/// Returns [`FileExtractionError`] if detection fails.
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
