use std::path::Path;

use crate::{ExtractionMetadata, FileExtractionError};

/// Uses and returns Magika filetype detection, alongside basic useful file meta.
pub(super) async fn extract_metadata(
    file_path: &Path,
) -> Result<ExtractionMetadata, FileExtractionError> {
    let detected = detect_file_type(file_path).await?;
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

/// Uses Google "Magika" ML model to intelligently infer file type from given path.
async fn detect_file_type(
    file_path: &Path,
) -> Result<DetectionResult, FileExtractionError> {
    let mut magika = akuna_core_detection::Session::new_default()?;
    let type_info = magika.identify_file_async(file_path).await?.info();

    Ok(DetectionResult {
        mime_type: type_info.mime_type.to_string(),
        label: type_info.label.to_string(),
        description: type_info.description.to_string(),
        is_text: type_info.is_text,
    })
}
