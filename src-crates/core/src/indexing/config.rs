use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// File indexing configuration section.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct IndexingConfig {
    /// Configured file locations to index.
    #[serde(default)]
    pub locations: Vec<IndexLocation>,
}

/// Configured filesystem location for indexing.
#[derive(Debug, Clone, Deserialize, Eq, PartialEq, Serialize)]
pub struct IndexLocation {
    /// Human-readable unique-ish display name.
    #[serde(deserialize_with = "deserialize_name")]
    pub name: String,
    /// Human-readable purpose for this indexed location.
    pub description: Option<String>,
    /// Canonical filesystem path to scan.
    pub path: PathBuf,
}

fn deserialize_name<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let name = String::deserialize(deserializer)?;
    if is_valid_name(&name) {
        return Ok(name);
    }

    Err(serde::de::Error::custom(
        "name must only contain ASCII letters, numbers, '-' or '_'",
    ))
}

/// Returns true when name uses supported simple characters.
pub fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().all(|char| {
            char.is_ascii_alphanumeric() || char == '-' || char == '_'
        })
}
