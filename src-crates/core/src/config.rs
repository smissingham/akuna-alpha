//! Application configuration loading and persistence.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use serde::de::IntoDeserializer;
use serde::{Deserialize, Serialize};

use crate::{
    APP_NAME,
    dirs::{AppDirType, get_app_dir},
    indexing::IndexingConfig,
};

const CONFIG_FILE_EXTENSION: &str = "yaml";

/// Root app configuration persisted as YAML.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AppConfig {
    /// File indexing configuration.
    #[serde(default)]
    pub indexing: IndexingConfig,
}

impl AppConfig {
    /// Loads config from default config directory.
    pub fn load() -> Result<Self> {
        Self::load_from_dir(&default_config_dir())
    }

    /// Loads config from a YAML file, returning defaults when missing.
    pub fn load_from_file(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }

        let yaml = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;

        Self::parse(&yaml)
    }

    /// Loads sorted YAML config files from a directory and merges them.
    pub fn load_from_dir(path: &Path) -> Result<Self> {
        let values = list_config_files(path)?
            .iter()
            .map(read_config_value)
            .collect::<Result<Vec<_>>>()?;

        if values.is_empty() {
            return Ok(Self::default());
        }

        Self::deserialize_config_value(merge_yaml_values(values)?)
    }

    /// Parses config YAML with strict unknown-field detection.
    pub fn parse(yaml: &str) -> Result<Self> {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml)
            .context("Failed to parse YAML config")?;
        Self::deserialize_config_value(value)
    }

    /// Writes config to default config directory.
    pub fn save(&self) -> Result<PathBuf> {
        self.save_to_file(&default_config_file())
    }

    /// Writes config to a YAML file.
    pub fn save_to_file(&self, path: &Path) -> Result<PathBuf> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create {}", parent.display())
            })?;
        }

        let yaml = serde_yaml::to_string(self)
            .context("Failed to serialize YAML config")?;
        fs::write(path, yaml)
            .with_context(|| format!("Failed to write {}", path.display()))?;

        Ok(path.to_path_buf())
    }

    fn deserialize_config_value(value: serde_yaml::Value) -> Result<Self> {
        let mut unknown_fields = Vec::<String>::new();
        let deserializer = value.into_deserializer();
        let config = serde_ignored::deserialize(deserializer, |path| {
            unknown_fields.push(path.to_string())
        })
        .context("Failed to deserialize configuration")?;

        if unknown_fields.is_empty() {
            return Ok(config);
        }

        unknown_fields.sort();
        unknown_fields.dedup();
        Err(anyhow!(
            "Unknown config field(s): {}",
            unknown_fields.join(", ")
        ))
    }
}

/// Returns default YAML config file path.
pub fn default_config_file() -> PathBuf {
    default_config_dir()
        .join(APP_NAME)
        .with_extension(CONFIG_FILE_EXTENSION)
}

fn default_config_dir() -> PathBuf {
    get_app_dir(AppDirType::Config)
}

fn list_config_files(path: &Path) -> Result<Vec<PathBuf>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut files = fs::read_dir(path)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|path| path.is_file() && is_config_file(path))
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn is_config_file(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str())
    else {
        return false;
    };

    file_name.ends_with(".yaml") || file_name.ends_with(".yml")
}

fn read_config_value(path: &PathBuf) -> Result<serde_yaml::Value> {
    let yaml = fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    serde_yaml::from_str(&yaml)
        .with_context(|| format!("Failed to parse {}", path.display()))
}

fn merge_yaml_values(
    values: Vec<serde_yaml::Value>,
) -> Result<serde_yaml::Value> {
    values
        .into_iter()
        .try_fold(serde_yaml::Value::Null, merge_yaml_value)
}

fn merge_yaml_value(
    base: serde_yaml::Value,
    next: serde_yaml::Value,
) -> Result<serde_yaml::Value> {
    match (base, next) {
        (base, serde_yaml::Value::Null) => Ok(base),
        (serde_yaml::Value::Null, next) => Ok(next),
        (
            serde_yaml::Value::Mapping(mut base),
            serde_yaml::Value::Mapping(next),
        ) => {
            for (key, value) in next {
                let merged = match base.remove(&key) {
                    Some(existing) => merge_yaml_value(existing, value)?,
                    None => value,
                };
                base.insert(key, merged);
            }
            Ok(serde_yaml::Value::Mapping(base))
        }
        (serde_yaml::Value::Sequence(_), serde_yaml::Value::Sequence(next)) => {
            Ok(serde_yaml::Value::Sequence(next))
        }
        (_, next) => Ok(next),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use crate::config::AppConfig;

    #[test]
    fn parse_rejects_unknown_fields() -> Result<()> {
        let yaml = "indexing:\n  unknown: true\n";

        let error = AppConfig::parse(yaml).expect_err("unknown field rejected");

        assert!(error.to_string().contains("Unknown config field"));
        Ok(())
    }
}
