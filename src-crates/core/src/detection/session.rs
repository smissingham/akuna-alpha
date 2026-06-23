use std::path::Path;

use burn::tensor::backend::Backend;

use crate::detection::models::magika::MagikaModel;
use crate::detection::{Detection, FileType, MagikaInferenceError};

/// Default Burn backend used by the detection module.
pub(crate) type DefaultBackend = burn_wgpu::Wgpu;

/// Convenience alias for a [`Session`] backed by the default WGPU backend.
pub type DefaultSession = Session<DefaultBackend>;

/// High-level Magika inference session wrapping a Magika classifier.
pub struct Session<B: Backend> {
    model: MagikaModel<B>,
}

impl Session<DefaultBackend> {
    /// Builds a session on the default WGPU device.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded model cannot be loaded.
    pub fn new_default() -> Result<Self, MagikaInferenceError> {
        Self::new(&burn_wgpu::WgpuDevice::default())
    }
}

impl<B: Backend<FloatElem = f32>> Session<B> {
    /// Builds a session from the embedded model on the given device.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedded model cannot be loaded.
    pub fn new(device: &B::Device) -> Result<Self, MagikaInferenceError> {
        let model = MagikaModel::<B>::from_embedded(device)?;
        Ok(Self { model })
    }

    /// Builds a session from a model file on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the model is invalid.
    pub fn from_file(
        device: &B::Device,
        path: impl AsRef<Path>,
    ) -> Result<Self, MagikaInferenceError> {
        let model = MagikaModel::<B>::from_file(device, path)?;
        Ok(Self { model })
    }

    /// Builds a session from raw model bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the model bytes cannot be parsed.
    pub fn from_bytes(
        device: &B::Device,
        bytes: &[u8],
    ) -> Result<Self, MagikaInferenceError> {
        let model = MagikaModel::<B>::from_bytes(device, bytes)?;
        Ok(Self { model })
    }

    /// Overrides the number of ranked alternatives returned per detection.
    pub fn with_top_k(self, top_k: usize) -> Self {
        Self {
            model: self.model.with_top_k(top_k),
        }
    }

    /// Identifies the file type of a file on disk (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or inference fails.
    pub fn identify_file_sync(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<FileType, MagikaInferenceError> {
        self.model.identify_path(path)
    }

    /// Identifies the file type of raw bytes (blocking).
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn identify_content_sync(
        &mut self,
        bytes: &[u8],
    ) -> Result<FileType, MagikaInferenceError> {
        self.model.identify_bytes(bytes)
    }

    /// Classifies the file at `path` and returns ranked alternatives (blocking).
    ///
    /// Wrap in `tokio::task::spawn_blocking` if calling from an async runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or inference fails.
    pub fn detect_file_sync(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<Detection, MagikaInferenceError> {
        self.model.detect_path(path)
    }

    /// Classifies raw bytes and returns ranked alternatives (blocking).
    ///
    /// Wrap in `tokio::task::spawn_blocking` if calling from an async runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn detect_content_sync(
        &self,
        bytes: &[u8],
    ) -> Result<Detection, MagikaInferenceError> {
        self.model.detect_bytes(bytes)
    }

    /// Classifies a batch of inputs and returns ranked alternatives each.
    ///
    /// # Errors
    ///
    /// Returns an error if any input fails inference.
    pub fn detect_content_batch_sync(
        &self,
        inputs: Vec<&[u8]>,
    ) -> Result<Vec<Detection>, MagikaInferenceError> {
        self.model.detect_batch(inputs)
    }
}
