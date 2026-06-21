use std::path::Path;

use burn::tensor::backend::Backend;

use crate::model::MagikaModel;
use crate::{Detection, FileType, MagikaInferenceError};

/// Default Burn backend used by `akuna-core-detection`.
pub(crate) type DefaultBackend = burn_wgpu::Wgpu;

/// Default device for the [`DefaultBackend`].
pub(crate) type DefaultDevice = burn_wgpu::WgpuDevice;

/// Convenience alias for a [`Session`] backed by the [`DefaultBackend`].
pub type DefaultSession = Session<DefaultBackend>;

/// High-level Magika inference session wrapping a [`MagikaModel`].
pub struct Session<B: Backend> {
    model: MagikaModel<B>,
}

impl Session<DefaultBackend> {
    /// Builds a session on the default WGPU device.
    pub fn new_default() -> Result<Self, MagikaInferenceError> {
        Self::new(&DefaultDevice::DefaultDevice)
    }
}

impl<B: Backend<FloatElem = f32>> Session<B> {
    /// Builds a session from the embedded model on the given device.
    pub fn new(device: &B::Device) -> Result<Self, MagikaInferenceError> {
        let model = MagikaModel::<B>::from_embedded(device)?;
        Ok(Self { model })
    }

    /// Builds a session from a model file on disk.
    pub fn from_file(
        device: &B::Device,
        path: impl AsRef<Path>,
    ) -> Result<Self, MagikaInferenceError> {
        let model = MagikaModel::<B>::from_file(device, path)?;
        Ok(Self { model })
    }

    /// Builds a session from raw model bytes.
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
    pub fn identify_file_sync(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<FileType, MagikaInferenceError> {
        self.model.identify_path(path)
    }

    /// Async variant of [`identify_file_sync`](Self::identify_file_sync).
    pub async fn identify_file_async(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<FileType, MagikaInferenceError> {
        self.model.identify_path(path)
    }

    /// Identifies the file type of raw bytes (blocking).
    pub fn identify_content_sync(
        &mut self,
        bytes: &[u8],
    ) -> Result<FileType, MagikaInferenceError> {
        self.model.identify_bytes(bytes)
    }

    /// Async variant of [`identify_content_sync`](Self::identify_content_sync).
    pub async fn identify_content_async(
        &mut self,
        bytes: &[u8],
    ) -> Result<FileType, MagikaInferenceError> {
        self.model.identify_bytes(bytes)
    }

    /// Classifies the file at `path` and returns ranked alternatives (blocking).
    pub fn detect_file_sync(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<Detection, MagikaInferenceError> {
        self.model.detect_path(path)
    }

    /// Async variant of [`detect_file_sync`](Self::detect_file_sync).
    pub async fn detect_file_async(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<Detection, MagikaInferenceError> {
        self.model.detect_path(path)
    }

    /// Classifies raw bytes and returns ranked alternatives (blocking).
    pub fn detect_content_sync(
        &self,
        bytes: &[u8],
    ) -> Result<Detection, MagikaInferenceError> {
        self.model.detect_bytes(bytes)
    }

    /// Async variant of [`detect_content_sync`](Self::detect_content_sync).
    pub async fn detect_content_async(
        &self,
        bytes: &[u8],
    ) -> Result<Detection, MagikaInferenceError> {
        self.model.detect_bytes(bytes)
    }

    /// Classifies a batch of inputs and returns ranked alternatives each.
    pub fn detect_content_batch_sync(
        &self,
        inputs: Vec<&[u8]>,
    ) -> Result<Vec<Detection>, MagikaInferenceError> {
        self.model.detect_batch(inputs)
    }
}
