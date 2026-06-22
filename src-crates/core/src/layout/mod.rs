//! Document layout detection.
//!
//! Layout detection is separate from OCR so callers can use page structure
//! without enabling text recognition engines.

use std::path::PathBuf;

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};
use image::DynamicImage;

pub(crate) mod pp_doclayout;

use crate::layout::pp_doclayout::{
    PpDocLayoutRuntime, load_pp_doclayout_runtime,
};

/// Default layout backend.
pub type DefaultLayoutBackend = Wgpu;

/// Default layout device.
pub type DefaultLayoutDevice = WgpuDevice;

/// Layout detector options.
#[derive(Debug, Clone, Default)]
pub struct LayoutOptions {
    /// Optional model download cache directory.
    pub cache_dir: Option<PathBuf>,
}

/// Layout detector runtime.
#[derive(Debug)]
pub struct LayoutDetector<B: Backend = DefaultLayoutBackend> {
    runtime: PpDocLayoutRuntime<B>,
    device: B::Device,
}

/// Layout output for one page/image.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct LayoutPage {
    /// Source image width in pixels.
    pub width: u32,
    /// Source image height in pixels.
    pub height: u32,
    /// Detected layout blocks in reading order.
    pub blocks: Vec<LayoutBlock>,
}

/// Detected layout block.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct LayoutBlock {
    /// Detector label.
    pub label: String,
    /// Detector confidence from 0 to 1.
    pub confidence: f32,
    /// Bounding box in image pixel coordinates.
    pub bbox: LayoutRect,
    /// Detector reading order.
    pub order: i64,
}

/// Axis-aligned layout bounding box.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct LayoutRect {
    /// Left coordinate in pixels.
    pub x: f32,
    /// Top coordinate in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
}

impl LayoutDetector<DefaultLayoutBackend> {
    /// Load default layout detector.
    ///
    /// # Errors
    ///
    /// Returns an error when model files cannot be loaded.
    pub async fn new(options: LayoutOptions) -> Result<Self> {
        let device = WgpuDevice::default();
        Self::new_with_device(&device, options).await
    }
}

impl<B> LayoutDetector<B>
where
    B: Backend<FloatElem = f32>,
{
    /// Load layout detector on caller-provided device.
    ///
    /// # Errors
    ///
    /// Returns an error when model files cannot be loaded.
    pub async fn new_with_device(
        device: &B::Device,
        options: LayoutOptions,
    ) -> Result<Self> {
        let runtime =
            load_pp_doclayout_runtime(device, options.cache_dir).await?;

        Ok(Self {
            runtime,
            device: device.clone(),
        })
    }

    /// Detect layout blocks from decoded image.
    ///
    /// # Errors
    ///
    /// Returns an error when preprocessing or inference fails.
    pub fn detect_image(&self, image: &DynamicImage) -> Result<LayoutPage> {
        let blocks = self
            .runtime
            .detect_image(image, &self.device)?
            .into_iter()
            .map(|detection| LayoutBlock {
                label: detection.label,
                confidence: detection.score,
                bbox: LayoutRect {
                    x: detection.bbox[0],
                    y: detection.bbox[1],
                    width: detection.bbox[2] - detection.bbox[0],
                    height: detection.bbox[3] - detection.bbox[1],
                },
                order: detection.order,
            })
            .collect();

        Ok(LayoutPage {
            width: image.width(),
            height: image.height(),
            blocks,
        })
    }
}
