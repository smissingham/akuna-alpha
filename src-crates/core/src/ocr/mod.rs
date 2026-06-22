//! Image OCR engines and OCR-specific result geometry.
//!
//! Keep OCR-specific ML/runtime concerns here.
//! Domain extraction structures live in [`crate::extraction`].

mod glm_ocr;
mod models;
mod ocrs_ocr;
mod pp_ocr;
mod text_pipeline;

use std::path::{Path, PathBuf};

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};

use self::glm_ocr::{GlmOcrModel, GlmOcrVariant, load_glm_ocr};
use self::ocrs_ocr::{OcrsRtenModel, load_ocrs_rten};
use self::pp_ocr::runtime::PpOcrRuntime;
use self::pp_ocr::spec::PpOcrV6Tier;
use self::text_pipeline::{
    crop_text_region, text_like_detection, useful_ocr_fragment,
};
use crate::layout::pp_doclayout::{
    PpDocLayoutRuntime, load_pp_doclayout_runtime,
};

/// Default OCR backend.
pub type DefaultBackend = Wgpu;

/// Default OCR device.
pub type DefaultDevice = WgpuDevice;

#[derive(Debug)]
enum LoadedOcrModel<B: Backend> {
    Glm(GlmOcrModel<B>),
    Ocrs(OcrsRtenModel),
    LayoutOcrs {
        layout: Option<PpDocLayoutRuntime<B>>,
        recognizer: OcrsRtenModel,
    },
    PpOcr(PpOcrRuntime<B>),
}

/// Region detection strategy.
#[non_exhaustive]
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub enum OcrDetector {
    /// PP-DocLayoutV3 document region detector.
    PpDocLayout,
    /// OCRS RTen word detector.
    OcrsRten,
    /// PaddleOCR PP-OCRv6 tiny detector.
    PpOcrV6TinyDet,
    /// PaddleOCR PP-OCRv6 small detector.
    PpOcrV6SmallDet,
    /// PaddleOCR PP-OCRv6 medium detector.
    #[default]
    PpOcrV6MediumDet,
    /// No explicit detection; recognizer receives whole image.
    None,
}

/// Text recognition strategy.
#[non_exhaustive]
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub enum OcrRecognizer {
    /// Z.ai GLM OCR recognizer.
    GlmOcr,
    /// OCRS RTen recognizer.
    OcrsRten,
    /// PaddleOCR PP-OCRv6 tiny recognizer.
    PpOcrV6TinyRec,
    /// PaddleOCR PP-OCRv6 small recognizer.
    PpOcrV6SmallRec,
    /// PaddleOCR PP-OCRv6 medium recognizer.
    #[default]
    PpOcrV6MediumRec,
}

/// Options for OCR model loading and inference.
#[derive(Debug, Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct OcrOptions {
    /// Region detector used before recognition.
    pub detector: OcrDetector,
    /// Text recognizer used after detection.
    pub recognizer: OcrRecognizer,
    /// Optional model download cache directory.
    pub cache_dir: Option<PathBuf>,
}

/// Minimal OCR interface for file and byte extraction.
#[derive(Debug)]
pub struct Ocr<B: Backend = DefaultBackend> {
    model: LoadedOcrModel<B>,
    device: B::Device,
}

impl Ocr<DefaultBackend> {
    /// Loads OCR model onto default WGPU device.
    pub async fn new(options: OcrOptions) -> Result<Self> {
        let device = WgpuDevice::default();
        Self::new_with_device(&device, options).await
    }
}

impl<B> Ocr<B>
where
    B: Backend<FloatElem = f32>,
{
    /// Loads OCR model onto provided device.
    pub async fn new_with_device(
        device: &B::Device,
        options: OcrOptions,
    ) -> Result<Self> {
        let model = match (options.detector, options.recognizer) {
            (OcrDetector::PpDocLayout, OcrRecognizer::GlmOcr)
            | (OcrDetector::None, OcrRecognizer::GlmOcr) => {
                LoadedOcrModel::Glm(
                    load_glm_ocr(
                        device,
                        GlmOcrVariant::OnnxCommunity,
                        options.cache_dir,
                    )
                    .await?,
                )
            }
            (OcrDetector::OcrsRten, OcrRecognizer::OcrsRten)
            | (OcrDetector::None, OcrRecognizer::OcrsRten) => {
                LoadedOcrModel::Ocrs(load_ocrs_rten(options.cache_dir).await?)
            }
            (OcrDetector::PpDocLayout, OcrRecognizer::OcrsRten) => {
                let layout = load_pp_doclayout_runtime(
                    device,
                    options.cache_dir.clone(),
                )
                .await
                .ok();
                let recognizer = load_ocrs_rten(options.cache_dir).await?;
                LoadedOcrModel::LayoutOcrs { layout, recognizer }
            }
            (OcrDetector::PpOcrV6TinyDet, OcrRecognizer::PpOcrV6TinyRec) => {
                LoadedOcrModel::PpOcr(
                    PpOcrRuntime::load(
                        PpOcrV6Tier::Tiny,
                        device,
                        options.cache_dir,
                    )
                    .await?,
                )
            }
            (OcrDetector::PpOcrV6SmallDet, OcrRecognizer::PpOcrV6SmallRec) => {
                LoadedOcrModel::PpOcr(
                    PpOcrRuntime::load(
                        PpOcrV6Tier::Small,
                        device,
                        options.cache_dir,
                    )
                    .await?,
                )
            }
            (
                OcrDetector::PpOcrV6MediumDet,
                OcrRecognizer::PpOcrV6MediumRec,
            ) => LoadedOcrModel::PpOcr(
                PpOcrRuntime::load(
                    PpOcrV6Tier::Medium,
                    device,
                    options.cache_dir,
                )
                .await?,
            ),
            (OcrDetector::OcrsRten, OcrRecognizer::GlmOcr) => {
                anyhow::bail!(
                    "OCRS detection cannot currently feed GLM recognition"
                )
            }
            (
                OcrDetector::PpOcrV6TinyDet
                | OcrDetector::PpOcrV6SmallDet
                | OcrDetector::PpOcrV6MediumDet,
                _,
            )
            | (
                _,
                OcrRecognizer::PpOcrV6TinyRec
                | OcrRecognizer::PpOcrV6SmallRec
                | OcrRecognizer::PpOcrV6MediumRec,
            ) => anyhow::bail!(
                "PP-OCRv6 detector and recognizer tiers must match"
            ),
        };

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    /// Extracts text from an image file.
    pub fn extract_file(&self, path: impl AsRef<Path>) -> Result<String> {
        Ok(self.extract_page_file(path)?.plain_text())
    }

    /// Extracts OCR blocks from an image file.
    pub fn extract_page_file(&self, path: impl AsRef<Path>) -> Result<OcrPage> {
        let bytes = std::fs::read(path.as_ref()).map_err(|error| {
            anyhow::anyhow!(
                "failed to read OCR input file {}: {error}",
                path.as_ref().display()
            )
        })?;

        self.extract_page_bytes(&bytes)
    }

    /// Extracts text from encoded image bytes.
    pub fn extract_bytes(&self, bytes: &[u8]) -> Result<String> {
        Ok(self.extract_page_bytes(bytes)?.plain_text())
    }

    /// Extracts OCR blocks from encoded image bytes.
    pub fn extract_page_bytes(&self, bytes: &[u8]) -> Result<OcrPage> {
        match &self.model {
            LoadedOcrModel::Glm(model) => extract_full_page_block(
                bytes,
                model.extract_bytes(bytes, &self.device)?,
            ),
            LoadedOcrModel::Ocrs(model) => {
                extract_full_page_block(bytes, model.extract_bytes(bytes)?)
            }
            LoadedOcrModel::LayoutOcrs { layout, recognizer } => {
                extract_page_with_layout_and_ocrs(
                    bytes,
                    &self.device,
                    layout,
                    recognizer,
                )
            }
            LoadedOcrModel::PpOcr(model) => {
                let image =
                    image::load_from_memory(bytes).map_err(|error| {
                        anyhow::anyhow!(
                            "failed to decode OCR input image: {error}"
                        )
                    })?;
                model.extract_page(&image, &self.device)
            }
        }
    }

    /// Returns configured detector and recognizer.
    pub fn pipeline(&self) -> (OcrDetector, OcrRecognizer) {
        match &self.model {
            LoadedOcrModel::Glm(model) => match model.variant() {
                GlmOcrVariant::OnnxCommunity => {
                    (OcrDetector::PpDocLayout, OcrRecognizer::GlmOcr)
                }
            },
            LoadedOcrModel::Ocrs(_) => {
                (OcrDetector::OcrsRten, OcrRecognizer::OcrsRten)
            }
            LoadedOcrModel::LayoutOcrs { .. } => {
                (OcrDetector::PpDocLayout, OcrRecognizer::OcrsRten)
            }
            LoadedOcrModel::PpOcr(model) => match model.tier() {
                PpOcrV6Tier::Tiny => {
                    (OcrDetector::PpOcrV6TinyDet, OcrRecognizer::PpOcrV6TinyRec)
                }
                PpOcrV6Tier::Small => (
                    OcrDetector::PpOcrV6SmallDet,
                    OcrRecognizer::PpOcrV6SmallRec,
                ),
                PpOcrV6Tier::Medium => (
                    OcrDetector::PpOcrV6MediumDet,
                    OcrRecognizer::PpOcrV6MediumRec,
                ),
            },
        }
    }
}

fn extract_full_page_block(bytes: &[u8], text: String) -> Result<OcrPage> {
    let image = image::load_from_memory(bytes).map_err(|error| {
        anyhow::anyhow!("failed to decode OCR input image: {error}")
    })?;
    let bbox = OcrRect {
        x: 0.0,
        y: 0.0,
        width: image.width() as f32,
        height: image.height() as f32,
    };

    Ok(OcrPage {
        width: image.width(),
        height: image.height(),
        blocks: vec![OcrBlock {
            text,
            bbox,
            confidence: None,
            kind: OcrBlockKind::Unknown,
        }],
    })
}

fn extract_page_with_layout_and_ocrs<B: Backend<FloatElem = f32>>(
    bytes: &[u8],
    device: &B::Device,
    layout: &Option<PpDocLayoutRuntime<B>>,
    recognizer: &OcrsRtenModel,
) -> Result<OcrPage> {
    let image = image::load_from_memory(bytes).map_err(|error| {
        anyhow::anyhow!("failed to decode OCR input image: {error}")
    })?;
    let Some(layout) = layout else {
        return extract_full_page_block(
            bytes,
            recognizer.extract_image(&image)?,
        );
    };
    let detections = layout.detect_image(&image, device).unwrap_or_default();
    let mut blocks = Vec::new();
    for detection in detections.into_iter().filter(text_like_detection) {
        let Ok(crop) = crop_text_region(&image, detection.bbox) else {
            continue;
        };
        let text = recognizer.extract_image(&crop)?;
        for line in text.lines().map(str::trim) {
            if useful_ocr_fragment(line) {
                blocks.push(OcrBlock {
                    text: line.to_string(),
                    bbox: OcrRect {
                        x: detection.bbox[0],
                        y: detection.bbox[1],
                        width: detection.bbox[2] - detection.bbox[0],
                        height: detection.bbox[3] - detection.bbox[1],
                    },
                    confidence: Some(detection.score),
                    kind: OcrBlockKind::Text,
                });
            }
        }
    }
    if blocks.is_empty() {
        return extract_full_page_block(
            bytes,
            recognizer.extract_image(&image)?,
        );
    }

    Ok(OcrPage {
        width: image.width(),
        height: image.height(),
        blocks,
    })
}

/// OCR output for one image/page.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct OcrPage {
    /// Source image width in pixels.
    pub width: u32,
    /// Source image height in pixels.
    pub height: u32,
    /// OCR blocks in reading order.
    pub blocks: Vec<OcrBlock>,
}

/// OCR text block with geometry and confidence.
#[derive(
    Debug,
    Clone,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct OcrBlock {
    /// Recognized text.
    pub text: String,
    /// Bounding box in image pixel coordinates.
    pub bbox: OcrRect,
    /// Recognition confidence from 0 to 1 when available.
    pub confidence: Option<f32>,
    /// OCR-local block kind.
    pub kind: OcrBlockKind,
}

/// Axis-aligned OCR bounding box.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub struct OcrRect {
    /// Left coordinate in pixels.
    pub x: f32,
    /// Top coordinate in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
}

impl OcrRect {
    fn from_points(points: [[f32; 2]; 4]) -> Self {
        let min_x = points
            .iter()
            .map(|point| point[0])
            .fold(f32::INFINITY, f32::min);
        let min_y = points
            .iter()
            .map(|point| point[1])
            .fold(f32::INFINITY, f32::min);
        let max_x = points
            .iter()
            .map(|point| point[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let max_y = points
            .iter()
            .map(|point| point[1])
            .fold(f32::NEG_INFINITY, f32::max);

        Self {
            x: min_x,
            y: min_y,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }
}

pub(crate) type Rect = OcrRect;

/// OCR-local block classification.
#[non_exhaustive]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Deserialize,
    serde::Serialize,
    utoipa::ToSchema,
)]
pub enum OcrBlockKind {
    /// Regular text.
    Text,
    /// Unknown OCR block kind.
    Unknown,
}

impl OcrPage {
    /// Render OCR blocks as newline-separated plain text.
    #[must_use]
    pub fn plain_text(&self) -> String {
        self.blocks
            .iter()
            .map(|block| block.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
