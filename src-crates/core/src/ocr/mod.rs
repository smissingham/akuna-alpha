//! Image OCR engines and OCR-specific result geometry.
//!
//! Keep OCR-specific ML/runtime concerns here.
//! Domain extraction structures live in [`crate::extraction`].

mod error;
mod glm_ocr;
mod layout;
mod output;
mod pp_ocr;

use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use burn_wgpu::{Wgpu, WgpuDevice};

use self::glm_ocr::{GlmOcrModel, GlmOcrVariant, load_glm_ocr};
use self::layout::{
    crop_text_region, text_like_detection, useful_ocr_fragment,
};
use self::pp_ocr::runtime::PpOcrRuntime;
use self::pp_ocr::spec::PpOcrV6Tier;
use crate::layout::pp_doclayout::{
    PpDocLayoutRuntime, load_pp_doclayout_runtime,
};
pub use error::OcrError;
pub(crate) use output::Rect;
pub use output::{OcrBlock, OcrBlockKind, OcrPage, OcrRect};

/// Default OCR backend.
pub type DefaultBackend = Wgpu;

/// Default OCR device.
pub type DefaultDevice = WgpuDevice;

#[derive(Debug)]
enum LoadedOcrModel<B: Backend> {
    Glm(Box<GlmOcrModel<B>>),
    LayoutGlm {
        layout: Box<PpDocLayoutRuntime<B>>,
        recognizer: Box<GlmOcrModel<B>>,
    },
    PpOcr(Box<PpOcrRuntime<B>>),
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
    pub async fn new(options: OcrOptions) -> Result<Self, OcrError> {
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
    ) -> Result<Self, OcrError> {
        let model = match (options.detector, options.recognizer) {
            (OcrDetector::PpDocLayout, OcrRecognizer::GlmOcr) => {
                let layout = load_pp_doclayout_runtime(
                    device,
                    options.cache_dir.clone(),
                )
                .await
                .map_err(|source| OcrError::Load { source })?;
                let recognizer = load_glm_ocr(
                    device,
                    GlmOcrVariant::OnnxCommunity,
                    options.cache_dir,
                )
                .await
                .map_err(|source| OcrError::Load { source })?;
                LoadedOcrModel::LayoutGlm {
                    layout: Box::new(layout),
                    recognizer: Box::new(recognizer),
                }
            }
            (OcrDetector::None, OcrRecognizer::GlmOcr) => {
                LoadedOcrModel::Glm(Box::new(
                    load_glm_ocr(
                        device,
                        GlmOcrVariant::OnnxCommunity,
                        options.cache_dir,
                    )
                    .await
                    .map_err(|source| OcrError::Load { source })?,
                ))
            }
            (OcrDetector::PpOcrV6TinyDet, OcrRecognizer::PpOcrV6TinyRec) => {
                LoadedOcrModel::PpOcr(Box::new(
                    PpOcrRuntime::load(
                        PpOcrV6Tier::Tiny,
                        device,
                        options.cache_dir,
                    )
                    .await
                    .map_err(|source| OcrError::Load { source })?,
                ))
            }
            (OcrDetector::PpOcrV6SmallDet, OcrRecognizer::PpOcrV6SmallRec) => {
                LoadedOcrModel::PpOcr(Box::new(
                    PpOcrRuntime::load(
                        PpOcrV6Tier::Small,
                        device,
                        options.cache_dir,
                    )
                    .await
                    .map_err(|source| OcrError::Load { source })?,
                ))
            }
            (
                OcrDetector::PpOcrV6MediumDet,
                OcrRecognizer::PpOcrV6MediumRec,
            ) => LoadedOcrModel::PpOcr(Box::new(
                PpOcrRuntime::load(
                    PpOcrV6Tier::Medium,
                    device,
                    options.cache_dir,
                )
                .await
                .map_err(|source| OcrError::Load { source })?,
            )),
            (
                detector @ (OcrDetector::PpOcrV6TinyDet
                | OcrDetector::PpOcrV6SmallDet
                | OcrDetector::PpOcrV6MediumDet),
                recognizer,
            )
            | (
                detector,
                recognizer @ (OcrRecognizer::PpOcrV6TinyRec
                | OcrRecognizer::PpOcrV6SmallRec
                | OcrRecognizer::PpOcrV6MediumRec),
            ) => {
                return Err(OcrError::UnsupportedPipeline {
                    detector,
                    recognizer,
                });
            }
        };

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    /// Extracts text from an image file.
    pub fn extract_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<String, OcrError> {
        Ok(self.extract_page_file(path)?.plain_text())
    }

    /// Extracts OCR blocks from an image file.
    pub fn extract_page_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<OcrPage, OcrError> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|source| OcrError::ReadFile {
                path: path.to_path_buf(),
                source,
            })?;

        self.extract_page_bytes(&bytes)
    }

    /// Extracts text from encoded image bytes.
    pub fn extract_bytes(&self, bytes: &[u8]) -> Result<String, OcrError> {
        Ok(self.extract_page_bytes(bytes)?.plain_text())
    }

    /// Extracts OCR blocks from encoded image bytes.
    pub fn extract_page_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<OcrPage, OcrError> {
        match &self.model {
            LoadedOcrModel::Glm(model) => extract_full_page_block(
                bytes,
                model
                    .extract_bytes(bytes, &self.device)
                    .map_err(|source| OcrError::Inference { source })?,
            ),
            LoadedOcrModel::LayoutGlm { layout, recognizer } => {
                extract_page_with_layout_and_glm(
                    bytes,
                    &self.device,
                    layout,
                    recognizer,
                )
            }
            LoadedOcrModel::PpOcr(model) => {
                let image = image::load_from_memory(bytes)
                    .map_err(|source| OcrError::DecodeImage { source })?;
                model
                    .extract_page(&image, &self.device)
                    .map_err(|source| OcrError::Inference { source })
            }
        }
    }

    /// Returns configured detector and recognizer.
    pub fn pipeline(&self) -> (OcrDetector, OcrRecognizer) {
        match &self.model {
            LoadedOcrModel::Glm(model) => match model.variant() {
                GlmOcrVariant::OnnxCommunity => {
                    (OcrDetector::None, OcrRecognizer::GlmOcr)
                }
            },
            LoadedOcrModel::LayoutGlm { .. } => {
                (OcrDetector::PpDocLayout, OcrRecognizer::GlmOcr)
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

fn extract_full_page_block(
    bytes: &[u8],
    text: String,
) -> Result<OcrPage, OcrError> {
    let image = image::load_from_memory(bytes)
        .map_err(|source| OcrError::DecodeImage { source })?;
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

fn extract_page_with_layout_and_glm<B: Backend<FloatElem = f32>>(
    bytes: &[u8],
    device: &B::Device,
    layout: &PpDocLayoutRuntime<B>,
    recognizer: &GlmOcrModel<B>,
) -> Result<OcrPage, OcrError> {
    let image = image::load_from_memory(bytes)
        .map_err(|source| OcrError::DecodeImage { source })?;
    let detections = layout
        .detect_image(&image, device)
        .map_err(|source| OcrError::Inference { source })?;

    let mut blocks = Vec::new();
    for detection in detections.into_iter().filter(text_like_detection) {
        let Ok(crop) = crop_text_region(&image, detection.bbox) else {
            continue;
        };
        let text = recognizer
            .extract_image(&crop, device)
            .map_err(|source| OcrError::Inference { source })?;
        let text = text.trim();
        if useful_ocr_fragment(text) {
            blocks.push(OcrBlock {
                text: text.to_string(),
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
    if blocks.is_empty() {
        return extract_full_page_block(
            bytes,
            recognizer
                .extract_image(&image, device)
                .map_err(|source| OcrError::Inference { source })?,
        );
    }

    Ok(OcrPage {
        width: image.width(),
        height: image.height(),
        blocks,
    })
}
