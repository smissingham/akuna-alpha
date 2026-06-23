use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Bytes, Tensor, TensorData};
use hf_hub::Repo;
use hf_hub::RepoType;
use hf_hub::api::sync::Api;
use image::{DynamicImage, GenericImageView};

use crate::ocr::models::pp_ocr::dictionary::load_dictionary;
use crate::ocr::models::pp_ocr::generated::{
    pp_ocrv6_medium_det, pp_ocrv6_medium_rec, pp_ocrv6_small_det,
    pp_ocrv6_small_rec, pp_ocrv6_tiny_det, pp_ocrv6_tiny_rec,
};
use crate::ocr::models::pp_ocr::postprocess::{
    postprocess_detector, postprocess_recognizer,
};
use crate::ocr::models::pp_ocr::preprocess::{
    PpOcrInput, preprocess_detector, preprocess_recognizer,
};
use crate::ocr::models::pp_ocr::spec::{
    PpOcrV6Tier, detector_config, recognizer_config,
};
use crate::ocr::{OcrBlock, OcrBlockKind, OcrPage, OcrRect};

const CROP_PADDING_RATIO: f32 = 0.08;
const MIN_CROP_PADDING: f32 = 2.0;
const HF_MODELS_OWNER: &str = "AKUNA_BURN_MODELS_OWNER";
const DEFAULT_HF_MODELS_OWNER: Option<&str> =
    option_env!("AKUNA_BURN_MODELS_OWNER");
const REPO_NAME: &str = "PP-OCRv6";

#[derive(Debug)]
pub(crate) struct PpOcrRuntime<B: Backend> {
    tier: PpOcrV6Tier,
    detector: DetectorModel<B>,
    recognizer: RecognizerModel<B>,
    dictionary: Vec<String>,
}

#[derive(Debug)]
enum DetectorModel<B: Backend> {
    Tiny(Box<pp_ocrv6_tiny_det::Model<B>>),
    Small(Box<pp_ocrv6_small_det::Model<B>>),
    Medium(Box<pp_ocrv6_medium_det::Model<B>>),
}

#[derive(Debug)]
enum RecognizerModel<B: Backend> {
    Tiny(Box<pp_ocrv6_tiny_rec::Model<B>>),
    Small(Box<pp_ocrv6_small_rec::Model<B>>),
    Medium(Box<pp_ocrv6_medium_rec::Model<B>>),
}

impl<B> PpOcrRuntime<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn tier(&self) -> PpOcrV6Tier {
        self.tier
    }

    pub(crate) async fn load(
        tier: PpOcrV6Tier,
        device: &B::Device,
        cache_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let recognizer = recognizer_config(tier);
        let dictionary =
            load_dictionary(&recognizer, cache_dir.as_deref()).await?;
        let detector = DetectorModel::load(tier, device)?;
        let recognizer = RecognizerModel::load(tier, device)?;

        Ok(Self {
            tier,
            detector,
            recognizer,
            dictionary,
        })
    }

    pub(crate) fn extract_page(
        &self,
        image: &DynamicImage,
        device: &B::Device,
    ) -> Result<OcrPage> {
        let detector_config = detector_config(self.tier);
        let detector_input = preprocess_detector(image, &detector_config)?;
        let original_width = detector_input.original_width;
        let original_height = detector_input.original_height;
        let detector_tensor = input_tensor(detector_input, device);
        let detector_output = self.detector.forward(detector_tensor);
        let boxes = postprocess_detector(
            detector_output,
            &detector_config,
            original_width,
            original_height,
        )?;

        let recognizer_config = recognizer_config(self.tier);
        let mut blocks = Vec::new();
        for text_box in boxes {
            let crop = crop_box(image, text_box.points)?;
            let recognizer_input =
                preprocess_recognizer(&crop, &recognizer_config)?;
            let recognizer_tensor = input_tensor(recognizer_input, device);
            let recognizer_output = self.recognizer.forward(recognizer_tensor);
            let text = postprocess_recognizer(
                recognizer_output,
                &self.dictionary,
                &recognizer_config,
            )?;
            let trimmed = text.text.trim();
            if !trimmed.is_empty() {
                blocks.push(OcrBlock {
                    text: trimmed.to_string(),
                    bbox: OcrRect::from_points(text_box.points),
                    confidence: Some(text.confidence.min(text_box.score)),
                    kind: OcrBlockKind::Text,
                });
            }
        }

        if blocks.is_empty() {
            let text = self.recognize_crop(image, device)?;
            blocks.push(OcrBlock {
                text: text.text,
                bbox: OcrRect {
                    x: 0.0,
                    y: 0.0,
                    width: image.width() as f32,
                    height: image.height() as f32,
                },
                confidence: Some(text.confidence),
                kind: OcrBlockKind::Unknown,
            });
        }

        Ok(OcrPage {
            width: image.width(),
            height: image.height(),
            blocks,
        })
    }

    fn recognize_crop(
        &self,
        image: &DynamicImage,
        device: &B::Device,
    ) -> Result<crate::ocr::models::pp_ocr::postprocess::RecognizedText> {
        let recognizer_config = recognizer_config(self.tier);
        let recognizer_input =
            preprocess_recognizer(image, &recognizer_config)?;
        let recognizer_tensor = input_tensor(recognizer_input, device);
        let recognizer_output = self.recognizer.forward(recognizer_tensor);
        let text = postprocess_recognizer(
            recognizer_output,
            &self.dictionary,
            &recognizer_config,
        )?;

        Ok(text)
    }
}

impl<B: Backend> DetectorModel<B> {
    fn load(tier: PpOcrV6Tier, device: &B::Device) -> Result<Self> {
        match tier {
            PpOcrV6Tier::Tiny => {
                Ok(Self::Tiny(Box::new(pp_ocrv6_tiny_det::Model::from_bytes(
                    bpk("tiny/det.bpk")?,
                    device,
                ))))
            }
            PpOcrV6Tier::Small => Ok(Self::Small(Box::new(
                pp_ocrv6_small_det::Model::from_bytes(
                    bpk("small/det.bpk")?,
                    device,
                ),
            ))),
            PpOcrV6Tier::Medium => Ok(Self::Medium(Box::new(
                pp_ocrv6_medium_det::Model::from_bytes(
                    bpk("medium/det.bpk")?,
                    device,
                ),
            ))),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Tiny(model) => model.forward(input),
            Self::Small(model) => model.forward(input),
            Self::Medium(model) => model.forward(input),
        }
    }
}

impl<B: Backend> RecognizerModel<B> {
    fn load(tier: PpOcrV6Tier, device: &B::Device) -> Result<Self> {
        match tier {
            PpOcrV6Tier::Tiny => {
                Ok(Self::Tiny(Box::new(pp_ocrv6_tiny_rec::Model::from_bytes(
                    bpk("tiny/rec.bpk")?,
                    device,
                ))))
            }
            PpOcrV6Tier::Small => Ok(Self::Small(Box::new(
                pp_ocrv6_small_rec::Model::from_bytes(
                    bpk("small/rec.bpk")?,
                    device,
                ),
            ))),
            PpOcrV6Tier::Medium => Ok(Self::Medium(Box::new(
                pp_ocrv6_medium_rec::Model::from_bytes(
                    bpk("medium/rec.bpk")?,
                    device,
                ),
            ))),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        match self {
            Self::Tiny(model) => model.forward(input),
            Self::Small(model) => model.forward(input),
            Self::Medium(model) => model.forward(input),
        }
    }
}

fn bpk(relative_path: &str) -> Result<Bytes> {
    let repo_id = model_repo()?;
    let local_path = Path::new(&repo_id).join(relative_path);
    if local_path.exists() && local_path.is_file() {
        let bytes = std::fs::read(&local_path).with_context(|| {
            format!(
                "failed to read PP-OCR Burnpack weights at {}",
                local_path.display()
            )
        })?;

        return Ok(Bytes::from_bytes_vec(bytes));
    }

    let path_in_repo = relative_path.to_owned();
    let path = Api::new()
        .context("failed to initialize Hugging Face API for PP-OCR weights")?
        .repo(Repo::with_revision(
            repo_id.clone(),
            RepoType::Model,
            "main".to_owned(),
        ))
        .get(&path_in_repo)
        .with_context(|| {
            format!(
                "failed to fetch PP-OCR Burnpack weights '{path_in_repo}' from {repo_id}"
            )
        })?;
    let bytes = std::fs::read(&path).with_context(|| {
        format!(
            "failed to read PP-OCR Burnpack weights at {}",
            path.display()
        )
    })?;

    Ok(Bytes::from_bytes_vec(bytes))
}

fn model_repo() -> Result<String> {
    if let Ok(owner) = std::env::var(HF_MODELS_OWNER) {
        return Ok(format!("{owner}/{REPO_NAME}"));
    }

    DEFAULT_HF_MODELS_OWNER
        .map(|owner| format!("{owner}/{REPO_NAME}"))
        .with_context(|| {
            format!(
                "{HF_MODELS_OWNER} must be set at build time or runtime to fetch model assets"
            )
        })
}

fn input_tensor<B: Backend>(
    input: PpOcrInput,
    device: &B::Device,
) -> Tensor<B, 4> {
    let data = TensorData::new(
        input.values,
        [1, input.channels, input.height, input.width],
    );

    Tensor::<B, 4>::from_data(data, device)
}

fn crop_box(
    image: &DynamicImage,
    points: [[f32; 2]; 4],
) -> Result<DynamicImage> {
    let (image_width, image_height) = image.dimensions();
    let raw_min_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .floor();
    let raw_min_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .floor();
    let raw_max_x = points
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil();
    let raw_max_y = points
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil();

    let crop_width = raw_max_x - raw_min_x;
    let crop_height = raw_max_y - raw_min_y;
    let x_padding = (crop_width * CROP_PADDING_RATIO).max(MIN_CROP_PADDING);
    let y_padding = (crop_height * CROP_PADDING_RATIO).max(MIN_CROP_PADDING);

    let min_x = (raw_min_x - x_padding).clamp(0.0, image_width as f32) as u32;
    let min_y = (raw_min_y - y_padding).clamp(0.0, image_height as f32) as u32;
    let max_x = (raw_max_x + x_padding).clamp(0.0, image_width as f32) as u32;
    let max_y = (raw_max_y + y_padding).clamp(0.0, image_height as f32) as u32;

    if max_x <= min_x || max_y <= min_y {
        anyhow::bail!("PP-OCR detected invalid crop bounds")
    }

    Ok(image.crop_imm(min_x, min_y, max_x - min_x, max_y - min_y))
}
