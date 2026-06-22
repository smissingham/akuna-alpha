use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::{Repo, RepoType, api::tokio::ApiBuilder};
use image::DynamicImage;
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;

use crate::ocr::text_pipeline::join_useful_lines;

const OCRS_REPO_ID: &str = "robertknight/ocrs";
const OCRS_REVISION: &str = "df0edd170279ab971b53e094c627255a87e1a503";
const OCRS_DETECTION_FILE: &str = "text-detection-ssfbcj81.rten";
const OCRS_RECOGNITION_FILE: &str = "text-rec-checkpoint-s52qdbqt.rten";

pub(crate) struct OcrsRtenModel {
    engine: OcrEngine,
}

impl std::fmt::Debug for OcrsRtenModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OcrsRtenModel")
            .finish_non_exhaustive()
    }
}

pub(crate) async fn load_ocrs_rten(
    cache_dir: Option<PathBuf>,
) -> Result<OcrsRtenModel> {
    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir);
    }

    let api = builder
        .build()
        .context("failed to initialize Hugging Face API for OCRS model")?;
    let repo = api.repo(Repo::with_revision(
        OCRS_REPO_ID.to_string(),
        RepoType::Model,
        OCRS_REVISION.to_string(),
    ));

    let detection_path =
        repo.get(OCRS_DETECTION_FILE).await.with_context(|| {
            format!(
                "failed to fetch OCRS detection model {OCRS_DETECTION_FILE}"
            )
        })?;
    let recognition_path =
        repo.get(OCRS_RECOGNITION_FILE).await.with_context(|| {
            format!(
                "failed to fetch OCRS recognition model {OCRS_RECOGNITION_FILE}"
            )
        })?;

    let detection_model =
        Model::load_file(&detection_path).with_context(|| {
            format!("failed to load {}", detection_path.display())
        })?;
    let recognition_model =
        Model::load_file(&recognition_path).with_context(|| {
            format!("failed to load {}", recognition_path.display())
        })?;
    let engine = OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        ..Default::default()
    })
    .context("failed to initialize OCRS engine")?;

    Ok(OcrsRtenModel { engine })
}

impl OcrsRtenModel {
    pub(crate) fn extract_bytes(&self, bytes: &[u8]) -> Result<String> {
        let image = image::load_from_memory(bytes)
            .context("failed to decode OCR input image")?
            .into_rgb8();
        self.extract_rgb_image(image)
    }

    pub(crate) fn extract_image(&self, image: &DynamicImage) -> Result<String> {
        self.extract_rgb_image(image.to_rgb8())
    }

    fn extract_rgb_image(&self, image: image::RgbImage) -> Result<String> {
        let source =
            ImageSource::from_bytes(image.as_raw(), image.dimensions())
                .context("failed to create OCRS image source")?;
        let input = self
            .engine
            .prepare_input(source)
            .context("failed to prepare OCRS input")?;

        let words = self
            .engine
            .detect_words(&input)
            .context("failed to run OCRS text detection")?;
        let lines = self.engine.find_text_lines(&input, &words);
        let text_lines = self
            .engine
            .recognize_text(&input, &lines)
            .context("failed to run OCRS text recognition")?;
        let text = join_useful_lines(
            text_lines
                .into_iter()
                .filter_map(|line| line.map(|line| line.to_string())),
        );

        Ok(text)
    }
}
