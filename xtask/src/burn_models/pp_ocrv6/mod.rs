//! PP-OCRv6 maintenance tasks.

mod build;

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};

/// HF repo + local asset folder, matches upstream PaddlePaddle naming.
const REPO_NAME: &str = "PP-OCRv6-burn";
/// Upstream project home for context + licensing.
const UPSTREAM_PROJECT_URL: &str = "https://github.com/PaddlePaddle/PaddleOCR";
/// License URL inherited from upstream.
const LICENSE_URL: &str = "https://www.apache.org/licenses/LICENSE-2.0";

const MODELS: &[Spec] = &[
    Spec {
        name: "tiny_det",
        repo: "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
        revision: "2ba1506c0380b8f0b03dd142459aac66d4421f6c",
        category: "ocr",
        tier: "tiny",
        generated_subdir: "pp_ocrv6_tiny_det",
        rust_file: "tiny_det.rs",
        bpk_file: "det.bpk",
        input_shape: [1, 3, 960, 960],
    },
    Spec {
        name: "small_det",
        repo: "PaddlePaddle/PP-OCRv6_small_det_onnx",
        revision: "28fe5895c24fd108c19eb3e8479f4ab385fbfc62",
        category: "ocr",
        tier: "small",
        generated_subdir: "pp_ocrv6_small_det",
        rust_file: "small_det.rs",
        bpk_file: "det.bpk",
        input_shape: [1, 3, 960, 960],
    },
    Spec {
        name: "medium_det",
        repo: "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        revision: "61323801669c338b7891481ec7bac61ce31b576a",
        category: "ocr",
        tier: "medium",
        generated_subdir: "pp_ocrv6_medium_det",
        rust_file: "medium_det.rs",
        bpk_file: "det.bpk",
        input_shape: [1, 3, 960, 960],
    },
    Spec {
        name: "tiny_rec",
        repo: "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
        revision: "2612ab37152ae0a677521bae4e1e3d4fb4cf7c30",
        category: "ocr",
        tier: "tiny",
        generated_subdir: "pp_ocrv6_tiny_rec",
        rust_file: "tiny_rec.rs",
        bpk_file: "rec.bpk",
        input_shape: [1, 3, 48, 320],
    },
    Spec {
        name: "small_rec",
        repo: "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        revision: "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
        category: "ocr",
        tier: "small",
        generated_subdir: "pp_ocrv6_small_rec",
        rust_file: "small_rec.rs",
        bpk_file: "rec.bpk",
        input_shape: [1, 3, 48, 320],
    },
    Spec {
        name: "medium_rec",
        repo: "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        revision: "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
        category: "ocr",
        tier: "medium",
        generated_subdir: "pp_ocrv6_medium_rec",
        rust_file: "medium_rec.rs",
        bpk_file: "rec.bpk",
        input_shape: [1, 3, 48, 320],
    },
];

struct Spec {
    name: &'static str,
    repo: &'static str,
    revision: &'static str,
    category: &'static str,
    tier: &'static str,
    generated_subdir: &'static str,
    rust_file: &'static str,
    bpk_file: &'static str,
    input_shape: [i64; 4],
}

struct Paths {
    generated: PathBuf,
    onnx: PathBuf,
    patched: PathBuf,
    burn: PathBuf,
}

/// Builds PP-OCR generated Rust and BurnPack weights.
pub(crate) fn build() -> Result<()> {
    build::run()
}

/// Pushes PP-OCR assets under `target/models/PP-OCRv6` to its own HF repo.
///
/// Repo is derived as `<hf_owner>/PP-OCRv6`. Requires `HF_TOKEN` and
/// `AKUNA_BURN_MODELS_OWNER` env vars (validated via shared helpers).
pub(crate) fn push() -> Result<()> {
    let token = super::hf_token()?;
    let owner = super::hf_owner()?;
    let repo_id = format!("{owner}/{REPO_NAME}");
    let assets = crate::repo_root()?.join("target/models").join(REPO_NAME);
    let status = Command::new("hf")
        .args([
            "upload",
            &repo_id,
            assets.to_str().context("non-utf8 PP-OCR asset path")?,
            ".",
            "--repo-type",
            "model",
        ])
        .env("HF_TOKEN", token)
        .status()
        .context("failed to run huggingface-cli")?;

    if !status.success() {
        bail!("huggingface-cli upload failed with status {status}");
    }

    Ok(())
}
