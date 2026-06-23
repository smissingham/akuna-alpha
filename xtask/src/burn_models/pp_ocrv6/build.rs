//! PP-OCR Burn model generation.
//!
//! Central orchestration for all PP-OCR artifacts used by the `xtask` pipeline.
//! It converts upstream ONNX checkpoints to BurnPack files and writes both
//! generated code and runtime manifests into shared model output locations.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;

use crate::burn_models::pp_ocrv6::{
    Paths, Spec, LICENSE_URL, MODELS, REPO_NAME, UPSTREAM_PROJECT_URL,
};
use crate::burn_models::HF_BASE_URL;
use anyhow::{bail, Context, Result};
use burn_onnx::ModelGen;
use onnx_ir::ModelProto;
use protobuf::Message;

/// Runs the full PP-OCR codegen pipeline.
///
/// The pipeline downloads ONNX checkpoints, patches input tensor shapes to match
/// fixed Burn expectations, runs `burn-onnx` generation, then publishes final
/// artifacts under `target/models` via the shared model writer.
pub(crate) fn run() -> Result<()> {
    let repo = crate::repo_root()?;
    let assets_dir = repo.join("target/models").join(REPO_NAME);
    let paths = Paths {
        generated: repo.join("src-crates/core/src/ocr/models/pp_ocr/generated"),
        onnx: std::env::temp_dir().join("akuna-pp-ocr-codegen/onnx"),
        patched: std::env::temp_dir().join("akuna-pp-ocr-codegen/patched"),
        burn: std::env::temp_dir().join("akuna-pp-ocr-codegen/burn"),
    };

    recreate_dir(&assets_dir)?;
    prepare_dirs(&paths)?;

    for model in MODELS.iter() {
        download_model(model, &paths.onnx)?;
        patch_model(model, &paths.onnx, &paths.patched)?;
        generate_model(model, &paths)?;
    }

    write_catalog()?;

    Ok(())
}

/// Prepares generated output and temporary directories.
///
/// Wipes and recreates all scratch paths to guarantee deterministic, reproducible
/// generation across runs.
fn prepare_dirs(paths: &Paths) -> Result<()> {
    recreate_dir(&paths.onnx)?;
    recreate_dir(&paths.patched)?;
    recreate_dir(&paths.burn)?;

    Ok(())
}

/// Recreates a directory from scratch.
///
/// Removes the full path when present and recreates it. Useful for temp and
/// generated trees that must not carry stale files.
fn recreate_dir(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_dir_all(path)
            .with_context(|| format!("failed to remove {}", path.display()))?;
    }
    fs::create_dir_all(path)
        .with_context(|| format!("failed to create {}", path.display()))
}

/// Sanitizes generated Rust for clippy compliance.
///
/// burn-onnx emits `.unwrap()` calls on `try_into()` for static shape arrays
/// that are guaranteed correct by construction, plus redundant same-type
/// casts (`x as i64` where `x` is already `i64`). Adds crate-level allows so
/// the checked-in generated sources pass strict clippy.
fn postprocess_generated(source: &str, _rust_file: &str) -> String {
    let allow = "#![allow(clippy::unwrap_used, clippy::unnecessary_cast)]\n";
    if source.starts_with(allow) {
        return source.to_string();
    }
    format!("{allow}{source}")
}

/// Downloads source ONNX for one PP-OCR model.
///
/// Uses the model metadata to fetch `inference.onnx` from Hugging Face and
/// stores it under the per-run ONNX scratch directory.
fn download_model(model: &Spec, onnx_dir: &Path) -> Result<()> {
    let url = format!(
        "{HF_BASE_URL}{}/resolve/{}/inference.onnx",
        model.repo, model.revision
    );
    let output = onnx_dir.join(format!("{}.onnx", model.name));
    let response = ureq::get(&url)
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    let mut body = response.into_body().into_reader();
    let mut file = fs::File::create(&output)
        .with_context(|| format!("failed to create {}", output.display()))?;
    io::copy(&mut body, &mut file)
        .with_context(|| format!("failed to write {}", output.display()))?;

    Ok(())
}

/// Patches dynamic ONNX input shape to Burn-supported static shape.
///
/// Rewrites the first graph input dimensions to the configured static size so
/// generated code can target fixed tensor dimensions.
fn patch_model(
    model: &Spec,
    onnx_dir: &Path,
    patched_dir: &Path,
) -> Result<()> {
    let input = onnx_dir.join(format!("{}.onnx", model.name));
    let output = patched_dir.join(format!("{}.onnx", model.name));
    let bytes = fs::read(&input)
        .with_context(|| format!("failed to read {}", input.display()))?;
    let mut proto = ModelProto::parse_from_bytes(&bytes)
        .with_context(|| format!("failed to parse {}", input.display()))?;
    let graph = proto
        .graph
        .as_mut()
        .context("PP-OCR ONNX model graph missing")?;
    let graph_input = graph
        .input
        .first_mut()
        .context("PP-OCR ONNX model input missing")?;
    let value_type = graph_input
        .type_
        .as_mut()
        .context("PP-OCR ONNX model input type missing")?;
    let tensor_type = value_type.mut_tensor_type();
    let shape = tensor_type
        .shape
        .as_mut()
        .context("PP-OCR ONNX model input shape missing")?;

    if shape.dim.len() != model.input_shape.len() {
        bail!(
            "PP-OCR ONNX input rank for {} is {}, expected {}",
            model.name,
            shape.dim.len(),
            model.input_shape.len()
        );
    }

    for (dim, value) in shape.dim.iter_mut().zip(model.input_shape) {
        dim.set_dim_value(value);
    }

    fs::write(&output, proto.write_to_bytes()?)
        .with_context(|| format!("failed to write {}", output.display()))?;

    Ok(())
}

/// Generates Burn Rust source and BurnPack weights.
///
/// Invokes `burn-onnx` for one model, copies checked-in generated Rust output,
/// and pushes `.bpk` + manifest through the shared tracing writer.
fn generate_model(model: &Spec, paths: &Paths) -> Result<()> {
    let input = paths.patched.join(format!("{}.onnx", model.name));
    let output_dir = paths.burn.join(model.name);

    ModelGen::new()
        .input(input.to_str().context("non-utf8 ONNX path")?)
        .out_dir(output_dir.to_str().context("non-utf8 output path")?)
        .development(false)
        .run_from_cli();

    let generated_dir = paths.generated.join(model.generated_subdir);
    fs::create_dir_all(&generated_dir).with_context(|| {
        format!("failed to create {}", generated_dir.display())
    })?;
    let generated_source =
        fs::read_to_string(output_dir.join(format!("{}.rs", model.name)))
            .with_context(|| {
                format!("failed to read generated Rust for {}", model.name)
            })?;
    let generated_source =
        postprocess_generated(&generated_source, model.rust_file);
    fs::write(generated_dir.join(model.rust_file), generated_source)
        .with_context(|| {
            format!("failed to write generated Rust for {}", model.name)
        })?;

    let parent_folder = format!("{REPO_NAME}/{}", model.tier);
    let bpk_source = output_dir.join(format!("{}.bpk", model.name));
    let bpk_bytes = fs::read(&bpk_source).with_context(|| {
        format!("failed to read generated BurnPack for {}", model.name)
    })?;
    crate::burn_models::write_file(&parent_folder, model.bpk_file, bpk_bytes)?;
    write_manifest(model, &parent_folder)?;

    Ok(())
}

/// Writes model asset provenance for one generated BurnPack.
///
/// Output is a JSON manifest with source, license, revision, and conversion
/// metadata for every generated model.
fn write_manifest(model: &Spec, parent_folder: &str) -> Result<()> {
    let manifest = serde_json::json!({
        "category": model.category,
        "family": REPO_NAME,
        "tier": model.tier,
        "name": model.name,
        "architecture": "burn-onnx-generated",
        "source_repo": model.repo,
        "source_url": format!("{HF_BASE_URL}{}", model.repo),
        "source_revision": model.revision,
        "source_license": "apache-2.0",
        "source_license_url": LICENSE_URL,
        "conversion": {
            "format": "burnpack",
            "weights": "converted from upstream ONNX weights",
            "architecture": "generated by burn-onnx"
        },
        "weights": model.bpk_file,
        "input_shape": model.input_shape,
    });
    let manifest = serde_json::to_string_pretty(&manifest)?;
    let manifest_path = format!("{}.manifest.json", model.name);
    crate::burn_models::write_file(parent_folder, manifest_path, manifest)
}

/// Writes PP-OCR repo catalog files.
///
/// Persists `model-index.json` plus a README that credits the upstream
/// PaddlePaddle sources, links to each HF repo, and surfaces the inherited
/// Apache 2.0 license.
fn write_catalog() -> Result<()> {
    let upstream_repos: Vec<&str> = MODELS
        .iter()
        .map(|model| model.repo)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let upstream_list = upstream_repos
        .iter()
        .map(|repo| format!("- [{repo}]({HF_BASE_URL}{repo})\n"))
        .collect::<String>();

    let catalog = serde_json::json!({
        "layout": {
            "<tier>/<file>.bpk": "PP-OCRv6 detector and recognizer BurnPack weights"
        },
        "models": upstream_repos,
    });
    crate::burn_models::write_file(
        REPO_NAME,
        "model-index.json",
        serde_json::to_string_pretty(&catalog)?,
    )?;

    let base_model_yaml = upstream_repos
        .iter()
        .map(|repo| format!("  - {repo}\n"))
        .collect::<String>();

    let readme = format!(
        "---\n\
license: apache-2.0\n\
library_name: burn\n\
base_model:\n\
{base_model_yaml}\
---\n\
# PP-OCRv6 BurnPack Weights\n\
\n\
Burn-converted weights for [PaddlePaddle PP-OCRv6]({UPSTREAM_PROJECT_URL}).\n\
\n\
## Sources\n\
\n\
{upstream_list}\
\n\
## License\n\
\n\
Apache License 2.0 — see <{LICENSE_URL}>.\n",
    );
    crate::burn_models::write_file(REPO_NAME, "README.md", readme)?;

    Ok(())
}
