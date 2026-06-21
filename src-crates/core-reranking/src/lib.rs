//! Simple text reranking models built with Burn.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core_reranking::TextReranker;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model = TextReranker::try_new().await?;
//!     let score = model.score("Rust ML", "Burn is a Rust ML framework")?;
//!     assert!(score.is_finite());
//!     Ok(())
//! }
//! ```

mod jina_v2;
mod xlm_roberta;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn::tensor::{Tensor, backend::Backend};
use burn_wgpu::{Wgpu, WgpuDevice};

use crate::jina_v2::{
    JinaV2RerankerModel, JinaV2RerankerVariant,
    load_pretrained_jina_v2_reranker,
};
use crate::xlm_roberta::{
    XlmRobertaRerankerModel, XlmRobertaRerankerVariant,
    load_pretrained_xlm_roberta_reranker,
};

/// Default Burn backend used by `akuna-core-reranking`.
pub(crate) type DefaultBackend = Wgpu;
/// Default device for the [`DefaultBackend`].
const DEFAULT_BATCH_SIZE: usize = 32;

/// Identifiers for the supported reranker models.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RerankerModel {
    /// BAAI/bge-reranker-base cross-encoder.
    #[default]
    BgeRerankerBase,
    /// JinaAI jina-reranker-v2-base-multilingual.
    JinaRerankerV2BaseMultilingual,
}

#[derive(Debug)]
enum LoadedRerankerModel<B: Backend> {
    XlmRoberta(XlmRobertaRerankerModel<B>),
    JinaV2(JinaV2RerankerModel<B>),
}

/// Construction options for a [`TextReranker`].
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct TextRerankerOptions {
    /// Which reranker model to load.
    pub model: RerankerModel,
    /// Optional Hugging Face cache directory override.
    pub cache_dir: Option<PathBuf>,
}

/// A single document ranked against a query.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResult {
    /// Original index of the document in the input slice.
    pub index: usize,
    /// Relevance score for the document.
    pub score: f32,
    /// Document text the score applies to.
    pub document: String,
}

/// Tunable behaviour for a single [`TextReranker::rerank_with_options`] call.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RerankOptions {
    /// Keep only the top `n` results when set.
    pub top_k: Option<usize>,
    /// Apply sigmoid normalization to scores.
    pub normalize: bool,
    /// Override the default inference batch size.
    pub batch_size: Option<usize>,
}

/// Cross-encoder text reranker backed by Burn.
#[derive(Debug)]
pub struct TextReranker<B: Backend = DefaultBackend> {
    model: LoadedRerankerModel<B>,
    device: B::Device,
}

impl TextReranker<DefaultBackend> {
    /// Loads the default reranker model with default options.
    pub async fn try_new() -> Result<Self> {
        Self::new(Default::default()).await
    }

    /// Loads a reranker on the default device using `options`.
    pub async fn new(options: TextRerankerOptions) -> Result<Self> {
        let device = WgpuDevice::default();
        Self::new_with_device(&device, options).await
    }
}

impl<B> TextReranker<B>
where
    B: Backend,
{
    /// Loads a reranker on a specific device using `options`.
    pub async fn new_with_device(
        device: &B::Device,
        options: TextRerankerOptions,
    ) -> Result<Self> {
        let model = match options.model {
            RerankerModel::BgeRerankerBase => LoadedRerankerModel::XlmRoberta(
                load_pretrained_xlm_roberta_reranker(
                    device,
                    XlmRobertaRerankerVariant::BgeRerankerBase,
                    options.cache_dir,
                )
                .await?,
            ),
            RerankerModel::JinaRerankerV2BaseMultilingual => {
                LoadedRerankerModel::JinaV2(
                    load_pretrained_jina_v2_reranker(
                        device,
                        JinaV2RerankerVariant::BaseMultilingual,
                        options.cache_dir,
                    )
                    .await?,
                )
            }
        };

        Ok(Self {
            model,
            device: device.clone(),
        })
    }

    /// Scores a single query/document pair.
    pub fn score(
        &self,
        query: impl AsRef<str>,
        document: impl AsRef<str>,
    ) -> Result<f32> {
        let mut scores =
            self.score_batch(&[(query.as_ref(), document.as_ref())], None)?;
        scores
            .pop()
            .context("expected one score for a single input pair")
    }

    /// Scores many query/document pairs in batches.
    pub fn score_batch<Q, D>(
        &self,
        pairs: &[(Q, D)],
        batch_size: Option<usize>,
    ) -> Result<Vec<f32>>
    where
        Q: AsRef<str>,
        D: AsRef<str>,
    {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = batch_size_or_default(pairs.len(), batch_size)?;
        let mut scores = Vec::with_capacity(pairs.len());

        for batch in pairs.chunks(batch_size) {
            let batch_pairs = batch
                .iter()
                .map(|(query, document)| (query.as_ref(), document.as_ref()))
                .collect::<Vec<_>>();
            let batch_scores = match &self.model {
                LoadedRerankerModel::XlmRoberta(model) => {
                    model.score(&batch_pairs, &self.device)?
                }
                LoadedRerankerModel::JinaV2(model) => {
                    model.score(&batch_pairs, &self.device)?
                }
            };
            scores.extend(tensor_to_vec(batch_scores)?);
        }

        Ok(scores)
    }

    /// Ranks documents against a query using default options.
    pub fn rerank<S: AsRef<str>>(
        &self,
        query: impl AsRef<str>,
        documents: &[S],
    ) -> Result<Vec<RerankResult>> {
        self.rerank_with_options(query, documents, Default::default())
    }

    /// Ranks documents against a query with custom options.
    pub fn rerank_with_options<S: AsRef<str>>(
        &self,
        query: impl AsRef<str>,
        documents: &[S],
        options: RerankOptions,
    ) -> Result<Vec<RerankResult>> {
        validate_top_k(options.top_k)?;
        let query = query.as_ref();
        let document_refs =
            documents.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        let pairs = document_refs
            .iter()
            .map(|document| (query, *document))
            .collect::<Vec<_>>();
        let scores = self.score_batch(&pairs, options.batch_size)?;
        let mut indexed_scores = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| {
                let score = if options.normalize {
                    sigmoid(score)
                } else {
                    score
                };
                (index, score)
            })
            .collect::<Vec<_>>();

        indexed_scores.sort_by(|left, right| right.1.total_cmp(&left.1));
        if let Some(top_k) = options.top_k {
            indexed_scores.truncate(top_k);
        }

        Ok(indexed_scores
            .into_iter()
            .map(|(index, score)| RerankResult {
                index,
                score,
                document: document_refs[index].to_string(),
            })
            .collect())
    }

    /// Returns the model variant backing this reranker.
    pub fn model(&self) -> RerankerModel {
        match &self.model {
            LoadedRerankerModel::XlmRoberta(model) => match model.variant {
                XlmRobertaRerankerVariant::BgeRerankerBase => {
                    RerankerModel::BgeRerankerBase
                }
            },
            LoadedRerankerModel::JinaV2(model) => match model.variant {
                JinaV2RerankerVariant::BaseMultilingual => {
                    RerankerModel::JinaRerankerV2BaseMultilingual
                }
            },
        }
    }
}

fn batch_size_or_default(
    item_count: usize,
    batch_size: Option<usize>,
) -> Result<usize> {
    let batch_size = batch_size.unwrap_or(item_count.min(DEFAULT_BATCH_SIZE));
    if batch_size == 0 {
        bail!("batch size must be greater than zero");
    }

    Ok(batch_size)
}

fn validate_top_k(top_k: Option<usize>) -> Result<()> {
    if matches!(top_k, Some(0)) {
        bail!("top_k must be greater than zero");
    }
    Ok(())
}

fn tensor_to_vec<B: Backend>(scores: Tensor<B, 1>) -> Result<Vec<f32>> {
    let data = scores.into_data().convert::<f32>();
    data.as_slice::<f32>()
        .map(|values| values.to_vec())
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to read reranker output tensor")
}

fn sigmoid(score: f32) -> f32 {
    1.0 / (1.0 + (-score).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn api_options_default_uses_bge_reranker_base() {
        assert_eq!(
            TextRerankerOptions::default().model,
            RerankerModel::BgeRerankerBase
        );
    }

    #[test]
    fn util_batch_size_validate_rejects_zero() {
        let error = batch_size_or_default(1, Some(0))
            .expect_err("zero batch size should fail");
        assert!(
            error
                .to_string()
                .contains("batch size must be greater than zero")
        );
    }

    #[test]
    fn util_top_k_validate_rejects_zero() {
        let error =
            validate_top_k(Some(0)).expect_err("zero top_k should fail");
        assert!(
            error
                .to_string()
                .contains("top_k must be greater than zero")
        );
    }

    #[test]
    fn util_top_k_validate_accepts_none_and_positive() {
        validate_top_k(None).expect("None top_k should pass");
        validate_top_k(Some(1)).expect("positive top_k should pass");
    }

    #[test]
    fn util_sigmoid_maps_scores_to_zero_one() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn util_sigmoid_bounded_for_extreme_scores() {
        assert!(sigmoid(1000.0).is_finite());
        assert!(sigmoid(-1000.0).is_finite());
        assert!(sigmoid(1000.0) <= 1.0);
        assert!(sigmoid(-1000.0) >= 0.0);
    }

    #[tokio::test]
    #[ignore = "downloads model and runs Python transformers reference"]
    async fn parity_bge_base_scores_match_flag_embedding() {
        let pairs = vec![
            (
                "Rust machine learning".to_string(),
                "Burn is a deep learning framework for Rust".to_string(),
            ),
            (
                "Rust machine learning".to_string(),
                "Bananas are yellow".to_string(),
            ),
        ];
        let model = TextReranker::new(TextRerankerOptions {
            model: RerankerModel::BgeRerankerBase,
            ..Default::default()
        })
        .await
        .expect("model should load");
        let actual = model
            .score_batch(&pairs, Some(2))
            .expect("Burn reranker should score pairs");
        let expected = reference_scores("BAAI/bge-reranker-base", &pairs)
            .expect("reference scores should compute");

        assert_scores_close(&actual, &expected, 1e-3);
    }

    #[tokio::test]
    #[ignore = "downloads model and runs Python transformers reference"]
    async fn parity_jina_v2_scores_match_transformers() {
        let pairs = vec![
            (
                "Rust machine learning".to_string(),
                "Burn is a deep learning framework for Rust".to_string(),
            ),
            (
                "Organic produce".to_string(),
                "Bananas are yellow".to_string(),
            ),
            (
                "Machine learning".to_string(),
                "Apprendimento automatico in Rust".to_string(),
            ),
        ];
        let model = TextReranker::new(TextRerankerOptions {
            model: RerankerModel::JinaRerankerV2BaseMultilingual,
            ..Default::default()
        })
        .await
        .expect("model should load");
        let actual = model
            .score_batch(&pairs, Some(2))
            .expect("Burn reranker should score pairs");
        let expected = reference_scores(
            "jinaai/jina-reranker-v2-base-multilingual",
            &pairs,
        )
        .expect("reference scores should compute");

        assert_scores_close(&actual, &expected, 1e-3);
    }

    fn reference_scores(
        model: &str,
        pairs: &[(String, String)],
    ) -> Result<Vec<f32>> {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new("uv")
            .args(["run", "scripts/reference_rerank.py", "--model", model])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to spawn uv reference rerank script")?;
        let mut stdin = child
            .stdin
            .take()
            .context("failed to open reference script stdin")?;
        let input = serde_json::to_vec(pairs)
            .context("failed to serialize reference input")?;
        stdin
            .write_all(&input)
            .context("failed to write reference input")?;
        drop(stdin);

        let output = child
            .wait_with_output()
            .context("failed to wait for reference script")?;
        if !output.status.success() {
            bail!(
                "reference script failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        serde_json::from_slice(&output.stdout)
            .context("failed to parse reference rerank scores")
    }

    fn assert_scores_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "score delta {delta} exceeded tolerance {tolerance}: actual {actual}, expected {expected}"
            );
        }
    }
}
