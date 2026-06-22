use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context, Result};
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm,
    LayerNormConfig, Linear, LinearConfig,
};
use burn::tensor::{DType, Int, Tensor, activation, backend::Backend};
use burn_store::{
    KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter,
    SafetensorsStore, TensorSnapshot,
};
use serde::Deserialize;
use tokenizers::{EncodeInput, Tokenizer, TruncationParams};

use crate::reranking::xlm_roberta::{
    SequenceClassificationHead, download_hf_model,
};

const JINA_RERANKER_V2_BASE_MULTILINGUAL_REPO_ID: &str =
    "jinaai/jina-reranker-v2-base-multilingual";
type TokenizedPairs<B> = (Tensor<B, 2, Int>, Tensor<B, 2>, Tensor<B, 2, Int>);

/// Available Jina reranker v2 model variants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum JinaV2RerankerVariant {
    /// `jinaai/jina-reranker-v2-base-multilingual`.
    #[default]
    BaseMultilingual,
}

impl JinaV2RerankerVariant {
    /// Returns the Hugging Face repository id for this variant.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::BaseMultilingual => {
                JINA_RERANKER_V2_BASE_MULTILINGUAL_REPO_ID
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct JinaV2Config {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    layer_norm_eps: f64,
}

#[derive(Module, Debug)]
struct JinaV2Embeddings<B: Backend> {
    word_embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Embedding<B>,
}

#[derive(Module, Debug)]
struct JinaV2Mixer<B: Backend> {
    wqkv: Linear<B>,
    out_proj: Linear<B>,
    num_heads: usize,
    head_dim: usize,
}

#[derive(Module, Debug)]
struct JinaV2Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

#[derive(Module, Debug)]
struct JinaV2Layer<B: Backend> {
    mixer: JinaV2Mixer<B>,
    mlp: JinaV2Mlp<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

#[derive(Module, Debug)]
struct JinaV2Model<B: Backend> {
    embeddings: JinaV2Embeddings<B>,
    emb_ln: LayerNorm<B>,
    layers: Vec<JinaV2Layer<B>>,
    dropout: Dropout,
    num_heads: usize,
    head_dim: usize,
}

#[derive(Module, Debug)]
struct JinaV2ForSequenceClassification<B: Backend> {
    roberta: JinaV2Model<B>,
    classifier: SequenceClassificationHead<B>,
}

/// Loaded Jina reranker v2 model ready for inference.
#[derive(Debug)]
pub(crate) struct JinaV2RerankerModel<B: Backend> {
    model: JinaV2ForSequenceClassification<B>,
    tokenizer: Tokenizer,
    pub(crate) variant: JinaV2RerankerVariant,
}

impl JinaV2Config {
    fn load_from_hf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).with_context(|| {
            format!("failed to read reranker config at {}", path.display())
        })?;

        serde_json::from_str(&content).with_context(|| {
            format!("failed to parse reranker config at {}", path.display())
        })
    }

    fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> JinaV2ForSequenceClassification<B> {
        let head_dim = self.hidden_size / self.num_attention_heads;
        JinaV2ForSequenceClassification {
            roberta: JinaV2Model {
                embeddings: JinaV2Embeddings::new(self, device),
                emb_ln: LayerNormConfig::new(self.hidden_size)
                    .with_epsilon(self.layer_norm_eps)
                    .init(device),
                layers: (0..self.num_hidden_layers)
                    .map(|_| JinaV2Layer::new(self, device))
                    .collect::<Vec<_>>(),
                dropout: DropoutConfig::new(0.0).init(),
                num_heads: self.num_attention_heads,
                head_dim,
            },
            classifier: SequenceClassificationHead {
                dense: LinearConfig::new(self.hidden_size, self.hidden_size)
                    .init(device),
                out_proj: LinearConfig::new(self.hidden_size, 1).init(device),
            },
        }
    }
}

impl<B: Backend> JinaV2Embeddings<B> {
    fn new(config: &JinaV2Config, device: &B::Device) -> Self {
        Self {
            word_embeddings: EmbeddingConfig::new(
                config.vocab_size,
                config.hidden_size,
            )
            .init(device),
            position_embeddings: EmbeddingConfig::new(
                config.max_position_embeddings,
                config.hidden_size,
            )
            .init(device),
            token_type_embeddings: EmbeddingConfig::new(
                config.type_vocab_size,
                config.hidden_size,
            )
            .init(device),
        }
    }

    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();
        let word_embeddings = self.word_embeddings.forward(input_ids);
        let position_ids =
            Tensor::<B, 1, Int>::arange(2..(seq_len as i64 + 2), &device)
                .reshape([1, seq_len])
                .expand([batch_size, seq_len]);
        let position_embeddings =
            self.position_embeddings.forward(position_ids);
        let token_type_ids = token_type_ids.unwrap_or_else(|| {
            Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &device)
        });
        let token_type_embeddings =
            self.token_type_embeddings.forward(token_type_ids);

        word_embeddings + position_embeddings + token_type_embeddings
    }
}

impl<B: Backend> JinaV2Mixer<B> {
    fn new(config: &JinaV2Config, device: &B::Device) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        Self {
            wqkv: LinearConfig::new(config.hidden_size, 3 * config.hidden_size)
                .init(device),
            out_proj: LinearConfig::new(config.hidden_size, config.hidden_size)
                .init(device),
            num_heads: config.num_attention_heads,
            head_dim,
        }
    }

    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_bias: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let qkv = self.wqkv.forward(hidden_states).reshape([
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        ]);
        let q = qkv
            .clone()
            .narrow(2, 0, 1)
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = qkv
            .clone()
            .narrow(2, 1, 1)
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = qkv
            .narrow(2, 2, 1)
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let attention_scores = q.matmul(k.swap_dims(2, 3))
            / (self.head_dim as f64).sqrt()
            + attention_bias;
        let attention_probs = activation::softmax(attention_scores, 3);
        let context = attention_probs.matmul(v).swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            hidden_size,
        ]);

        self.out_proj.forward(context)
    }
}

impl<B: Backend> JinaV2Mlp<B> {
    fn new(config: &JinaV2Config, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(
                config.hidden_size,
                config.intermediate_size,
            )
            .init(device),
            fc2: LinearConfig::new(
                config.intermediate_size,
                config.hidden_size,
            )
            .init(device),
        }
    }

    fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2
            .forward(activation::gelu(self.fc1.forward(hidden_states)))
    }
}

impl<B: Backend> JinaV2Layer<B> {
    fn new(config: &JinaV2Config, device: &B::Device) -> Self {
        Self {
            mixer: JinaV2Mixer::new(config, device),
            mlp: JinaV2Mlp::new(config, device),
            norm1: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            norm2: LayerNormConfig::new(config.hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            dropout: DropoutConfig::new(0.0).init(),
        }
    }

    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_bias: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let mixer_out =
            self.mixer.forward(hidden_states.clone(), attention_bias);
        let hidden_states = self
            .norm1
            .forward(self.dropout.forward(mixer_out) + hidden_states);
        let mlp_out = self.mlp.forward(hidden_states.clone());

        self.norm2
            .forward(self.dropout.forward(mlp_out) + hidden_states)
    }
}

impl<B: Backend> JinaV2Model<B> {
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len] = attention_mask.dims();
        let embeddings = self.embeddings.forward(input_ids, token_type_ids);
        let hidden_states =
            self.dropout.forward(self.emb_ln.forward(embeddings));
        let attention_bias = attention_mask_to_bias::<B>(
            attention_mask,
            batch_size,
            seq_len,
            self.num_heads,
        );

        self.layers
            .iter()
            .fold(hidden_states, |hidden_states, layer| {
                layer.forward(hidden_states, attention_bias.clone())
            })
    }
}

impl<B: Backend> JinaV2ForSequenceClassification<B> {
    fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2>,
        token_type_ids: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 1> {
        let hidden_states =
            self.roberta
                .forward(input_ids, attention_mask, token_type_ids);
        let [batch_size, _, hidden_size] = hidden_states.dims();
        let pooled = hidden_states
            .narrow(1, 0, 1)
            .reshape([batch_size, hidden_size]);
        let logits = self.classifier.forward(pooled);
        let [batch_size, _] = logits.dims();

        logits.reshape([batch_size])
    }
}

impl<B: Backend> JinaV2RerankerModel<B> {
    /// Tokenizes `pairs` and runs a forward pass returning a 1-D logits tensor with one score per pair.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub(crate) fn score(
        &self,
        pairs: &[(&str, &str)],
        device: &B::Device,
    ) -> Result<Tensor<B, 1>> {
        let (input_ids, attention_mask, token_type_ids) =
            tokenize_pairs(&self.tokenizer, pairs, device)?;

        Ok(self
            .model
            .forward(input_ids, attention_mask, Some(token_type_ids)))
    }
}

/// Downloads, parses, and loads a Jina reranker v2 model onto `device`.
///
/// # Errors
///
/// Returns an error if the model download, config parse, weight load, or tokenizer setup fails.
pub(crate) async fn load_pretrained_jina_v2_reranker<B>(
    device: &B::Device,
    variant: JinaV2RerankerVariant,
    cache_dir: Option<PathBuf>,
) -> Result<JinaV2RerankerModel<B>>
where
    B: Backend,
{
    let files = download_hf_model(variant.repo_id(), cache_dir).await?;
    let config = JinaV2Config::load_from_hf(&files.config_path)?;
    let mut model = config.init(device);
    load_pretrained_weights(&mut model, &files.weights_path)?;
    let mut tokenizer = Tokenizer::from_file(&files.tokenizer_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| {
            format!(
                "failed to load reranker tokenizer from {}",
                files.tokenizer_path.display()
            )
        })?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: config.max_position_embeddings - 2,
            ..Default::default()
        }))
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to configure tokenizer truncation")?;

    Ok(JinaV2RerankerModel {
        model,
        tokenizer,
        variant,
    })
}

fn load_pretrained_weights<B: Backend>(
    model: &mut JinaV2ForSequenceClassification<B>,
    checkpoint_path: impl AsRef<Path>,
) -> Result<()> {
    let key_mappings = vec![
        (
            "^roberta\\.encoder\\.layers\\.([0-9]+)\\.mixer\\.Wqkv\\.",
            "roberta.layers.$1.mixer.wqkv.",
        ),
        (
            "^roberta\\.encoder\\.layers\\.([0-9]+)\\.mixer\\.out_proj\\.",
            "roberta.layers.$1.mixer.out_proj.",
        ),
        (
            "^roberta\\.encoder\\.layers\\.([0-9]+)\\.mlp\\.fc1\\.",
            "roberta.layers.$1.mlp.fc1.",
        ),
        (
            "^roberta\\.encoder\\.layers\\.([0-9]+)\\.mlp\\.fc2\\.",
            "roberta.layers.$1.mlp.fc2.",
        ),
        (
            "^roberta\\.encoder\\.layers\\.([0-9]+)\\.norm1\\.",
            "roberta.layers.$1.norm1.",
        ),
        (
            "^roberta\\.encoder\\.layers\\.([0-9]+)\\.norm2\\.",
            "roberta.layers.$1.norm2.",
        ),
    ];
    let remapper = KeyRemapper::from_patterns(key_mappings)
        .context("failed to create reranker weight remapper")?;
    let adapter = PyTorchToBurnAdapter.chain(Bfloat16ToFloat32Adapter);
    let mut store = SafetensorsStore::from_file(checkpoint_path.as_ref())
        .with_from_adapter(adapter)
        .remap(remapper);

    model.load_from(&mut store).with_context(|| {
        format!(
            "failed to load reranker weights from {}",
            checkpoint_path.as_ref().display()
        )
    })?;

    Ok(())
}

#[derive(Debug, Clone, Default)]
struct Bfloat16ToFloat32Adapter;

impl ModuleAdapter for Bfloat16ToFloat32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype != DType::BF16 {
            return snapshot.clone();
        }
        let original_data_fn = snapshot.clone_data_fn();
        let cast_data_fn = Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(DType::F32))
        });

        TensorSnapshot::from_closure(
            cast_data_fn,
            DType::F32,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

fn tokenize_pairs<B: Backend>(
    tokenizer: &Tokenizer,
    pairs: &[(&str, &str)],
    device: &B::Device,
) -> Result<TokenizedPairs<B>> {
    let inputs = pairs
        .iter()
        .map(|(query, document)| {
            EncodeInput::Dual((*query).into(), (*document).into())
        })
        .collect::<Vec<_>>();
    let encodings = tokenizer
        .encode_batch(inputs, true)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to tokenize reranker input batch")?;
    let max_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(1);
    let batch_size = pairs.len();
    let mut input_ids = vec![0i32; batch_size * max_len];
    let mut attention_mask = vec![0.0f32; batch_size * max_len];
    let mut token_type_ids = vec![0i32; batch_size * max_len];

    for (batch_index, encoding) in encodings.iter().enumerate() {
        for (token_index, token_id) in encoding.get_ids().iter().enumerate() {
            let position = batch_index * max_len + token_index;
            input_ids[position] = *token_id as i32;
            attention_mask[position] =
                encoding.get_attention_mask()[token_index] as f32;
            token_type_ids[position] =
                encoding.get_type_ids()[token_index] as i32;
        }
    }

    Ok((
        Tensor::<B, 1, Int>::from_ints(input_ids.as_slice(), device)
            .reshape([batch_size, max_len]),
        Tensor::<B, 1>::from_floats(attention_mask.as_slice(), device)
            .reshape([batch_size, max_len]),
        Tensor::<B, 1, Int>::from_ints(token_type_ids.as_slice(), device)
            .reshape([batch_size, max_len]),
    ))
}

fn attention_mask_to_bias<B: Backend>(
    attention_mask: Tensor<B, 2>,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
) -> Tensor<B, 4> {
    let device = attention_mask.device();
    let mask = attention_mask
        .reshape([batch_size, 1, 1, seq_len])
        .expand([batch_size, num_heads, seq_len, seq_len]);
    let ones = Tensor::<B, 4>::ones(mask.dims(), &device);

    (ones - mask) * -10000.0
}
