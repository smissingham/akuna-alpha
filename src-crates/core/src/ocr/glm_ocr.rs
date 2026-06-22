use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use burn::tensor::{
    Int, Tensor, TensorData, activation::softmax, backend::Backend,
};
use half::bf16;
use hf_hub::{Repo, RepoType, api::tokio::ApiBuilder};
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::layout::pp_doclayout::{
    PpDocLayoutRuntime, load_pp_doclayout_runtime,
};
use crate::ocr::text_pipeline::{
    crop_text_region, text_like_detection, useful_ocr_fragment,
};

const GLM_OCR_REPO_ID: &str = "zai-org/GLM-OCR";
const GLM_OCR_REVISION: &str = "ca5d8b3e287e52589e37c28385d9655ee4372f9d";
#[allow(dead_code)]
const GLM_OCR_PROMPT_TEXT: &str = "Text Recognition:/nothink";
#[allow(dead_code)]
const GLM_OCR_PROMPT_PREFIX: &str = "[gMASK]<sop><|user|>\n<|begin_of_image|>";
#[allow(dead_code)]
const GLM_OCR_PROMPT_SUFFIX: &str =
    "<|end_of_image|>Text Recognition:/nothink<|assistant|>\n<think></think>\n";
#[allow(dead_code)]
const GLM_OCR_IMAGE_PLACEHOLDER: &str = "<|image|>";
#[allow(dead_code)]
const GLM_OCR_NATIVE_MAX_PIXELS: usize = 500_000;
#[allow(dead_code)]
const GLM_OCR_REPETITION_PENALTY: f32 = 1.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GlmOcrVisionDims {
    layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    heads: usize,
    temporal_patch_size: usize,
    channels: usize,
    patch_size: usize,
    merge_size: usize,
    output_hidden_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GlmOcrTextDims {
    layers: usize,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    attention_q_size: usize,
    attention_kv_size: usize,
}

const GLM_OCR_VISION_DIMS: GlmOcrVisionDims = GlmOcrVisionDims {
    layers: 24,
    hidden_size: 1024,
    intermediate_size: 4096,
    heads: 16,
    temporal_patch_size: 2,
    channels: 3,
    patch_size: 14,
    merge_size: 2,
    output_hidden_size: 1536,
};

const GLM_OCR_TEXT_DIMS: GlmOcrTextDims = GlmOcrTextDims {
    layers: 16,
    vocab_size: 59_392,
    hidden_size: 1_536,
    intermediate_size: 4_608,
    heads: 16,
    kv_heads: 8,
    head_dim: 128,
    attention_q_size: 2_048,
    attention_kv_size: 1_024,
};

const GLM_OCR_MROPE_SECTION: [usize; 3] = [16, 24, 24];

const GLM_VISION_PATCH_EMBED_WEIGHT: &str =
    "model.visual.patch_embed.proj.weight";
const GLM_VISION_PATCH_EMBED_BIAS: &str = "model.visual.patch_embed.proj.bias";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum GlmOcrVariant {
    #[default]
    OnnxCommunity,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct GlmOcrFiles {
    pub(crate) config_path: PathBuf,
    pub(crate) generation_config_path: PathBuf,
    pub(crate) preprocessor_config_path: PathBuf,
    pub(crate) tokenizer_path: PathBuf,
    pub(crate) weights_path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
struct GlmOcrConfig {
    image_start_token_id: u32,
    image_end_token_id: u32,
    image_token_id: u32,
    text_config: TextConfig,
    vision_config: VisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct TextConfig {
    eos_token_id: Vec<u32>,
    pad_token_id: u32,
    vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct VisionConfig {
    image_size: usize,
    patch_size: usize,
    spatial_merge_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct PreprocessorConfig {
    size: ImageSize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ImageSize {
    shortest_edge: usize,
    longest_edge: usize,
}

#[derive(Debug)]
pub(crate) struct GlmOcrModel<B: Backend> {
    variant: GlmOcrVariant,
    #[allow(dead_code)]
    config: GlmOcrConfig,
    #[allow(dead_code)]
    preprocessor: PreprocessorConfig,
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    #[allow(dead_code)]
    files: Option<GlmOcrFiles>,
    #[allow(dead_code)]
    vision: GlmVisionEncoder<B>,
    #[allow(dead_code)]
    text: GlmTextModel<B>,
    #[allow(dead_code)]
    layout: Option<PpDocLayoutRuntime<B>>,
    backend: PhantomData<B>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct GlmVisionPatchEmbed<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct GlmRmsNorm<B: Backend> {
    weight: Tensor<B, 1>,
    epsilon: f64,
}

#[derive(Debug)]
#[allow(dead_code)]
struct GlmVisionBlock<B: Backend> {
    norm1: GlmRmsNorm<B>,
    norm2: GlmRmsNorm<B>,
    q_norm: GlmRmsNorm<B>,
    k_norm: GlmRmsNorm<B>,
    qkv_weight: Tensor<B, 2>,
    qkv_bias: Tensor<B, 1>,
    proj_weight: Tensor<B, 2>,
    proj_bias: Tensor<B, 1>,
    gate_weight: Tensor<B, 2>,
    gate_bias: Tensor<B, 1>,
    up_weight: Tensor<B, 2>,
    up_bias: Tensor<B, 1>,
    down_weight: Tensor<B, 2>,
    down_bias: Tensor<B, 1>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct GlmVisionEncoder<B: Backend> {
    patch_embed: GlmVisionPatchEmbed<B>,
    blocks: Vec<GlmVisionBlock<B>>,
    post_norm: GlmRmsNorm<B>,
    merger: GlmVisionMerger<B>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct GlmVisionMerger<B: Backend> {
    downsample_weight: Tensor<B, 2>,
    downsample_bias: Tensor<B, 1>,
    proj_weight: Tensor<B, 2>,
    norm_weight: Tensor<B, 1>,
    norm_bias: Tensor<B, 1>,
    gate_weight: Tensor<B, 2>,
    up_weight: Tensor<B, 2>,
    down_weight: Tensor<B, 2>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct GlmTextDecoderLayer<B: Backend> {
    input_layernorm: GlmRmsNorm<B>,
    post_self_attn_layernorm: GlmRmsNorm<B>,
    post_attention_layernorm: GlmRmsNorm<B>,
    post_mlp_layernorm: GlmRmsNorm<B>,
    q_proj_weight: Tensor<B, 2>,
    k_proj_weight: Tensor<B, 2>,
    v_proj_weight: Tensor<B, 2>,
    o_proj_weight: Tensor<B, 2>,
    gate_up_proj_weight: Tensor<B, 2>,
    down_proj_weight: Tensor<B, 2>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GlmTextLayerCache<B: Backend> {
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
}

#[derive(Debug)]
#[allow(dead_code)]
struct GlmTextModel<B: Backend> {
    embed_tokens: Tensor<B, 2>,
    layers: Vec<GlmTextDecoderLayer<B>>,
    norm: GlmRmsNorm<B>,
    lm_head: Tensor<B, 2>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
struct GlmOcrPrompt {
    input_ids: Vec<u32>,
    mm_token_type_ids: Vec<u8>,
    image_token_range: std::ops::Range<usize>,
    position_ids: [Vec<i64>; 3],
}

impl GlmOcrVariant {
    fn repo_id(self) -> &'static str {
        match self {
            Self::OnnxCommunity => GLM_OCR_REPO_ID,
        }
    }

    fn revision(self) -> &'static str {
        match self {
            Self::OnnxCommunity => GLM_OCR_REVISION,
        }
    }
}

impl<B> GlmOcrModel<B>
where
    B: Backend<FloatElem = f32>,
{
    pub(crate) fn extract_bytes(
        &self,
        bytes: &[u8],
        device: &B::Device,
    ) -> Result<String> {
        let image = image::load_from_memory(bytes)
            .context("failed to decode OCR input image")?;
        self.extract_image(&image, device)
    }

    fn extract_image(
        &self,
        image: &DynamicImage,
        device: &B::Device,
    ) -> Result<String> {
        if let Some(layout) = &self.layout {
            let detections = match layout.detect_image(image, device) {
                Ok(detections) => detections,
                Err(error) => {
                    if std::env::var_os("AKUNA_OCR_DEBUG_LAYOUT").is_some() {
                        eprintln!("layout detection failed: {error:#}");
                    }
                    Vec::new()
                }
            };
            if std::env::var_os("AKUNA_OCR_DEBUG_LAYOUT").is_some() {
                for detection in &detections {
                    eprintln!(
                        "layout label={} score={:.3} bbox={:?}",
                        detection.label, detection.score, detection.bbox
                    );
                }
            }
            let mut parts = Vec::new();
            for detection in detections.into_iter().filter(text_like_detection)
            {
                let Ok(crop) = crop_text_region(image, detection.bbox) else {
                    continue;
                };
                let text = self.extract_image_native(&crop, device)?;
                let text = text.trim();
                if useful_ocr_fragment(text) {
                    parts.push(text.to_string());
                }
            }
            if !parts.is_empty() {
                return Ok(parts.join("\n"));
            }
        }

        self.extract_image_native(image, device)
    }

    fn extract_image_native(
        &self,
        image: &DynamicImage,
        device: &B::Device,
    ) -> Result<String> {
        let input = preprocess_image(image, &self.preprocessor)?;
        let vision_features = self.vision.forward(&input, device);
        let prompt = build_ocr_prompt(&self.tokenizer, &self.config, &input)?;
        let [_vision_tokens, vision_hidden] = vision_features.dims();
        if vision_hidden != GLM_OCR_TEXT_DIMS.hidden_size {
            bail!(
                "GLM OCR native tensor shape unsupported: vision hidden {vision_hidden} cannot replace text hidden {} without merger/downsample",
                GLM_OCR_TEXT_DIMS.hidden_size
            );
        }
        if prompt.image_token_range.len() != _vision_tokens {
            bail!(
                "GLM OCR native tensor shape unsupported: image tokens {} differ from vision features {}",
                prompt.image_token_range.len(),
                _vision_tokens
            );
        }

        let mut input_ids = prompt.input_ids;
        let position_ids = prompt.position_ids;
        let mut generated = Vec::new();
        let embeddings =
            self.text.embed_token_ids(input_ids.as_slice(), device);
        let embeddings = embeddings.slice_assign(
            [
                prompt.image_token_range.clone(),
                0..GLM_OCR_TEXT_DIMS.hidden_size,
            ],
            vision_features.clone(),
        );
        let (logits, mut caches) = self.text.forward_embeddings_with_cache(
            embeddings,
            &position_ids,
            None,
        );
        let mut next_token = last_argmax_with_repetition_penalty(
            logits,
            &generated,
            GLM_OCR_REPETITION_PENALTY,
        )?;
        let mut current_position = next_decode_position(&position_ids);

        let max_tokens = std::env::var("AKUNA_OCR_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(10);
        for _ in 0..max_tokens {
            if self.config.text_config.eos_token_id.contains(&next_token) {
                break;
            }

            input_ids.push(next_token);
            generated.push(next_token);
            let token_embedding =
                self.text.embed_token_ids(&[next_token], device);
            let decode_position_ids = [
                vec![current_position],
                vec![current_position],
                vec![current_position],
            ];
            let (logits, next_caches) =
                self.text.forward_embeddings_with_cache(
                    token_embedding,
                    &decode_position_ids,
                    Some(caches),
                );
            caches = next_caches;
            next_token = last_argmax_with_repetition_penalty(
                logits,
                &generated,
                GLM_OCR_REPETITION_PENALTY,
            )?;
            current_position += 1;
        }

        self.tokenizer
            .decode(&generated, true)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .context("failed to decode GLM OCR generated tokens")
    }

    pub(crate) fn variant(&self) -> GlmOcrVariant {
        self.variant
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(variant: GlmOcrVariant) -> Self
    where
        B: Backend<FloatElem = f32>,
        B::Device: Default,
    {
        Self {
            variant,
            config: GlmOcrConfig::for_test(),
            preprocessor: PreprocessorConfig::for_test(),
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            files: None,
            vision: GlmVisionEncoder::new_for_test(1, &B::Device::default()),
            text: GlmTextModel::new_for_test(1, &B::Device::default()),
            layout: None,
            backend: PhantomData,
        }
    }
}

impl<B> GlmVisionPatchEmbed<B>
where
    B: Backend<FloatElem = f32>,
{
    #[allow(dead_code)]
    fn from_weights(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        let dims = GLM_OCR_VISION_DIMS;
        let patch_width = dims.patch_width();
        let weight_values = read_bf16_tensor_values(
            tensors,
            GLM_VISION_PATCH_EMBED_WEIGHT,
            &[
                dims.hidden_size,
                dims.channels,
                dims.temporal_patch_size,
                dims.patch_size,
                dims.patch_size,
            ],
        )?;
        let weight_values = transpose_flattened_patch_weight(
            &weight_values,
            dims.hidden_size,
            patch_width,
        );
        let bias_data = read_bf16_tensor_data(
            tensors,
            GLM_VISION_PATCH_EMBED_BIAS,
            &[dims.hidden_size],
        )?;

        Ok(Self {
            weight: Tensor::from_data(
                TensorData::new(weight_values, [patch_width, dims.hidden_size]),
                device,
            ),
            bias: Tensor::from_data(bias_data, device),
        })
    }

    #[allow(dead_code)]
    fn new_for_test(device: &B::Device) -> Self {
        Self {
            weight: Tensor::zeros(
                [
                    GLM_OCR_VISION_DIMS.patch_width(),
                    GLM_OCR_VISION_DIMS.hidden_size,
                ],
                device,
            ),
            bias: Tensor::zeros([GLM_OCR_VISION_DIMS.hidden_size], device),
        }
    }

    #[allow(dead_code)]
    fn forward(&self, input: &VisionInput, device: &B::Device) -> Tensor<B, 2> {
        let values = Tensor::<B, 2>::from_data(
            TensorData::new(
                input.values.clone(),
                [input.patches, input.patch_width],
            ),
            device,
        );

        values.matmul(self.weight.clone()) + self.bias.clone().unsqueeze()
    }
}

#[allow(dead_code)]
fn transpose_flattened_patch_weight(
    values: &[f32],
    output_size: usize,
    input_size: usize,
) -> Vec<f32> {
    let mut transposed = vec![0.0; values.len()];
    for output in 0..output_size {
        for input in 0..input_size {
            transposed[input * output_size + output] =
                values[output * input_size + input];
        }
    }

    transposed
}

impl GlmOcrVisionDims {
    #[allow(dead_code)]
    fn patch_width(self) -> usize {
        self.channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
    }
}

impl<B> GlmRmsNorm<B>
where
    B: Backend<FloatElem = f32>,
{
    #[allow(dead_code)]
    fn from_weights(
        tensors: &SafeTensors<'_>,
        name: &str,
        hidden_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: Tensor::from_data(
                read_bf16_tensor_data(tensors, name, &[hidden_size])?,
                device,
            ),
            epsilon: 1.0e-5,
        })
    }

    #[allow(dead_code)]
    fn new_for_test(hidden_size: usize, device: &B::Device) -> Self {
        Self {
            weight: Tensor::ones([hidden_size], device),
            epsilon: 1.0e-5,
        }
    }

    #[allow(dead_code)]
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_rows, hidden_size] = x.dims();
        let variance =
            (x.clone() * x.clone()).sum_dim(1) * (1.0 / hidden_size as f64);
        let inv_rms = (variance + self.epsilon).sqrt().recip();

        (x * inv_rms) * self.weight.clone().unsqueeze()
    }
}

impl<B> GlmVisionBlock<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Default,
{
    #[allow(dead_code)]
    fn from_weights(
        tensors: &SafeTensors<'_>,
        layer: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let dims = GLM_OCR_VISION_DIMS;
        let prefix = format!("model.visual.blocks.{layer}");

        Ok(Self {
            norm1: GlmRmsNorm::from_weights(
                tensors,
                &format!("{prefix}.norm1.weight"),
                dims.hidden_size,
                device,
            )?,
            norm2: GlmRmsNorm::from_weights(
                tensors,
                &format!("{prefix}.norm2.weight"),
                dims.hidden_size,
                device,
            )?,
            q_norm: GlmRmsNorm::from_weights(
                tensors,
                &format!("{prefix}.attn.q_norm.weight"),
                dims.hidden_size / dims.heads,
                device,
            )?,
            k_norm: GlmRmsNorm::from_weights(
                tensors,
                &format!("{prefix}.attn.k_norm.weight"),
                dims.hidden_size / dims.heads,
                device,
            )?,
            qkv_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.attn.qkv.weight"),
                dims.hidden_size * 3,
                dims.hidden_size,
                device,
            )?,
            qkv_bias: Tensor::from_data(
                read_bf16_tensor_data(
                    tensors,
                    &format!("{prefix}.attn.qkv.bias"),
                    &[dims.hidden_size * 3],
                )?,
                device,
            ),
            proj_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.attn.proj.weight"),
                dims.hidden_size,
                dims.hidden_size,
                device,
            )?,
            proj_bias: Tensor::from_data(
                read_bf16_tensor_data(
                    tensors,
                    &format!("{prefix}.attn.proj.bias"),
                    &[dims.hidden_size],
                )?,
                device,
            ),
            gate_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.mlp.gate_proj.weight"),
                dims.intermediate_size,
                dims.hidden_size,
                device,
            )?,
            gate_bias: Tensor::from_data(
                read_bf16_tensor_data(
                    tensors,
                    &format!("{prefix}.mlp.gate_proj.bias"),
                    &[dims.intermediate_size],
                )?,
                device,
            ),
            up_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.mlp.up_proj.weight"),
                dims.intermediate_size,
                dims.hidden_size,
                device,
            )?,
            up_bias: Tensor::from_data(
                read_bf16_tensor_data(
                    tensors,
                    &format!("{prefix}.mlp.up_proj.bias"),
                    &[dims.intermediate_size],
                )?,
                device,
            ),
            down_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.mlp.down_proj.weight"),
                dims.hidden_size,
                dims.intermediate_size,
                device,
            )?,
            down_bias: Tensor::from_data(
                read_bf16_tensor_data(
                    tensors,
                    &format!("{prefix}.mlp.down_proj.bias"),
                    &[dims.hidden_size],
                )?,
                device,
            ),
        })
    }

    #[allow(dead_code)]
    fn new_for_test() -> Self {
        let dims = GLM_OCR_VISION_DIMS;
        let device = B::Device::default();

        Self {
            norm1: GlmRmsNorm::new_for_test(dims.hidden_size, &device),
            norm2: GlmRmsNorm::new_for_test(dims.hidden_size, &device),
            q_norm: GlmRmsNorm::new_for_test(
                dims.hidden_size / dims.heads,
                &device,
            ),
            k_norm: GlmRmsNorm::new_for_test(
                dims.hidden_size / dims.heads,
                &device,
            ),
            qkv_weight: Tensor::zeros(
                [dims.hidden_size, dims.hidden_size * 3],
                &device,
            ),
            qkv_bias: Tensor::zeros([dims.hidden_size * 3], &device),
            proj_weight: Tensor::zeros(
                [dims.hidden_size, dims.hidden_size],
                &device,
            ),
            proj_bias: Tensor::zeros([dims.hidden_size], &device),
            gate_weight: Tensor::zeros(
                [dims.hidden_size, dims.intermediate_size],
                &device,
            ),
            gate_bias: Tensor::zeros([dims.intermediate_size], &device),
            up_weight: Tensor::zeros(
                [dims.hidden_size, dims.intermediate_size],
                &device,
            ),
            up_bias: Tensor::zeros([dims.intermediate_size], &device),
            down_weight: Tensor::zeros(
                [dims.intermediate_size, dims.hidden_size],
                &device,
            ),
            down_bias: Tensor::zeros([dims.hidden_size], &device),
        }
    }

    #[allow(dead_code)]
    fn forward(&self, x: Tensor<B, 2>, grid_thw: [i64; 3]) -> Tensor<B, 2> {
        let dims = GLM_OCR_VISION_DIMS;
        let [patches, hidden_size] = x.dims();
        let head_dim = hidden_size / dims.heads;
        let residual = x.clone();
        let normalized = self.norm1.forward(x);
        let qkv = normalized.matmul(self.qkv_weight.clone())
            + self.qkv_bias.clone().unsqueeze();
        let q = qkv
            .clone()
            .narrow(1, 0, hidden_size)
            .reshape([patches, dims.heads, head_dim])
            .swap_dims(0, 1);
        let q = self
            .q_norm
            .forward(q.reshape([dims.heads * patches, head_dim]))
            .reshape([dims.heads, patches, head_dim]);
        let k = qkv
            .clone()
            .narrow(1, hidden_size, hidden_size)
            .reshape([patches, dims.heads, head_dim])
            .swap_dims(0, 1);
        let k = self
            .k_norm
            .forward(k.reshape([dims.heads * patches, head_dim]))
            .reshape([dims.heads, patches, head_dim]);
        let (q, k) = apply_vision_rotary(q, k, grid_thw);
        let k = k.swap_dims(1, 2);
        let v = qkv
            .narrow(1, hidden_size * 2, hidden_size)
            .reshape([patches, dims.heads, head_dim])
            .swap_dims(0, 1);
        let scores = q.matmul(k) * (1.0 / (head_dim as f64).sqrt());
        let attention = softmax(scores, 2).matmul(v);
        let attention = attention
            .swap_dims(0, 1)
            .reshape([patches, hidden_size])
            .matmul(self.proj_weight.clone())
            + self.proj_bias.clone().unsqueeze();
        let x = residual + attention;
        let residual = x.clone();
        let normalized = self.norm2.forward(x);
        let gate = normalized.clone().matmul(self.gate_weight.clone())
            + self.gate_bias.clone().unsqueeze();
        let up = normalized.matmul(self.up_weight.clone())
            + self.up_bias.clone().unsqueeze();
        let mlp = (silu(gate) * up).matmul(self.down_weight.clone())
            + self.down_bias.clone().unsqueeze();

        residual + mlp
    }
}

impl<B> GlmVisionEncoder<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Default,
{
    #[allow(dead_code)]
    fn from_weights(
        tensors: &SafeTensors<'_>,
        layer_count: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let layers = (0..layer_count)
            .map(|layer| GlmVisionBlock::from_weights(tensors, layer, device))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            patch_embed: GlmVisionPatchEmbed::from_weights(tensors, device)?,
            blocks: layers,
            post_norm: GlmRmsNorm::from_weights(
                tensors,
                "model.visual.post_layernorm.weight",
                GLM_OCR_VISION_DIMS.hidden_size,
                device,
            )?,
            merger: GlmVisionMerger::from_weights(tensors, device)?,
        })
    }

    #[allow(dead_code)]
    fn new_for_test(block_count: usize, device: &B::Device) -> Self {
        let blocks = (0..block_count)
            .map(|_block| GlmVisionBlock::new_for_test())
            .collect();

        Self {
            patch_embed: GlmVisionPatchEmbed::new_for_test(device),
            blocks,
            post_norm: GlmRmsNorm::new_for_test(
                GLM_OCR_VISION_DIMS.hidden_size,
                device,
            ),
            merger: GlmVisionMerger::new_for_test(device),
        }
    }

    #[allow(dead_code)]
    fn forward(&self, input: &VisionInput, device: &B::Device) -> Tensor<B, 2> {
        let x = self.patch_embed.forward(input, device);
        let x = self
            .blocks
            .iter()
            .fold(x, |x, block| block.forward(x, input.grid_thw));

        self.merger
            .forward(self.post_norm.forward(x), input.grid_thw)
    }
}

#[allow(dead_code)]
fn apply_vision_rotary<B>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    grid_thw: [i64; 3],
) -> (Tensor<B, 3>, Tensor<B, 3>)
where
    B: Backend<FloatElem = f32>,
    B::Device: Default,
{
    let dims = GLM_OCR_VISION_DIMS;
    let head_dim = dims.hidden_size / dims.heads;
    let position_ids = build_vision_position_ids(grid_thw, dims.merge_size);
    let device = B::Device::default();
    let mut cos = Vec::with_capacity(position_ids.len() * head_dim);
    let mut sin = Vec::with_capacity(position_ids.len() * head_dim);

    for [row, column] in position_ids {
        let mut rotary = Vec::with_capacity(head_dim / 2);
        for position in [row, column] {
            for offset in 0..(head_dim / 4) {
                let inv_freq = 1.0_f32
                    / 10_000.0_f32
                        .powf((2 * offset) as f32 / (head_dim / 2) as f32);
                rotary.push(position as f32 * inv_freq);
            }
        }
        for value in rotary.iter().chain(rotary.iter()) {
            cos.push(value.cos());
            sin.push(value.sin());
        }
    }

    let patches = cos.len() / head_dim;
    let cos: Tensor<B, 3> = Tensor::<B, 2>::from_data(
        TensorData::new(cos, [patches, head_dim]),
        &device,
    )
    .unsqueeze_dim::<3>(1);
    let sin: Tensor<B, 3> = Tensor::<B, 2>::from_data(
        TensorData::new(sin, [patches, head_dim]),
        &device,
    )
    .unsqueeze_dim::<3>(1);
    let cos = cos.swap_dims(0, 1);
    let sin = sin.swap_dims(0, 1);
    let q_rotated =
        q.clone() * cos.clone() + rotate_half_vision(q) * sin.clone();
    let k_rotated = k.clone() * cos + rotate_half_vision(k) * sin;

    (q_rotated, k_rotated)
}

#[allow(dead_code)]
fn build_vision_position_ids(
    grid_thw: [i64; 3],
    merge_size: usize,
) -> Vec<[usize; 2]> {
    let [time, height, width] = grid_thw.map(|value| value as usize);
    let mut ids = Vec::with_capacity(time * height * width);

    for _frame in 0..time {
        for block_row in 0..height / merge_size {
            for block_column in 0..width / merge_size {
                for row_offset in 0..merge_size {
                    for column_offset in 0..merge_size {
                        ids.push([
                            block_row * merge_size + row_offset,
                            block_column * merge_size + column_offset,
                        ]);
                    }
                }
            }
        }
    }

    ids
}

#[allow(dead_code)]
fn rotate_half_vision<B>(x: Tensor<B, 3>) -> Tensor<B, 3>
where
    B: Backend<FloatElem = f32>,
{
    let dims = GLM_OCR_VISION_DIMS;
    let head_dim = dims.hidden_size / dims.heads;
    let [heads, patches, _head_dim] = x.dims();
    let first = x.clone().narrow(2, 0, head_dim / 2);
    let second = x.narrow(2, head_dim / 2, head_dim / 2) * -1.0;

    Tensor::cat(vec![second, first], 2).reshape([heads, patches, head_dim])
}

impl<B> GlmVisionMerger<B>
where
    B: Backend<FloatElem = f32>,
{
    #[allow(dead_code)]
    fn from_weights(
        tensors: &SafeTensors<'_>,
        device: &B::Device,
    ) -> Result<Self> {
        let dims = GLM_OCR_VISION_DIMS;

        Ok(Self {
            downsample_weight: read_flattened_downsample_weight(
                tensors, device,
            )?,
            downsample_bias: Tensor::from_data(
                read_bf16_tensor_data(
                    tensors,
                    "model.visual.downsample.bias",
                    &[dims.output_hidden_size],
                )?,
                device,
            ),
            proj_weight: read_transposed_bf16_tensor(
                tensors,
                "model.visual.merger.proj.weight",
                dims.output_hidden_size,
                dims.output_hidden_size,
                device,
            )?,
            norm_weight: Tensor::from_data(
                read_bf16_tensor_data(
                    tensors,
                    "model.visual.merger.post_projection_norm.weight",
                    &[dims.output_hidden_size],
                )?,
                device,
            ),
            norm_bias: Tensor::from_data(
                read_bf16_tensor_data(
                    tensors,
                    "model.visual.merger.post_projection_norm.bias",
                    &[dims.output_hidden_size],
                )?,
                device,
            ),
            gate_weight: read_transposed_bf16_tensor(
                tensors,
                "model.visual.merger.gate_proj.weight",
                dims.output_hidden_size * 3,
                dims.output_hidden_size,
                device,
            )?,
            up_weight: read_transposed_bf16_tensor(
                tensors,
                "model.visual.merger.up_proj.weight",
                dims.output_hidden_size * 3,
                dims.output_hidden_size,
                device,
            )?,
            down_weight: read_transposed_bf16_tensor(
                tensors,
                "model.visual.merger.down_proj.weight",
                dims.output_hidden_size,
                dims.output_hidden_size * 3,
                device,
            )?,
        })
    }

    #[allow(dead_code)]
    fn new_for_test(device: &B::Device) -> Self {
        let dims = GLM_OCR_VISION_DIMS;

        Self {
            downsample_weight: Tensor::zeros(
                [
                    dims.hidden_size * dims.merge_size * dims.merge_size,
                    dims.output_hidden_size,
                ],
                device,
            ),
            downsample_bias: Tensor::zeros([dims.output_hidden_size], device),
            proj_weight: Tensor::zeros(
                [dims.output_hidden_size, dims.output_hidden_size],
                device,
            ),
            norm_weight: Tensor::ones([dims.output_hidden_size], device),
            norm_bias: Tensor::zeros([dims.output_hidden_size], device),
            gate_weight: Tensor::zeros(
                [dims.output_hidden_size, dims.output_hidden_size * 3],
                device,
            ),
            up_weight: Tensor::zeros(
                [dims.output_hidden_size, dims.output_hidden_size * 3],
                device,
            ),
            down_weight: Tensor::zeros(
                [dims.output_hidden_size * 3, dims.output_hidden_size],
                device,
            ),
        }
    }

    #[allow(dead_code)]
    fn forward(&self, x: Tensor<B, 2>, grid_thw: [i64; 3]) -> Tensor<B, 2> {
        let dims = GLM_OCR_VISION_DIMS;
        let [time, height, width] = grid_thw.map(|value| value as usize);
        let merged_height = height / dims.merge_size;
        let merged_width = width / dims.merge_size;
        let tokens = time * merged_height * merged_width;
        let hidden = x.reshape([
            tokens,
            dims.merge_size,
            dims.merge_size,
            dims.hidden_size,
        ]);
        let hidden = hidden.swap_dims(1, 3).swap_dims(2, 3).reshape([
            tokens,
            dims.hidden_size * dims.merge_size * dims.merge_size,
        ]);
        let projected = hidden.matmul(self.downsample_weight.clone())
            + self.downsample_bias.clone().unsqueeze();
        let projected = projected.matmul(self.proj_weight.clone());
        let projected = layer_norm(
            gelu(projected),
            self.norm_weight.clone(),
            self.norm_bias.clone(),
        );
        let gate = projected.clone().matmul(self.gate_weight.clone());
        let up = projected.matmul(self.up_weight.clone());

        (silu(gate) * up).matmul(self.down_weight.clone())
    }
}

impl<B> GlmTextDecoderLayer<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Default,
{
    #[allow(dead_code)]
    fn from_weights(
        tensors: &SafeTensors<'_>,
        layer: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let dims = GLM_OCR_TEXT_DIMS;
        let prefix = format!("model.language_model.layers.{layer}");

        Ok(Self {
            input_layernorm: GlmRmsNorm::from_weights(
                tensors,
                &format!("{prefix}.input_layernorm.weight"),
                dims.hidden_size,
                device,
            )?,
            post_self_attn_layernorm: GlmRmsNorm::from_weights(
                tensors,
                &format!("{prefix}.post_self_attn_layernorm.weight"),
                dims.hidden_size,
                device,
            )?,
            post_attention_layernorm: GlmRmsNorm::from_weights(
                tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                dims.hidden_size,
                device,
            )?,
            post_mlp_layernorm: GlmRmsNorm::from_weights(
                tensors,
                &format!("{prefix}.post_mlp_layernorm.weight"),
                dims.hidden_size,
                device,
            )?,
            q_proj_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.self_attn.q_proj.weight"),
                dims.attention_q_size,
                dims.hidden_size,
                device,
            )?,
            k_proj_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.self_attn.k_proj.weight"),
                dims.attention_kv_size,
                dims.hidden_size,
                device,
            )?,
            v_proj_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.self_attn.v_proj.weight"),
                dims.attention_kv_size,
                dims.hidden_size,
                device,
            )?,
            o_proj_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                dims.hidden_size,
                dims.attention_q_size,
                device,
            )?,
            gate_up_proj_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.mlp.gate_up_proj.weight"),
                dims.intermediate_size * 2,
                dims.hidden_size,
                device,
            )?,
            down_proj_weight: read_transposed_bf16_tensor(
                tensors,
                &format!("{prefix}.mlp.down_proj.weight"),
                dims.hidden_size,
                dims.intermediate_size,
                device,
            )?,
        })
    }

    #[allow(dead_code)]
    fn new_for_test() -> Self {
        let dims = GLM_OCR_TEXT_DIMS;
        let device = B::Device::default();

        Self {
            input_layernorm: GlmRmsNorm::new_for_test(
                dims.hidden_size,
                &device,
            ),
            post_self_attn_layernorm: GlmRmsNorm::new_for_test(
                dims.hidden_size,
                &device,
            ),
            post_attention_layernorm: GlmRmsNorm::new_for_test(
                dims.hidden_size,
                &device,
            ),
            post_mlp_layernorm: GlmRmsNorm::new_for_test(
                dims.hidden_size,
                &device,
            ),
            q_proj_weight: Tensor::zeros(
                [dims.hidden_size, dims.attention_q_size],
                &device,
            ),
            k_proj_weight: Tensor::zeros(
                [dims.hidden_size, dims.attention_kv_size],
                &device,
            ),
            v_proj_weight: Tensor::zeros(
                [dims.hidden_size, dims.attention_kv_size],
                &device,
            ),
            o_proj_weight: Tensor::zeros(
                [dims.attention_q_size, dims.hidden_size],
                &device,
            ),
            gate_up_proj_weight: Tensor::zeros(
                [dims.hidden_size, dims.intermediate_size * 2],
                &device,
            ),
            down_proj_weight: Tensor::zeros(
                [dims.intermediate_size, dims.hidden_size],
                &device,
            ),
        }
    }

    #[allow(dead_code)]
    fn forward(
        &self,
        x: Tensor<B, 2>,
        position_ids: &[Vec<i64>; 3],
    ) -> Tensor<B, 2> {
        self.forward_with_cache(x, position_ids, None).0
    }

    #[allow(dead_code)]
    fn forward_with_cache(
        &self,
        x: Tensor<B, 2>,
        position_ids: &[Vec<i64>; 3],
        cache: Option<GlmTextLayerCache<B>>,
    ) -> (Tensor<B, 2>, GlmTextLayerCache<B>) {
        let dims = GLM_OCR_TEXT_DIMS;
        let [seq, hidden_size] = x.dims();
        let groups = dims.heads / dims.kv_heads;
        let residual = x.clone();
        let normalized = self.input_layernorm.forward(x);
        let q = normalized
            .clone()
            .matmul(self.q_proj_weight.clone())
            .reshape([seq, dims.kv_heads, groups, dims.head_dim])
            .swap_dims(0, 1)
            .swap_dims(1, 2);
        let k = normalized
            .clone()
            .matmul(self.k_proj_weight.clone())
            .reshape([seq, dims.kv_heads, dims.head_dim])
            .swap_dims(0, 1)
            .unsqueeze_dim(1);
        let (q, k) = apply_text_rotary(q, k, position_ids);
        let v = normalized
            .matmul(self.v_proj_weight.clone())
            .reshape([seq, dims.kv_heads, dims.head_dim])
            .swap_dims(0, 1)
            .unsqueeze_dim(1);
        let (k, v) = if let Some(cache) = cache {
            (
                Tensor::cat(vec![cache.key, k], 2),
                Tensor::cat(vec![cache.value, v], 2),
            )
        } else {
            (k, v)
        };
        let next_cache = GlmTextLayerCache {
            key: k.clone(),
            value: v.clone(),
        };
        let k = k.swap_dims(2, 3);
        let scores = q.matmul(k) * (1.0 / (dims.head_dim as f64).sqrt());
        let [_kv_heads, _cache_groups, cache_len, _head_dim] =
            next_cache.key.dims();
        let scores = if cache_len == seq {
            scores + causal_attention_mask::<B>(dims.kv_heads, groups, seq)
        } else {
            scores
        };
        let attention = softmax(scores, 3).matmul(v);
        let attention = attention
            .swap_dims(1, 2)
            .swap_dims(0, 1)
            .reshape([seq, dims.attention_q_size])
            .matmul(self.o_proj_weight.clone());
        let attention = self.post_self_attn_layernorm.forward(attention);
        let x = residual + attention;
        let residual = x.clone();
        let normalized = self.post_attention_layernorm.forward(x);
        let gate_up = normalized.matmul(self.gate_up_proj_weight.clone());
        let gate = gate_up.clone().narrow(1, 0, dims.intermediate_size);
        let up =
            gate_up.narrow(1, dims.intermediate_size, dims.intermediate_size);
        let mlp = (silu(gate) * up).matmul(self.down_proj_weight.clone());
        let mlp = self.post_mlp_layernorm.forward(mlp);

        ((residual + mlp).reshape([seq, hidden_size]), next_cache)
    }
}

impl<B> GlmTextModel<B>
where
    B: Backend<FloatElem = f32>,
    B::Device: Default,
{
    #[allow(dead_code)]
    fn from_weights(
        tensors: &SafeTensors<'_>,
        layer_count: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let dims = GLM_OCR_TEXT_DIMS;
        let layers = (0..layer_count)
            .map(|layer| {
                GlmTextDecoderLayer::from_weights(tensors, layer, device)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            embed_tokens: Tensor::from_data(
                read_bf16_tensor_data(
                    tensors,
                    "model.language_model.embed_tokens.weight",
                    &[dims.vocab_size, dims.hidden_size],
                )?,
                device,
            ),
            layers,
            norm: GlmRmsNorm::from_weights(
                tensors,
                "model.language_model.norm.weight",
                dims.hidden_size,
                device,
            )?,
            lm_head: read_transposed_bf16_tensor(
                tensors,
                "lm_head.weight",
                dims.vocab_size,
                dims.hidden_size,
                device,
            )?,
        })
    }

    #[allow(dead_code)]
    fn new_for_test(layer_count: usize, device: &B::Device) -> Self {
        let dims = GLM_OCR_TEXT_DIMS;
        let layers = (0..layer_count)
            .map(|_layer| GlmTextDecoderLayer::new_for_test())
            .collect();

        Self {
            embed_tokens: Tensor::zeros(
                [dims.vocab_size, dims.hidden_size],
                device,
            ),
            layers,
            norm: GlmRmsNorm::new_for_test(dims.hidden_size, device),
            lm_head: Tensor::zeros([dims.hidden_size, dims.vocab_size], device),
        }
    }

    #[allow(dead_code)]
    fn embed_token_ids(
        &self,
        token_ids: &[u32],
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let indices = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                token_ids
                    .iter()
                    .map(|id| i64::from(*id))
                    .collect::<Vec<_>>(),
                [token_ids.len()],
            ),
            device,
        );

        self.embed_tokens.clone().select(0, indices)
    }

    #[allow(dead_code)]
    fn forward_embeddings(
        &self,
        embeddings: Tensor<B, 2>,
        position_ids: &[Vec<i64>; 3],
    ) -> Tensor<B, 2> {
        self.forward_embeddings_with_cache(embeddings, position_ids, None)
            .0
    }

    #[allow(dead_code)]
    fn forward_embeddings_with_cache(
        &self,
        embeddings: Tensor<B, 2>,
        position_ids: &[Vec<i64>; 3],
        caches: Option<Vec<GlmTextLayerCache<B>>>,
    ) -> (Tensor<B, 2>, Vec<GlmTextLayerCache<B>>) {
        let mut hidden = embeddings;
        let mut next_caches = Vec::with_capacity(self.layers.len());
        let mut cache_iter = caches.unwrap_or_default().into_iter();

        for layer in &self.layers {
            let cache = cache_iter.next();
            let (next_hidden, next_cache) =
                layer.forward_with_cache(hidden, position_ids, cache);
            hidden = next_hidden;
            next_caches.push(next_cache);
        }

        (
            self.norm.forward(hidden).matmul(self.lm_head.clone()),
            next_caches,
        )
    }
}

#[allow(dead_code)]
fn apply_text_rotary<B>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    position_ids: &[Vec<i64>; 3],
) -> (Tensor<B, 4>, Tensor<B, 4>)
where
    B: Backend<FloatElem = f32>,
    B::Device: Default,
{
    let dims = GLM_OCR_TEXT_DIMS;
    let seq = position_ids[0].len();
    let device = B::Device::default();
    let mut cos = Vec::with_capacity(seq * dims.head_dim);
    let mut sin = Vec::with_capacity(seq * dims.head_dim);

    for index in 0..seq {
        let mut freqs = Vec::with_capacity(dims.head_dim / 2);
        for (axis, section) in GLM_OCR_MROPE_SECTION.into_iter().enumerate() {
            let position = position_ids[axis][index] as f32;
            for offset in 0..section {
                let frequency_index = freqs.len() + offset;
                let inv_freq = 1.0_f32
                    / 10_000.0_f32.powf(
                        (2 * frequency_index) as f32 / dims.head_dim as f32,
                    );
                freqs.push(position * inv_freq);
            }
        }
        for value in freqs {
            let cos_value = value.cos();
            let sin_value = value.sin();
            cos.push(cos_value);
            cos.push(cos_value);
            sin.push(sin_value);
            sin.push(sin_value);
        }
    }

    let cos: Tensor<B, 4> = Tensor::<B, 2>::from_data(
        TensorData::new(cos, [seq, dims.head_dim]),
        &device,
    )
    .unsqueeze_dim::<3>(0)
    .unsqueeze_dim::<4>(0);
    let sin: Tensor<B, 4> = Tensor::<B, 2>::from_data(
        TensorData::new(sin, [seq, dims.head_dim]),
        &device,
    )
    .unsqueeze_dim::<3>(0)
    .unsqueeze_dim::<4>(0);
    let q_rotated = q.clone() * cos.clone() + rotate_half_llm(q) * sin.clone();
    let k_rotated = k.clone() * cos + rotate_half_llm(k) * sin;

    (q_rotated, k_rotated)
}

#[allow(dead_code)]
fn rotate_half_llm<B>(x: Tensor<B, 4>) -> Tensor<B, 4>
where
    B: Backend<FloatElem = f32>,
{
    let dims = GLM_OCR_TEXT_DIMS;
    let [kv_heads, groups, seq, _head_dim] = x.dims();
    let paired = x.reshape([kv_heads, groups, seq, dims.head_dim / 2, 2]);
    let even = paired.clone().narrow(4, 0, 1);
    let negative_odd = paired.narrow(4, 1, 1) * -1.0;

    Tensor::cat(vec![negative_odd, even], 4).reshape([
        kv_heads,
        groups,
        seq,
        dims.head_dim,
    ])
}

#[allow(dead_code)]
fn causal_attention_mask<B>(
    kv_heads: usize,
    groups: usize,
    seq: usize,
) -> Tensor<B, 4>
where
    B: Backend<FloatElem = f32>,
    B::Device: Default,
{
    let device = B::Device::default();
    let mut values = Vec::with_capacity(kv_heads * groups * seq * seq);

    for _kv_head in 0..kv_heads {
        for _group in 0..groups {
            for query in 0..seq {
                for key in 0..seq {
                    values.push(if key > query { -1.0e9 } else { 0.0 });
                }
            }
        }
    }

    Tensor::from_data(
        TensorData::new(values, [kv_heads, groups, seq, seq]),
        &device,
    )
}

#[allow(dead_code)]
fn silu<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    x.clone() / ((x * -1.0).exp() + 1.0)
}

#[allow(dead_code)]
fn gelu<B: Backend<FloatElem = f32>, const D: usize>(
    x: Tensor<B, D>,
) -> Tensor<B, D> {
    let cubic = x.clone() * x.clone() * x.clone();
    let inner = (x.clone() + cubic * 0.044_715) * 0.797_884_6;
    x * ((inner.tanh() + 1.0) * 0.5)
}

#[allow(dead_code)]
fn layer_norm<B>(
    x: Tensor<B, 2>,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 1>,
) -> Tensor<B, 2>
where
    B: Backend<FloatElem = f32>,
{
    let [_rows, hidden_size] = x.dims();
    let mean = x.clone().sum_dim(1) * (1.0 / hidden_size as f64);
    let centered = x - mean;
    let variance = (centered.clone() * centered.clone()).sum_dim(1)
        * (1.0 / hidden_size as f64);

    centered * (variance + 1.0e-5).sqrt().recip() * weight.unsqueeze()
        + bias.unsqueeze()
}

fn last_argmax_with_repetition_penalty<B>(
    logits: Tensor<B, 2>,
    generated_tokens: &[u32],
    penalty: f32,
) -> Result<u32>
where
    B: Backend<FloatElem = f32>,
{
    let [rows, columns] = logits.dims();
    if rows == 0 || columns == 0 {
        bail!("GLM OCR logits cannot be empty");
    }

    let values = logits.into_data().to_vec::<f32>()?;
    let start = (rows - 1) * columns;
    let mut last_logits = values[start..start + columns].to_vec();
    if penalty != 1.0 {
        for token_id in generated_tokens {
            let index = usize::try_from(*token_id)?;
            if index >= last_logits.len() {
                continue;
            }
            if last_logits[index] > 0.0 {
                last_logits[index] /= penalty;
            } else {
                last_logits[index] *= penalty;
            }
        }
    }

    let token = last_logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _value)| index)
        .context("GLM OCR logits cannot be empty")?;

    Ok(u32::try_from(token)?)
}

#[allow(dead_code)]
fn last_argmax<B>(logits: Tensor<B, 2>) -> Result<u32>
where
    B: Backend<FloatElem = f32>,
{
    let generated_tokens: &[u32] = &[];
    last_argmax_with_repetition_penalty(logits, generated_tokens, 1.0)
}

pub(crate) async fn load_glm_ocr<B>(
    device: &B::Device,
    variant: GlmOcrVariant,
    cache_dir: Option<PathBuf>,
) -> Result<GlmOcrModel<B>>
where
    B: Backend<FloatElem = f32>,
    B::Device: Default,
{
    let files = download_glm_ocr_model(variant, cache_dir.clone()).await?;
    let config = load_json::<GlmOcrConfig>(&files.config_path)
        .context("failed to load GLM OCR config")?;
    let preprocessor =
        load_json::<PreprocessorConfig>(&files.preprocessor_config_path)
            .context("failed to load GLM OCR preprocessor config")?;
    config.validate(&preprocessor)?;
    let tokenizer = Tokenizer::from_file(&files.tokenizer_path)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .context("failed to load GLM OCR tokenizer")?;
    let weights = std::fs::read(&files.weights_path).with_context(|| {
        format!("failed to read {}", files.weights_path.display())
    })?;
    let tensors = SafeTensors::deserialize(&weights)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| {
            format!("failed to parse {}", files.weights_path.display())
        })?;
    validate_weights(&tensors)?;
    let vision = GlmVisionEncoder::from_weights(
        &tensors,
        GLM_OCR_VISION_DIMS.layers,
        device,
    )?;
    let text =
        GlmTextModel::from_weights(&tensors, GLM_OCR_TEXT_DIMS.layers, device)?;
    let layout = load_pp_doclayout_runtime(device, cache_dir).await.ok();

    Ok(GlmOcrModel {
        variant,
        config,
        preprocessor,
        tokenizer,
        files: Some(files),
        vision,
        text,
        layout,
        backend: PhantomData,
    })
}

async fn download_glm_ocr_model(
    variant: GlmOcrVariant,
    cache_dir: Option<PathBuf>,
) -> Result<GlmOcrFiles> {
    let mut builder = ApiBuilder::new().with_progress(true);
    if let Some(cache_dir) = cache_dir {
        builder = builder.with_cache_dir(cache_dir);
    }

    let api = builder
        .build()
        .context("failed to initialize Hugging Face API for OCR model")?;
    let repo = api.repo(Repo::with_revision(
        variant.repo_id().to_string(),
        RepoType::Model,
        variant.revision().to_string(),
    ));

    Ok(GlmOcrFiles {
        config_path: get_repo_file(&repo, variant, "config.json").await?,
        generation_config_path: get_repo_file(
            &repo,
            variant,
            "generation_config.json",
        )
        .await?,
        preprocessor_config_path: get_repo_file(
            &repo,
            variant,
            "preprocessor_config.json",
        )
        .await?,
        tokenizer_path: get_repo_file(&repo, variant, "tokenizer.json").await?,
        weights_path: get_repo_file(&repo, variant, "model.safetensors")
            .await?,
    })
}

async fn get_repo_file(
    repo: &hf_hub::api::tokio::ApiRepo,
    variant: GlmOcrVariant,
    file: &str,
) -> Result<PathBuf> {
    repo.get(file).await.with_context(|| {
        format!(
            "failed to fetch OCR model file {file} from {}",
            variant.repo_id()
        )
    })
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn validate_weights(tensors: &SafeTensors<'_>) -> Result<()> {
    for (name, shape) in required_weight_shapes() {
        let tensor = tensors
            .tensor(&name)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .with_context(|| format!("missing GLM OCR tensor {name}"))?;
        if tensor.dtype() != Dtype::BF16 {
            bail!("GLM OCR tensor {name} must be BF16");
        }
        if tensor.shape() != shape {
            bail!(
                "GLM OCR tensor {name} shape mismatch: expected {:?}, got {:?}",
                &shape,
                tensor.shape()
            );
        }
    }

    Ok(())
}

#[allow(dead_code)]
fn read_bf16_tensor_values(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| format!("missing GLM OCR tensor {name}"))?;
    if tensor.dtype() != Dtype::BF16 {
        bail!("GLM OCR tensor {name} must be BF16");
    }
    if tensor.shape() != shape {
        bail!(
            "GLM OCR tensor {name} shape mismatch: expected {:?}, got {:?}",
            shape,
            tensor.shape()
        );
    }

    Ok(tensor
        .data()
        .chunks_exact(2)
        .map(|bytes| {
            bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
        })
        .collect())
}

#[allow(dead_code)]
fn read_bf16_tensor_data(
    tensors: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<TensorData> {
    Ok(TensorData::new(
        read_bf16_tensor_values(tensors, name, shape)?,
        shape.to_vec(),
    ))
}

#[allow(dead_code)]
fn read_transposed_bf16_tensor<B>(
    tensors: &SafeTensors<'_>,
    name: &str,
    rows: usize,
    columns: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2>>
where
    B: Backend<FloatElem = f32>,
{
    let values = read_bf16_tensor_values(tensors, name, &[rows, columns])?;
    let values = transpose_flattened_matrix(&values, rows, columns);

    Ok(Tensor::from_data(
        TensorData::new(values, [columns, rows]),
        device,
    ))
}

#[allow(dead_code)]
fn read_flattened_downsample_weight<B>(
    tensors: &SafeTensors<'_>,
    device: &B::Device,
) -> Result<Tensor<B, 2>>
where
    B: Backend<FloatElem = f32>,
{
    let dims = GLM_OCR_VISION_DIMS;
    let values = read_bf16_tensor_values(
        tensors,
        "model.visual.downsample.weight",
        &[
            dims.output_hidden_size,
            dims.hidden_size,
            dims.merge_size,
            dims.merge_size,
        ],
    )?;
    let mut flattened = vec![0.0; values.len()];

    for output in 0..dims.output_hidden_size {
        for input in 0..dims.hidden_size {
            for row in 0..dims.merge_size {
                for column in 0..dims.merge_size {
                    let source = (((output * dims.hidden_size + input)
                        * dims.merge_size
                        + row)
                        * dims.merge_size)
                        + column;
                    let target = ((input * dims.merge_size + row)
                        * dims.merge_size
                        + column)
                        * dims.output_hidden_size
                        + output;
                    flattened[target] = values[source];
                }
            }
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(
            flattened,
            [
                dims.hidden_size * dims.merge_size * dims.merge_size,
                dims.output_hidden_size,
            ],
        ),
        device,
    ))
}

#[allow(dead_code)]
fn transpose_flattened_matrix(
    values: &[f32],
    rows: usize,
    columns: usize,
) -> Vec<f32> {
    let mut transposed = vec![0.0; values.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[column * rows + row] = values[row * columns + column];
        }
    }

    transposed
}

fn required_weight_shapes() -> Vec<(String, Vec<usize>)> {
    let vision = GLM_OCR_VISION_DIMS;
    let text = GLM_OCR_TEXT_DIMS;

    let mut shapes = vec![
        (
            "model.visual.patch_embed.proj.weight".to_string(),
            vec![
                vision.hidden_size,
                vision.channels,
                vision.temporal_patch_size,
                vision.patch_size,
                vision.patch_size,
            ],
        ),
        (
            "model.visual.patch_embed.proj.bias".to_string(),
            vec![vision.hidden_size],
        ),
        (
            "model.visual.post_layernorm.weight".to_string(),
            vec![vision.hidden_size],
        ),
        (
            "model.visual.downsample.weight".to_string(),
            vec![
                vision.output_hidden_size,
                vision.hidden_size,
                vision.merge_size,
                vision.merge_size,
            ],
        ),
        (
            "model.visual.downsample.bias".to_string(),
            vec![vision.output_hidden_size],
        ),
        (
            "model.visual.merger.proj.weight".to_string(),
            vec![vision.output_hidden_size, vision.output_hidden_size],
        ),
        (
            "model.visual.merger.post_projection_norm.weight".to_string(),
            vec![vision.output_hidden_size],
        ),
        (
            "model.visual.merger.post_projection_norm.bias".to_string(),
            vec![vision.output_hidden_size],
        ),
        (
            "model.visual.merger.gate_proj.weight".to_string(),
            vec![vision.output_hidden_size * 3, vision.output_hidden_size],
        ),
        (
            "model.visual.merger.up_proj.weight".to_string(),
            vec![vision.output_hidden_size * 3, vision.output_hidden_size],
        ),
        (
            "model.visual.merger.down_proj.weight".to_string(),
            vec![vision.output_hidden_size, vision.output_hidden_size * 3],
        ),
        (
            "model.language_model.embed_tokens.weight".to_string(),
            vec![text.vocab_size, text.hidden_size],
        ),
        (
            "model.language_model.norm.weight".to_string(),
            vec![text.hidden_size],
        ),
        (
            "lm_head.weight".to_string(),
            vec![text.vocab_size, text.hidden_size],
        ),
    ];
    for layer in 0..vision.layers {
        shapes.extend(vision_block_weight_shapes(layer));
    }
    for layer in 0..text.layers {
        shapes.extend(text_layer_weight_shapes(layer));
    }

    shapes
}

fn vision_block_weight_shapes(layer: usize) -> Vec<(String, Vec<usize>)> {
    let dims = GLM_OCR_VISION_DIMS;
    let prefix = format!("model.visual.blocks.{layer}");

    vec![
        (format!("{prefix}.norm1.weight"), vec![dims.hidden_size]),
        (format!("{prefix}.norm2.weight"), vec![dims.hidden_size]),
        (
            format!("{prefix}.attn.q_norm.weight"),
            vec![dims.hidden_size / dims.heads],
        ),
        (
            format!("{prefix}.attn.k_norm.weight"),
            vec![dims.hidden_size / dims.heads],
        ),
        (
            format!("{prefix}.attn.qkv.weight"),
            vec![dims.hidden_size * 3, dims.hidden_size],
        ),
        (
            format!("{prefix}.attn.qkv.bias"),
            vec![dims.hidden_size * 3],
        ),
        (
            format!("{prefix}.attn.proj.weight"),
            vec![dims.hidden_size, dims.hidden_size],
        ),
        (format!("{prefix}.attn.proj.bias"), vec![dims.hidden_size]),
        (
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![dims.intermediate_size, dims.hidden_size],
        ),
        (
            format!("{prefix}.mlp.gate_proj.bias"),
            vec![dims.intermediate_size],
        ),
        (
            format!("{prefix}.mlp.up_proj.weight"),
            vec![dims.intermediate_size, dims.hidden_size],
        ),
        (
            format!("{prefix}.mlp.up_proj.bias"),
            vec![dims.intermediate_size],
        ),
        (
            format!("{prefix}.mlp.down_proj.weight"),
            vec![dims.hidden_size, dims.intermediate_size],
        ),
        (
            format!("{prefix}.mlp.down_proj.bias"),
            vec![dims.hidden_size],
        ),
    ]
}

fn text_layer_weight_shapes(layer: usize) -> Vec<(String, Vec<usize>)> {
    let dims = GLM_OCR_TEXT_DIMS;
    let prefix = format!("model.language_model.layers.{layer}");

    vec![
        (
            format!("{prefix}.input_layernorm.weight"),
            vec![dims.hidden_size],
        ),
        (
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![dims.hidden_size],
        ),
        (
            format!("{prefix}.post_self_attn_layernorm.weight"),
            vec![dims.hidden_size],
        ),
        (
            format!("{prefix}.post_mlp_layernorm.weight"),
            vec![dims.hidden_size],
        ),
        (
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![dims.attention_q_size, dims.hidden_size],
        ),
        (
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![dims.attention_kv_size, dims.hidden_size],
        ),
        (
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![dims.attention_kv_size, dims.hidden_size],
        ),
        (
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![dims.hidden_size, dims.attention_q_size],
        ),
        (
            format!("{prefix}.mlp.gate_up_proj.weight"),
            vec![dims.intermediate_size * 2, dims.hidden_size],
        ),
        (
            format!("{prefix}.mlp.down_proj.weight"),
            vec![dims.hidden_size, dims.intermediate_size],
        ),
    ]
}

impl GlmOcrConfig {
    fn validate(&self, preprocessor: &PreprocessorConfig) -> Result<()> {
        if self.image_start_token_id == self.image_end_token_id {
            bail!("GLM OCR image start/end token ids must differ");
        }
        if self.image_token_id == self.image_start_token_id
            || self.image_token_id == self.image_end_token_id
        {
            bail!("GLM OCR image token id must differ from boundary tokens");
        }
        if self.text_config.eos_token_id.is_empty() {
            bail!("GLM OCR EOS token ids are missing");
        }
        if self.text_config.pad_token_id >= self.text_config.vocab_size as u32 {
            bail!("GLM OCR PAD token id exceeds vocab size");
        }
        if self.vision_config.image_size == 0 {
            bail!("GLM OCR vision image size must be greater than zero");
        }
        if self.vision_config.patch_size != preprocessor.patch_size {
            bail!("GLM OCR config/preprocessor patch sizes differ");
        }
        if self.vision_config.spatial_merge_size != preprocessor.merge_size {
            bail!("GLM OCR config/preprocessor merge sizes differ");
        }

        Ok(())
    }

    #[cfg(test)]
    fn for_test() -> Self {
        Self {
            image_start_token_id: 59256,
            image_end_token_id: 59257,
            image_token_id: 59280,
            text_config: TextConfig {
                eos_token_id: vec![59246, 59253],
                pad_token_id: 59246,
                vocab_size: 59392,
            },
            vision_config: VisionConfig {
                image_size: 336,
                patch_size: 14,
                spatial_merge_size: 2,
            },
        }
    }
}

impl PreprocessorConfig {
    #[cfg(test)]
    fn for_test() -> Self {
        Self {
            size: ImageSize {
                shortest_edge: 12_544,
                longest_edge: 9_633_792,
            },
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            image_std: [0.26862954, 0.261_302_6, 0.275_777_1],
            patch_size: 14,
            temporal_patch_size: 2,
            merge_size: 2,
        }
    }
}

#[derive(Debug, PartialEq)]
struct VisionInput {
    values: Vec<f32>,
    patches: usize,
    patch_width: usize,
    grid_thw: [i64; 3],
}

fn preprocess_image(
    image: &DynamicImage,
    config: &PreprocessorConfig,
) -> Result<VisionInput> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        bail!("OCR input image cannot be empty");
    }

    let (target_width, target_height) = resized_dimensions(
        width,
        height,
        config.patch_size * config.merge_size,
        config.size.shortest_edge,
        config.size.longest_edge.min(GLM_OCR_NATIVE_MAX_PIXELS),
    )?;
    let rgb = image
        .resize_exact(target_width, target_height, FilterType::CatmullRom)
        .to_rgb8();
    let target_width = target_width as usize;
    let target_height = target_height as usize;
    let grid_h = target_height / config.patch_size;
    let grid_w = target_width / config.patch_size;
    let patches = grid_h * grid_w;
    let patch_area = config.patch_size * config.patch_size;
    let patch_width = 3 * config.temporal_patch_size * patch_area;
    let mut values = Vec::with_capacity(patches * patch_width);

    for block_y in 0..grid_h / config.merge_size {
        for block_x in 0..grid_w / config.merge_size {
            for merge_y in 0..config.merge_size {
                for merge_x in 0..config.merge_size {
                    let patch_y = block_y * config.merge_size + merge_y;
                    let patch_x = block_x * config.merge_size + merge_x;
                    for channel in 0..3 {
                        for _temporal in 0..config.temporal_patch_size {
                            for y in 0..config.patch_size {
                                for x in 0..config.patch_size {
                                    let pixel = rgb.get_pixel(
                                        (patch_x * config.patch_size + x)
                                            as u32,
                                        (patch_y * config.patch_size + y)
                                            as u32,
                                    );
                                    let scaled =
                                        f32::from(pixel[channel]) / 255.0;
                                    values.push(
                                        (scaled - config.image_mean[channel])
                                            / config.image_std[channel],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(VisionInput {
        values,
        patches,
        patch_width,
        grid_thw: [1, grid_h as i64, grid_w as i64],
    })
}

fn resized_dimensions(
    width: u32,
    height: u32,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(u32, u32)> {
    let factor = u32::try_from(factor)
        .context("GLM OCR resize factor does not fit into u32")?;
    if factor == 0 {
        bail!("GLM OCR resize factor must be greater than zero");
    }
    if width.max(height) / width.min(height) > 200 {
        bail!("GLM OCR input image aspect ratio is too large");
    }

    let mut resized_width = round_to_factor(width, factor).max(factor);
    let mut resized_height = round_to_factor(height, factor).max(factor);
    let pixels =
        usize::try_from(resized_width)? * usize::try_from(resized_height)?;
    if pixels > max_pixels {
        let scale = (max_pixels as f64 / pixels as f64).sqrt();
        resized_width =
            floor_to_factor((f64::from(resized_width) * scale) as u32, factor)
                .max(factor);
        resized_height =
            floor_to_factor((f64::from(resized_height) * scale) as u32, factor)
                .max(factor);
        return Ok((resized_width, resized_height));
    }
    if pixels < min_pixels {
        let scale = (min_pixels as f64 / pixels as f64).sqrt();
        resized_width =
            ceil_to_factor((f64::from(resized_width) * scale) as u32, factor)
                .max(factor);
        resized_height =
            ceil_to_factor((f64::from(resized_height) * scale) as u32, factor)
                .max(factor);
    }

    Ok((resized_width, resized_height))
}

fn round_to_factor(value: u32, factor: u32) -> u32 {
    ((value + factor / 2) / factor) * factor
}

fn ceil_to_factor(value: u32, factor: u32) -> u32 {
    value.div_ceil(factor) * factor
}

fn floor_to_factor(value: u32, factor: u32) -> u32 {
    (value / factor) * factor
}

#[allow(dead_code)]
fn build_ocr_prompt(
    tokenizer: &Tokenizer,
    config: &GlmOcrConfig,
    input: &VisionInput,
) -> Result<GlmOcrPrompt> {
    let merge_area = config.vision_config.spatial_merge_size.pow(2);
    if merge_area == 0 || !input.patches.is_multiple_of(merge_area) {
        bail!("GLM OCR image patches must divide by merge area");
    }

    let image_tokens = input.patches / merge_area;
    let prompt_suffix =
        if std::env::var_os("AKUNA_OCR_OFFICIAL_PROMPT").is_some() {
            "<|end_of_image|>Text Recognition:<|assistant|>\n"
        } else {
            GLM_OCR_PROMPT_SUFFIX
        };
    let prompt_text = format!(
        "{}{}{}",
        GLM_OCR_PROMPT_PREFIX,
        GLM_OCR_IMAGE_PLACEHOLDER.repeat(image_tokens),
        prompt_suffix,
    );
    let input_ids = encode_prompt_segment(tokenizer, &prompt_text)?;
    let image_token_start = input_ids
        .iter()
        .position(|token_id| *token_id == config.image_token_id)
        .context("GLM OCR prompt did not encode image placeholders")?;
    let image_token_end = input_ids[image_token_start..]
        .iter()
        .position(|token_id| *token_id != config.image_token_id)
        .map(|offset| image_token_start + offset)
        .unwrap_or(input_ids.len());
    if image_token_end - image_token_start != image_tokens {
        bail!(
            "GLM OCR encoded image token count does not match vision features"
        );
    }

    let mut mm_token_type_ids = vec![0; input_ids.len()];
    mm_token_type_ids[image_token_start..image_token_end].fill(1);
    let position_ids = build_single_image_position_ids(
        input_ids.len(),
        image_token_start..image_token_end,
        input.grid_thw,
        config.vision_config.spatial_merge_size,
    )?;

    Ok(GlmOcrPrompt {
        input_ids,
        mm_token_type_ids,
        image_token_range: image_token_start..image_token_end,
        position_ids,
    })
}

#[allow(dead_code)]
fn encode_prompt_segment(
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?
        .get_ids()
        .to_vec())
}

fn next_decode_position(position_ids: &[Vec<i64>; 3]) -> i64 {
    position_ids
        .iter()
        .flat_map(|axis| axis.iter())
        .copied()
        .max()
        .unwrap_or_default()
        + 1
}

#[allow(dead_code)]
fn build_single_image_position_ids(
    sequence_length: usize,
    image_token_range: std::ops::Range<usize>,
    grid_thw: [i64; 3],
    merge_size: usize,
) -> Result<[Vec<i64>; 3]> {
    let merge_size = i64::try_from(merge_size)?;
    let [grid_t, grid_h, grid_w] = grid_thw;
    if merge_size == 0
        || [grid_t, grid_h, grid_w]
            .iter()
            .any(|dimension| *dimension <= 0)
    {
        bail!("GLM OCR image grid and merge size must be positive");
    }
    if image_token_range.end > sequence_length {
        bail!("GLM OCR image token range exceeds sequence length");
    }
    if grid_h % merge_size != 0 || grid_w % merge_size != 0 {
        bail!("GLM OCR image grid must divide by merge size");
    }

    let grid_h = grid_h / merge_size;
    let grid_w = grid_w / merge_size;
    let expected_image_tokens = usize::try_from(grid_t * grid_h * grid_w)?;
    if image_token_range.len() != expected_image_tokens {
        bail!("GLM OCR image token count does not match grid");
    }

    let mut position_ids = [
        Vec::with_capacity(sequence_length),
        Vec::with_capacity(sequence_length),
        Vec::with_capacity(sequence_length),
    ];
    let image_start = i64::try_from(image_token_range.start)?;

    for index in 0..image_token_range.start {
        let position = i64::try_from(index)?;
        for axis in &mut position_ids {
            axis.push(position);
        }
    }
    for temporal in 0..grid_t {
        for height in 0..grid_h {
            for width in 0..grid_w {
                position_ids[0].push(image_start + temporal);
                position_ids[1].push(image_start + height);
                position_ids[2].push(image_start + width);
            }
        }
    }

    let text_start = image_start + grid_t.max(grid_h).max(grid_w);
    for offset in 0..sequence_length - image_token_range.end {
        let position = text_start + i64::try_from(offset)?;
        for axis in &mut position_ids {
            axis.push(position);
        }
    }

    Ok(position_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;
    use burn_wgpu::{Wgpu, WgpuDevice};
    use image::{ImageBuffer, ImageFormat, Rgb};
    use std::io::Cursor;
    use tokenizers::{AddedToken, models::wordlevel::WordLevel};

    #[test]
    fn variant_repo_id_returns_official_glm_ocr() {
        assert_eq!(GlmOcrVariant::OnnxCommunity.repo_id(), "zai-org/GLM-OCR");
    }

    #[test]
    fn config_validation_accepts_matching_preprocessor() {
        GlmOcrConfig::for_test()
            .validate(&PreprocessorConfig::for_test())
            .expect("test config should validate");
    }

    #[test]
    fn image_preprocess_returns_glm_patch_matrix() {
        let image = ImageBuffer::from_pixel(2, 2, Rgb([255, 0, 128]));
        let input = preprocess_image(
            &DynamicImage::ImageRgb8(image),
            &PreprocessorConfig::for_test(),
        )
        .expect("image should preprocess");

        assert_eq!(input.patches, 64);
        assert_eq!(input.patch_width, 3 * 2 * 14 * 14);
        assert_eq!(input.grid_thw, [1, 8, 8]);
        assert_eq!(input.values.len(), 64 * 3 * 2 * 14 * 14);
    }

    #[test]
    fn ocr_prompt_expands_image_tokens_and_marks_types() {
        let tokenizer = prompt_test_tokenizer();
        let config = GlmOcrConfig::for_test();
        let input = VisionInput {
            values: Vec::new(),
            patches: 16,
            patch_width: 0,
            grid_thw: [1, 4, 4],
        };

        let prompt = build_ocr_prompt(&tokenizer, &config, &input)
            .expect("prompt should build");

        assert_eq!(prompt.image_token_range.len(), 4);
        assert_eq!(
            prompt.input_ids[prompt.image_token_range.clone()],
            [config.image_token_id; 4],
        );
        assert_eq!(
            prompt.mm_token_type_ids[prompt.image_token_range.clone()],
            [1; 4],
        );
        assert_eq!(
            prompt
                .mm_token_type_ids
                .iter()
                .filter(|value| **value == 1)
                .count(),
            4,
        );
        let start = i64::try_from(prompt.image_token_range.start).unwrap();
        assert_eq!(
            prompt.position_ids[0][prompt.image_token_range.clone()],
            [start, start, start, start],
        );
        assert_eq!(
            prompt.position_ids[1][prompt.image_token_range.clone()],
            [start, start, start + 1, start + 1],
        );
        assert_eq!(
            prompt.position_ids[2][prompt.image_token_range.clone()],
            [start, start + 1, start, start + 1],
        );
    }

    #[test]
    fn mrope_position_ids_match_single_image_grid() {
        let position_ids =
            build_single_image_position_ids(8, 2..6, [1, 4, 4], 2)
                .expect("position ids should build");

        assert_eq!(position_ids[0], vec![0, 1, 2, 2, 2, 2, 4, 5]);
        assert_eq!(position_ids[1], vec![0, 1, 2, 2, 3, 3, 4, 5]);
        assert_eq!(position_ids[2], vec![0, 1, 2, 3, 2, 3, 4, 5]);
    }

    #[test]
    fn vision_patch_embed_forward_returns_hidden_states() {
        let device = Default::default();
        let image = ImageBuffer::from_pixel(2, 2, Rgb([255, 0, 128]));
        let input = preprocess_image(
            &DynamicImage::ImageRgb8(image),
            &PreprocessorConfig::for_test(),
        )
        .expect("image should preprocess");
        let patch_embed =
            GlmVisionPatchEmbed::<burn_cpu::Cpu>::new_for_test(&device);

        let output = patch_embed.forward(&input, &device);

        assert_eq!(
            output.dims(),
            [input.patches, GLM_OCR_VISION_DIMS.hidden_size]
        );
    }

    #[test]
    fn rms_norm_forward_preserves_shape() {
        let device = Default::default();
        let x = Tensor::<burn_cpu::Cpu, 2>::ones([2, 4], &device);
        let norm = GlmRmsNorm::<burn_cpu::Cpu>::new_for_test(4, &device);

        let output = norm.forward(x);

        assert_eq!(output.dims(), [2, 4]);
    }

    #[test]
    fn vision_block_forward_preserves_hidden_shape() {
        let device = Default::default();
        let patches = 4;
        let block = GlmVisionBlock::<burn_cpu::Cpu>::new_for_test();
        let x = Tensor::<burn_cpu::Cpu, 2>::ones(
            [patches, GLM_OCR_VISION_DIMS.hidden_size],
            &device,
        );

        let output = block.forward(x, [1, 2, 2]);

        assert_eq!(output.dims(), [patches, GLM_OCR_VISION_DIMS.hidden_size]);
    }

    #[test]
    fn full_vision_encoder_forward_returns_image_features() {
        let device = Default::default();
        let input = VisionInput {
            values: vec![0.0; 4 * GLM_OCR_VISION_DIMS.patch_width()],
            patches: 4,
            patch_width: GLM_OCR_VISION_DIMS.patch_width(),
            grid_thw: [1, 2, 2],
        };
        let encoder =
            GlmVisionEncoder::<burn_cpu::Cpu>::new_for_test(2, &device);

        let output = encoder.forward(&input, &device);

        assert_eq!(output.dims(), [1, GLM_OCR_VISION_DIMS.output_hidden_size]);
    }

    #[test]
    fn text_decoder_layer_forward_preserves_hidden_shape() {
        let device = Default::default();
        let seq = 4;
        let layer = GlmTextDecoderLayer::<burn_cpu::Cpu>::new_for_test();
        let x = Tensor::<burn_cpu::Cpu, 2>::ones(
            [seq, GLM_OCR_TEXT_DIMS.hidden_size],
            &device,
        );

        let position_ids =
            [vec![0, 1, 2, 3], vec![0, 1, 2, 3], vec![0, 1, 2, 3]];
        let output = layer.forward(x, &position_ids);

        assert_eq!(output.dims(), [seq, GLM_OCR_TEXT_DIMS.hidden_size]);
    }

    #[ignore = "full vocab matmul is slow on CPU"]
    #[test]
    fn text_model_forward_returns_vocab_logits() {
        let device = Default::default();
        let token_ids = [1u32, 2, 3];
        let model = GlmTextModel::<burn_cpu::Cpu>::new_for_test(1, &device);
        let embeddings = model.embed_token_ids(token_ids.as_slice(), &device);

        let position_ids = [vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
        let output = model.forward_embeddings(embeddings, &position_ids);

        assert_eq!(
            output.dims(),
            [token_ids.len(), GLM_OCR_TEXT_DIMS.vocab_size]
        );
    }

    #[test]
    fn image_preprocess_rejects_invalid_bytes() {
        let model =
            GlmOcrModel::<Wgpu>::new_for_test(GlmOcrVariant::OnnxCommunity);
        let device = WgpuDevice::default();
        let error = model
            .extract_bytes(b"not an image".as_slice(), &device)
            .expect_err("invalid image should fail");

        assert!(error.to_string().contains("decode OCR input image"));
    }

    #[test]
    fn official_weight_shapes_match_glm_ocr_dimensions() {
        let shapes = required_weight_shapes();

        assert_eq!(shapes.len(), 24 * 14 + 16 * 10 + 14);
        assert_eq!(shapes[0].1, vec![1024, 3, 2, 14, 14]);
        assert!(shapes.iter().any(|(name, shape)| {
            name == "model.language_model.layers.15.self_attn.o_proj.weight"
                && shape == &vec![1_536, 2_048]
        }));
        assert_eq!(
            shapes.last().expect("shapes should not be empty").1,
            vec![1_536, 4_608]
        );
    }

    #[test]
    fn read_bf16_tensor_values_converts_to_f32() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&bf16::from_f32(1.5).to_bits().to_le_bytes());
        bytes.extend_from_slice(&bf16::from_f32(-2.0).to_bits().to_le_bytes());
        let serialized = safetensors::serialize(
            [(
                "tensor",
                safetensors::tensor::TensorView::new(
                    Dtype::BF16,
                    vec![2],
                    &bytes,
                )
                .expect("tensor view should build"),
            )],
            None,
        )
        .expect("safetensors should serialize");
        let tensors = SafeTensors::deserialize(&serialized)
            .expect("safetensors should deserialize");

        let shape = vec![2];
        let values = read_bf16_tensor_values(&tensors, "tensor", &shape)
            .expect("tensor should read");

        assert_eq!(values, vec![1.5, -2.0]);
    }

    #[test]
    fn patch_embed_weight_transposes_official_layout() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let transposed = transpose_flattened_patch_weight(&values, 2, 3);

        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn matrix_weight_transpose_reuses_official_layout() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let transposed = transpose_flattened_matrix(&values, 2, 3);

        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[ignore = "downloads GLM OCR safetensors"]
    #[tokio::test]
    async fn live_glm_ocr_downloads_model_files() {
        let device = WgpuDevice::default();
        let model =
            load_glm_ocr::<Wgpu>(&device, GlmOcrVariant::OnnxCommunity, None)
                .await
                .expect("GLM OCR files should load");
        let files = model.files.expect("model should record downloaded files");

        assert!(files.weights_path.exists());
    }

    #[ignore = "downloads and constructs GLM OCR safetensors"]
    #[tokio::test]
    async fn live_glm_ocr_constructs_native_weighted_model() {
        let device = WgpuDevice::default();
        let model =
            load_glm_ocr::<Wgpu>(&device, GlmOcrVariant::OnnxCommunity, None)
                .await
                .expect("GLM OCR weighted model should construct");

        assert_eq!(model.vision.blocks.len(), GLM_OCR_VISION_DIMS.layers);
        assert_eq!(model.text.layers.len(), GLM_OCR_TEXT_DIMS.layers);
    }

    #[test]
    fn last_argmax_returns_max_token_from_last_row() {
        let device = Default::default();
        let logits = Tensor::<burn_cpu::Cpu, 2>::from_data(
            TensorData::new(vec![0.0, 3.0, 1.0, 5.0, 4.0, 6.0], [2, 3]),
            &device,
        );

        let token = last_argmax(logits).expect("argmax should decode");

        assert_eq!(token, 2);
    }

    #[ignore = "downloads GLM OCR safetensors and runs native inference"]
    #[tokio::test]
    async fn live_glm_ocr_extracts_generated_tokens_or_shape_runtime() {
        let image = ImageBuffer::from_pixel(2, 2, Rgb([255, 255, 255]));
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .expect("test image should encode");
        let device = WgpuDevice::default();
        let model =
            load_glm_ocr::<Wgpu>(&device, GlmOcrVariant::OnnxCommunity, None)
                .await
                .expect("GLM OCR weighted model should construct");

        match model.extract_bytes(bytes.get_ref(), &device) {
            Ok(text) => assert!(!text.is_empty()),
            Err(error) => assert!(
                error
                    .to_string()
                    .contains("native tensor shape unsupported"),
                "unexpected OCR runtime error: {error}"
            ),
        }
    }

    fn prompt_test_tokenizer() -> Tokenizer {
        let vocabulary = AHashMap::from([
            ("[gMASK]".to_string(), 59248),
            ("<sop>".to_string(), 59250),
            ("<|user|>".to_string(), 59253),
            ("<|assistant|>".to_string(), 59254),
            ("<|begin_of_image|>".to_string(), 59256),
            ("<|end_of_image|>".to_string(), 59257),
            (GLM_OCR_IMAGE_PLACEHOLDER.to_string(), 59280),
            (GLM_OCR_PROMPT_TEXT.to_string(), 7),
            ("<think></think>".to_string(), 8),
            ("[UNK]".to_string(), 0u32),
        ]);
        let model = WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel tokenizer should build");

        let mut tokenizer = Tokenizer::new(model);
        tokenizer.add_special_tokens(&[
            AddedToken::from("[gMASK]", true),
            AddedToken::from("<sop>", true),
            AddedToken::from("<|user|>", true),
            AddedToken::from("<|assistant|>", true),
            AddedToken::from("<|begin_of_image|>", true),
            AddedToken::from("<|end_of_image|>", true),
            AddedToken::from(GLM_OCR_IMAGE_PLACEHOLDER, true),
            AddedToken::from(GLM_OCR_PROMPT_TEXT, true),
            AddedToken::from("<think></think>", true),
        ]);

        tokenizer
    }
}
