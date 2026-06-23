#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PpOcrV6Tier {
    Tiny,
    Small,
    Medium,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpOcrModelSpec {
    pub(crate) repo_id: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) static_shape: [usize; 4],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpOcrDetectorConfig {
    pub(crate) spec: PpOcrModelSpec,
    pub(crate) limit_side_len: u32,
    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
    pub(crate) db_thresh: f32,
    pub(crate) db_box_thresh: f32,
    pub(crate) db_unclip_ratio: f32,
    pub(crate) max_candidates: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpOcrRecognizerConfig {
    pub(crate) spec: PpOcrModelSpec,
    pub(crate) mean: [f32; 3],
    pub(crate) std: [f32; 3],
    pub(crate) num_classes: usize,
}

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub(crate) fn detector_config(tier: PpOcrV6Tier) -> PpOcrDetectorConfig {
    let (repo_id, revision, db_box_thresh) = match tier {
        PpOcrV6Tier::Tiny => (
            "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
            "2ba1506c0380b8f0b03dd142459aac66d4421f6c",
            0.4,
        ),
        PpOcrV6Tier::Small => (
            "PaddlePaddle/PP-OCRv6_small_det_onnx",
            "28fe5895c24fd108c19eb3e8479f4ab385fbfc62",
            0.45,
        ),
        PpOcrV6Tier::Medium => (
            "PaddlePaddle/PP-OCRv6_medium_det_onnx",
            "61323801669c338b7891481ec7bac61ce31b576a",
            0.45,
        ),
    };

    PpOcrDetectorConfig {
        spec: PpOcrModelSpec {
            repo_id,
            revision,
            static_shape: [1, 3, 960, 960],
        },
        limit_side_len: 960,
        mean: MEAN,
        std: STD,
        db_thresh: 0.2,
        db_box_thresh,
        db_unclip_ratio: 1.4,
        max_candidates: 3000,
    }
}

pub(crate) fn recognizer_config(tier: PpOcrV6Tier) -> PpOcrRecognizerConfig {
    let (repo_id, revision, num_classes) = match tier {
        PpOcrV6Tier::Tiny => (
            "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
            "2612ab37152ae0a677521bae4e1e3d4fb4cf7c30",
            6906,
        ),
        PpOcrV6Tier::Small => (
            "PaddlePaddle/PP-OCRv6_small_rec_onnx",
            "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
            18710,
        ),
        PpOcrV6Tier::Medium => (
            "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
            "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
            18710,
        ),
    };

    PpOcrRecognizerConfig {
        spec: PpOcrModelSpec {
            repo_id,
            revision,
            static_shape: [1, 3, 48, 320],
        },
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        num_classes,
    }
}
