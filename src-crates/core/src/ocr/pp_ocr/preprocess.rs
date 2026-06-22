use anyhow::Result;
use image::{DynamicImage, GenericImageView, imageops::FilterType};

use crate::ocr::pp_ocr::spec::{PpOcrDetectorConfig, PpOcrRecognizerConfig};

#[derive(Debug)]
pub(crate) struct PpOcrInput {
    pub(crate) values: Vec<f32>,
    pub(crate) channels: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) original_width: u32,
    pub(crate) original_height: u32,
    pub(crate) resized_width: u32,
    pub(crate) resized_height: u32,
}

pub(crate) fn preprocess_detector(
    image: &DynamicImage,
    config: &PpOcrDetectorConfig,
) -> Result<PpOcrInput> {
    let (_, _, target_height, target_width) =
        static_shape(config.spec.static_shape);
    let (original_width, original_height) = image.dimensions();
    let scale =
        resize_scale(original_width, original_height, config.limit_side_len);
    let resized_width = ((original_width as f32 * scale).round() as u32)
        .clamp(1, target_width as u32);
    let resized_height = ((original_height as f32 * scale).round() as u32)
        .clamp(1, target_height as u32);
    let resized =
        image.resize_exact(resized_width, resized_height, FilterType::Triangle);
    let values = normalized_bgr_nchw(
        &resized,
        target_width,
        target_height,
        config.mean,
        config.std,
        true,
    );

    Ok(PpOcrInput {
        values,
        channels: 3,
        height: target_height,
        width: target_width,
        original_width,
        original_height,
        resized_width,
        resized_height,
    })
}

pub(crate) fn preprocess_recognizer(
    image: &DynamicImage,
    config: &PpOcrRecognizerConfig,
) -> Result<PpOcrInput> {
    let (_, _, target_height, target_width) =
        static_shape(config.spec.static_shape);
    let (original_width, original_height) = image.dimensions();
    let scale = target_height as f32 / original_height.max(1) as f32;
    let resized_width = ((original_width as f32 * scale).ceil() as u32)
        .clamp(1, target_width as u32);
    let resized = image.resize_exact(
        resized_width,
        target_height as u32,
        FilterType::Triangle,
    );
    let values = normalized_bgr_nchw(
        &resized,
        target_width,
        target_height,
        config.mean,
        config.std,
        true,
    );

    Ok(PpOcrInput {
        values,
        channels: 3,
        height: target_height,
        width: target_width,
        original_width,
        original_height,
        resized_width,
        resized_height: target_height as u32,
    })
}

fn resize_scale(width: u32, height: u32, limit_side_len: u32) -> f32 {
    let longest = width.max(height).max(1) as f32;
    limit_side_len as f32 / longest
}

fn normalized_bgr_nchw(
    image: &DynamicImage,
    target_width: usize,
    target_height: usize,
    mean: [f32; 3],
    std: [f32; 3],
    use_bgr: bool,
) -> Vec<f32> {
    let rgb = image.to_rgb8();
    let mut values = vec![0.0; 3 * target_height * target_width];
    let copy_width = rgb.width().min(target_width as u32) as usize;
    let copy_height = rgb.height().min(target_height as u32) as usize;

    for y in 0..copy_height {
        for x in 0..copy_width {
            let pixel = rgb.get_pixel(x as u32, y as u32).0;
            let values_for_pixel = if use_bgr {
                [pixel[2], pixel[1], pixel[0]]
            } else {
                [pixel[0], pixel[1], pixel[2]]
            };
            for channel in 0..3 {
                let index = channel * target_height * target_width
                    + y * target_width
                    + x;
                values[index] = (values_for_pixel[channel] as f32 / 255.0
                    - mean[channel])
                    / std[channel];
            }
        }
    }

    values
}

fn static_shape(shape: [usize; 4]) -> (usize, usize, usize, usize) {
    (shape[0], shape[1], shape[2], shape[3])
}
