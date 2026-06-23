use anyhow::{Result, bail};
use image::{DynamicImage, GenericImageView};

use crate::layout::models::pp_doclayout::LayoutDetection;

pub(crate) fn useful_ocr_fragment(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    if text.starts_with("Figure ")
        || text.starts_with("Answer:")
        || text.starts_with("Source:")
        || text.starts_with("Source ")
    {
        return false;
    }

    text.chars()
        .filter(|character| character.is_alphabetic())
        .count()
        >= 2
}

pub(crate) fn text_like_detection(detection: &LayoutDetection) -> bool {
    if detection.score < 0.1 {
        return false;
    }
    matches!(
        detection.label.as_str(),
        "abstract"
            | "aside_text"
            | "content"
            | "doc_title"
            | "footer"
            | "footnote"
            | "header"
            | "paragraph_title"
            | "reference"
            | "reference_content"
            | "text"
            | "vision_footnote"
    )
}

pub(crate) fn crop_text_region(
    image: &DynamicImage,
    bbox: [f32; 4],
) -> Result<DynamicImage> {
    let (image_width, image_height) = image.dimensions();
    let box_width = (bbox[2] - bbox[0]).max(1.0);
    let box_height = (bbox[3] - bbox[1]).max(1.0);
    let pad_x = (box_width * 0.28).clamp(24.0, 220.0);
    let pad_y = (box_height * 0.04).clamp(2.0, 12.0);
    let x1 = (bbox[0] - pad_x).floor().clamp(0.0, image_width as f32) as u32;
    let y1 = (bbox[1] - pad_y).floor().clamp(0.0, image_height as f32) as u32;
    let x2 = (bbox[2] + pad_x).ceil().clamp(0.0, image_width as f32) as u32;
    let y2 = (bbox[3] + pad_y).ceil().clamp(0.0, image_height as f32) as u32;
    if x2 <= x1 || y2 <= y1 {
        bail!("layout crop has invalid bounds");
    }

    Ok(image.crop_imm(x1, y1, x2 - x1, y2 - y1))
}
