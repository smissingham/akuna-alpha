/// One ranked label guess produced by the classifier.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedAlternative {
    /// Human-readable label for the candidate type.
    pub label: String,
    /// Optional MIME type associated with the label.
    pub mime_type: Option<String>,
    /// Model confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Top-level result of classifying a single input.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Human-readable label of the most likely type.
    pub label: String,
    /// Optional MIME type of the most likely type.
    pub mime_type: Option<String>,
    /// Confidence score of the top prediction.
    pub confidence: f32,
    /// Alternative guesses ranked by confidence.
    pub alternatives: Vec<RankedAlternative>,
}
