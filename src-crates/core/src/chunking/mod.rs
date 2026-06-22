//! Text chunking strategies built on memchunk.
//!
//! Grammar-aware chunking via tree-sitter is available behind the
//! `tree-sitter` feature flag.
//!
//! # Example
//!
//! ```rust
//! use akuna_core::chunking::chunk_text;
//!
//! let content = "Hello world. Rust is great.";
//! let chunks = chunk_text(None, content, None);
//! assert!(!chunks.is_empty());
//! ```

#[cfg(feature = "tree-sitter")]
mod tree_sitter;

use std::collections::HashMap;

/// Text chunking configuration.
#[derive(Default)]
pub struct ChunkingConfig {
    /// Optional target chunk size in bytes.
    pub target_size: Option<usize>,
    /// Optional delimiter bytes keyed by file extension.
    pub delimiters_by_ft: HashMap<String, Vec<u8>>,
    /// Optional delimiter bytes for chunk boundary selection.
    pub delimiters: Option<Vec<u8>>,
}

/// Chunks text content using optional chunking configuration.
///
/// `file_extension` selects file-type-specific delimiters and tree-sitter
/// grammar; it should not include the leading dot.
pub fn chunk_text<'a>(
    config: Option<&ChunkingConfig>,
    content: &'a str,
    file_extension: Option<&str>,
) -> Vec<&'a str> {
    let default_config = ChunkingConfig::default();
    let config = config.unwrap_or(&default_config);
    let delimiters = delimiters_from_config(config, file_extension);
    let target_size = target_size_from_config(config);

    chunk_with_strategy(content, delimiters, target_size, file_extension)
}

/// Resolves configured chunk size, falling back to memchunk default.
fn target_size_from_config(config: &ChunkingConfig) -> usize {
    config
        .target_size
        .filter(|size| *size > 0)
        .unwrap_or(memchunk::DEFAULT_TARGET_SIZE)
}

#[cfg(feature = "tree-sitter")]
fn chunk_with_strategy<'a>(
    content: &'a str,
    delimiters: Option<&[u8]>,
    target_size: usize,
    file_extension: Option<&str>,
) -> Vec<&'a str> {
    tree_sitter::chunk_content(content, delimiters, target_size, file_extension)
        .unwrap_or_else(|| {
            chunk_with_delimiters(content, delimiters, target_size)
        })
}

#[cfg(not(feature = "tree-sitter"))]
fn chunk_with_strategy<'a>(
    content: &'a str,
    delimiters: Option<&[u8]>,
    target_size: usize,
    _file_extension: Option<&str>,
) -> Vec<&'a str> {
    chunk_with_delimiters(content, delimiters, target_size)
}

pub(crate) fn chunk_with_delimiters<'a>(
    content: &'a str,
    delimiters: Option<&[u8]>,
    target_size: usize,
) -> Vec<&'a str> {
    let chunker = memchunk::chunk(content.as_bytes())
        .size(target_size)
        .consecutive()
        .forward_fallback();

    let chunker = if let Some(delimiters) = delimiters {
        chunker.delimiters(delimiters)
    } else {
        chunker
    };

    let mut chunks = Vec::new();
    let mut start = 0;

    for chunk in chunker {
        let Some(end) = chunk_end(content, chunk) else {
            continue;
        };

        if start < end {
            chunks.push(&content[start..end]);
        }

        start = end;
    }

    chunks
}

/// Returns valid string boundary for byte chunk end.
fn chunk_end(content: &str, chunk: &[u8]) -> Option<usize> {
    let content_start = content.as_ptr() as usize;
    let chunk_start = chunk.as_ptr() as usize;
    let start = chunk_start.checked_sub(content_start)?;

    Some(next_char_boundary(content, start + chunk.len()))
}

/// Moves index forward to valid UTF-8 boundary.
fn next_char_boundary(content: &str, index: usize) -> usize {
    let mut index = index.min(content.len());

    while index < content.len() && !content.is_char_boundary(index) {
        index += 1;
    }

    index
}

fn delimiters_from_config<'a>(
    config: &'a ChunkingConfig,
    file_extension: Option<&str>,
) -> Option<&'a [u8]> {
    file_extension
        .and_then(|file_extension| config.delimiters_by_ft.get(file_extension))
        .map(Vec::as_slice)
        .or(config.delimiters.as_deref())
}
