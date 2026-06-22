//! Akuna core knowledge tooling library.
//!
//! Provides optional feature-gated modules for file-type detection,
//! embeddings, extraction, reranking, and graph storage.
//! Consumers enable only the features they need.
//!
//! # Modules
//!
//! - [`detection`] — file-type detection (feature `detection`)
//! - [`embedding`] — text embeddings (feature `embedding`)
//! - [`ocr`] — image OCR engines (feature `ocr`)
//! - [`reranking`] — text reranking (feature `reranking`)
//! - [`extraction`] — file extraction (feature `extraction`)
//! - [`storage`] — graph storage and retrieval (feature `storage`)
//!
//! # Example
//!
//! Enable the `extraction` feature and call a module function:
//!
//! ```no_run
//! use akuna_core::extraction::extract_text_bytes;
//!
//! # async fn example() -> Result<(), akuna_core::extraction::FileExtractionError> {
//! let text = extract_text_bytes(b"hello\nworld", None).await?;
//! assert_eq!(text, "hello\nworld");
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "extraction")]
mod chunking;

/// File-type detection APIs.
#[cfg(feature = "detection")]
pub mod detection;

/// Text embeddings.
#[cfg(feature = "embedding")]
pub mod embedding;

/// Text reranking APIs.
#[cfg(feature = "reranking")]
pub mod reranking;

/// Image OCR APIs.
#[cfg(feature = "ocr")]
pub mod ocr;

/// Document layout detection APIs.
#[cfg(feature = "detection")]
pub mod layout;

/// File extraction APIs.
#[cfg(feature = "extraction")]
pub mod extraction;

/// Graph storage and retrieval APIs.
#[cfg(feature = "storage")]
pub mod storage;
