//! Akuna core knowledge tooling library.
//!
//! Provides optional feature-gated modules for text chunking, file-type
//! detection, embeddings, extraction, reranking, and graph storage.
//! Consumers enable only the features they need.
//!
//! # Modules
//!
//! - [`chunking`] — text chunking (feature `chunking`)
//! - [`detection`] — file-type detection (feature `detection`)
//! - [`embedding`] — text embeddings (feature `embedding`)
//! - [`reranking`] — text reranking (feature `reranking`)
//! - [`extraction`] — file extraction (feature `extraction`)
//! - [`storage`] — graph storage and retrieval (feature `storage`)
//!
//! # Example
//!
//! Enable the `chunking` feature and call a module function:
//!
//! ```no_run
//! use akuna_core::chunking::chunk_text;
//!
//! let chunks = chunk_text(None, "hello\nworld", Some("txt"));
//! ```

/// Text chunking APIs.
#[cfg(feature = "chunking")]
pub mod chunking;

/// File-type detection APIs.
#[cfg(feature = "detection")]
pub mod detection;

/// Text embeddings.
#[cfg(feature = "embedding")]
pub mod embedding;

/// Text reranking APIs.
#[cfg(feature = "reranking")]
pub mod reranking;

/// File extraction APIs.
#[cfg(feature = "extraction")]
pub mod extraction;

/// Graph storage and retrieval APIs.
#[cfg(feature = "storage")]
pub mod storage;
