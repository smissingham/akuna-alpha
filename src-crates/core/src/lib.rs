//! Shared public API's for common actions across the library.

extern crate self as akuna_core;

/// Application tracing helpers.
pub mod tracing;

/// Text chunking APIs.
#[cfg(feature = "chunking")]
pub mod chunking {
    pub use akuna_core_chunking::*;
}

/// File-type detection APIs.
#[cfg(feature = "detection")]
pub mod detection {
    pub use akuna_core_detection::*;
}

/// Text embeddings.
#[cfg(feature = "embedding")]
pub mod embedding {
    pub use akuna_core_embedding::*;
}

/// Text reranking APIs.
#[cfg(feature = "reranking")]
pub mod reranking {
    pub use akuna_core_reranking::*;
}

/// File extraction APIs.
#[cfg(feature = "extraction")]
pub mod extraction {
    pub use akuna_core_extraction::*;
}

/// Graph storage and retrieval APIs.
#[cfg(feature = "storage")]
pub mod storage {
    pub use akuna_core_storage::graph::*;
}
