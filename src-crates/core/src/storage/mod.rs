//! Graph storage and retrieval built on `grafeo`.
//!
//! Provides the backend-neutral [`graph::GraphDbContext`] trait alongside
//! node, edge, and search types. Backend implementations are crate-private
//! and reached through [`graph::open_context`] or [`graph::in_memory_context`].
//!
//! All [`graph`] items are also available directly under `storage::`, e.g.
//! `akuna_core::storage::GraphNode` works the same as
//! `akuna_core::storage::graph::GraphNode`.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core::storage::graph::{
//!     in_memory_context, GraphDbContext, GraphNode,
//! };
//!
//! let ctx = in_memory_context();
//! let node = GraphNode {
//!     id: "rust".to_string(),
//!     labels: vec!["Concept".to_string()],
//!     name: "Rust".to_string(),
//!     description: None,
//!     metadata: None,
//! };
//! ctx.put_node(&node, &[]).expect("node stored");
//! ```

pub mod graph;

pub use graph::*;
