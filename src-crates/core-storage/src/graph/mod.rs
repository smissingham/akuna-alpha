//! Graph storage domain types, traits, and backend implementations.
//!
//! Backend implementations live under [`backend`] (kept crate-private) and
//! are intentionally kept out of the public surface. Consumers should depend
//! on the traits and types exposed here and use [`open_context`] /
//! [`in_memory_context`] to obtain a [`Box<dyn GraphDbContext>`].

mod backend;
mod error;

use serde::{Deserialize, Serialize};

pub use error::*;

/// Backend-neutral graph storage mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphStorage {
    /// Data is kept in memory only.
    InMemory,
    /// Data is persisted at the given path.
    Persistent(std::path::PathBuf),
}

/// Flexible relationship between knowledge graph nodes.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GraphEdge {
    /// Source node labels.
    pub source_labels: Vec<String>,
    /// Stable source node identifier within its labels.
    pub source: String,
    /// Relationship type from source to target.
    pub predicate: String,
    /// Stable target node identifier within its labels.
    pub target: String,
    /// Target node labels.
    pub target_labels: Vec<String>,
}

/// Flexible knowledge graph concept with caller-defined labels and metadata.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GraphNode {
    /// Stable concept identifier within its labels.
    pub id: String,
    /// Graph labels for this concept.
    pub labels: Vec<String>,
    /// Human-readable concept name.
    pub name: String,
    /// Optional concept description.
    pub description: Option<String>,
    /// Serializable concept metadata.
    pub metadata: Option<serde_json::Value>,
}

/// Node search request sent to graph storage.
pub struct GraphNodeSearchQuery {
    /// Optional label to search within.
    pub label: Option<String>,
    /// Search text.
    pub query: String,
    /// Maximum result count.
    pub limit: usize,
}

/// Ranked graph node search result.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct GraphNodeSearchResult {
    /// Matching graph node.
    pub node: GraphNode,
    /// Fused relevance score.
    pub score: f64,
}

/// Typed graph storage context.
pub trait GraphDbContext: Send + Sync {
    /// Storage mode backing this graph context.
    fn storage(&self) -> &GraphStorage;

    /// Stores a graph node.
    fn put_node(
        &self,
        node: &GraphNode,
        search_embedding: &[f32],
    ) -> Result<(), GraphError>;

    /// Reads a graph node by id and labels.
    fn get_node(
        &self,
        labels: &[&str],
        id: &str,
    ) -> Result<Option<GraphNode>, GraphError>;

    /// Deletes an existing graph node by id and labels.
    fn delete_node(&self, labels: &[&str], id: &str) -> Result<(), GraphError>;

    /// Searches graph nodes by hybrid text and vector relevance.
    fn search_nodes(
        &self,
        query: &GraphNodeSearchQuery,
        query_embedding: &[f32],
    ) -> Result<Vec<GraphNodeSearchResult>, GraphError>;

    /// Stores a graph edge between existing node ids.
    fn put_edge(&self, edge: &GraphEdge) -> Result<(), GraphError>;

    /// Deletes an existing graph edge by structural identity.
    fn delete_edge(&self, edge: &GraphEdge) -> Result<(), GraphError>;

    /// Consumes the context and removes any persisted storage it owns.
    fn destroy(self: Box<Self>) -> Result<(), GraphError>;
}

/// Builds the search text used to index a node for hybrid retrieval.
pub fn search_text(node: &GraphNode) -> String {
    let mut parts = vec![node.name.as_str()];
    if let Some(description) = node.description.as_deref() {
        parts.push(description);
    }

    parts.join("\n")
}

/// Opens a persistent graph storage context at the given path.
///
/// Backend selection is currently fixed but will become configurable via
/// [`StorageConfig`] in a future release. Callers should depend only on the
/// [`GraphDbContext`] trait returned here.
pub fn open_context(
    path: impl AsRef<std::path::Path>,
) -> Result<Box<dyn GraphDbContext>, GraphError> {
    Ok(Box::new(backend::GrafeoDbContext::new(
        path.as_ref().to_path_buf(),
    )?))
}

/// Creates an in-memory graph storage context suitable for tests and ephemeral use.
pub fn in_memory_context() -> Box<dyn GraphDbContext> {
    Box::new(backend::GrafeoDbContext::new_in_memory())
}
