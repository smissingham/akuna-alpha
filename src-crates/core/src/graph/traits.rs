//! Graph storage traits.

use crate::GraphError;
use crate::graph::{
    storage,
    structs::{GraphEdge, GraphNode},
};

/// Typed graph storage context.
pub trait GraphDbContext {
    /// Storage mode backing this graph context.
    fn storage(&self) -> &storage::GraphStorage;

    /// Stores a graph node.
    fn put_node(&self, node: &GraphNode) -> Result<(), GraphError>;

    /// Reads a graph node by id and labels.
    fn get_node(
        &self,
        labels: &[&str],
        id: impl AsRef<str>,
    ) -> Result<Option<GraphNode>, GraphError>;

    /// Deletes an existing graph node by id and labels.
    fn delete_node(
        &self,
        labels: &[&str],
        id: impl AsRef<str>,
    ) -> Result<(), GraphError>;

    /// Stores a graph edge between existing node ids.
    fn put_edge(&self, edge: &GraphEdge) -> Result<(), GraphError>;

    /// Deletes an existing graph edge by structural identity.
    fn delete_edge(&self, edge: &GraphEdge) -> Result<(), GraphError>;

    /// Consumes the context and removes any persisted storage it owns.
    fn destroy(self) -> Result<(), GraphError>
    where
        Self: Sized;
}
