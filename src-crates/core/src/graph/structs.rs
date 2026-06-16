use serde::{Deserialize, Serialize};

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
