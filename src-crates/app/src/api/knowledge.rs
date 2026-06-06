//! Knowledge service interface and shared I/O types.

use akuna_core::graph::{
    storage::grafeo::GrafeoDbContext,
    structs::{Assertion, Edge, Node, Provenance},
    traits::{GraphDbContext, GraphNode},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::api::error::ServiceError;

/// Knowledge service request shared by API adapters.
pub struct KnowledgeRequest {
    /// CRUD action: create, read, update, or delete.
    pub action: String,
    /// Knowledge type: node, assertion, provenance, or edge.
    pub knowledge_type: String,
    /// Request body for selected operation.
    pub body: serde_json::Value,
}

/// Knowledge service response shared by API adapters.
pub struct KnowledgeResponse {
    /// JSON response body.
    pub body: serde_json::Value,
}

/// Input for reading or deleting a graph node.
#[derive(Deserialize, utoipa::ToSchema)]
pub struct ReadNode {
    /// Stable graph node ID.
    pub id: String,
    /// Labels scoping node ID.
    pub labels: Vec<String>,
}

/// Thin knowledge graph interface used by adapters.
pub struct KnowledgeService<'a> {
    graph: &'a GrafeoDbContext,
}

impl<'a> KnowledgeService<'a> {
    /// Builds a knowledge service over graph storage.
    pub fn new(graph: &'a GrafeoDbContext) -> Self {
        Self { graph }
    }

    /// Executes a knowledge CRUD request.
    pub fn execute(
        &self,
        input: KnowledgeRequest,
    ) -> Result<KnowledgeResponse, ServiceError> {
        let body = match (input.action.as_str(), input.knowledge_type.as_str())
        {
            (action, "node") => {
                node_crud::<Node>(self.graph, action, input.body)
            }
            (action, "assertion") => {
                node_crud::<Assertion>(self.graph, action, input.body)
            }
            (action, "provenance") => {
                node_crud::<Provenance>(self.graph, action, input.body)
            }
            (action, "edge") => edge_crud(self.graph, action, input.body),
            _ => Err(ServiceError::bad_request("unknown knowledge type")),
        }?;

        Ok(KnowledgeResponse { body })
    }
}

fn node_crud<T>(
    graph: &GrafeoDbContext,
    action: &str,
    body: serde_json::Value,
) -> Result<serde_json::Value, ServiceError>
where
    T: GraphNode + DeserializeOwned + Serialize + 'static,
{
    match action {
        "create" | "update" => {
            let node: T = serde_json::from_value(body)?;
            graph.put_node(&node)?;
            Ok(serde_json::to_value(node)?)
        }
        "read" => {
            let read: ReadNode = serde_json::from_value(body)?;
            let labels =
                read.labels.iter().map(String::as_str).collect::<Vec<_>>();
            let Some(node) = graph.get_node::<T>(&labels, read.id)? else {
                return Err(ServiceError::not_found(
                    "knowledge entity not found",
                ));
            };

            Ok(serde_json::to_value(node)?)
        }
        "delete" => {
            let read: ReadNode = serde_json::from_value(body)?;
            let labels =
                read.labels.iter().map(String::as_str).collect::<Vec<_>>();
            graph.delete_node::<T>(&labels, read.id)?;
            Ok(serde_json::json!({ "deleted": true }))
        }
        _ => Err(ServiceError::bad_request("unknown knowledge action")),
    }
}

fn edge_crud(
    graph: &GrafeoDbContext,
    action: &str,
    body: serde_json::Value,
) -> Result<serde_json::Value, ServiceError> {
    match action {
        "create" | "update" => {
            let edge: Edge = serde_json::from_value(body)?;
            graph.put_edge(&edge)?;
            Ok(serde_json::to_value(edge)?)
        }
        "delete" => {
            let edge: Edge = serde_json::from_value(body)?;
            graph.delete_edge(&edge)?;
            Ok(serde_json::json!({ "deleted": true }))
        }
        "read" => {
            Err(ServiceError::bad_request("edge read is not available yet"))
        }
        _ => Err(ServiceError::bad_request("unknown knowledge action")),
    }
}
