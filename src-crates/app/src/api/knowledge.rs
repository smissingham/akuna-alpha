//! Knowledge service interface and shared I/O types.

use akuna_core::graph::{
    storage::grafeo::GrafeoDbContext,
    structs::{Assertion, Edge, Node, Provenance},
    traits::{GraphDbContext, GraphNode},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::api::error::ServiceError;

/// CRUD actions available through the knowledge API.
#[derive(Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum KnowledgeAction {
    /// Creates a knowledge entity.
    Create,
    /// Reads a knowledge entity.
    Read,
    /// Updates a knowledge entity.
    Update,
    /// Deletes a knowledge entity.
    Delete,
}

/// Knowledge entity types available through the knowledge API.
#[derive(Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum KnowledgeType {
    /// Graph node entity.
    Node,
    /// Graph edge entity.
    Edge,
    /// Assertion node entity.
    Assertion,
    /// Provenance node entity.
    Provenance,
}

/// Knowledge service request shared by API adapters.
pub struct KnowledgeRequest {
    /// CRUD action: create, read, update, or delete.
    pub action: KnowledgeAction,
    /// Knowledge type: node, assertion, provenance, or edge.
    pub knowledge_type: KnowledgeType,
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
        let body = match input.knowledge_type {
            KnowledgeType::Node => {
                node_crud::<Node>(self.graph, input.action, input.body)
            }
            KnowledgeType::Assertion => {
                node_crud::<Assertion>(self.graph, input.action, input.body)
            }
            KnowledgeType::Provenance => {
                node_crud::<Provenance>(self.graph, input.action, input.body)
            }
            KnowledgeType::Edge => {
                edge_crud(self.graph, input.action, input.body)
            }
        }?;

        Ok(KnowledgeResponse { body })
    }
}

fn node_crud<T>(
    graph: &GrafeoDbContext,
    action: KnowledgeAction,
    body: serde_json::Value,
) -> Result<serde_json::Value, ServiceError>
where
    T: GraphNode + DeserializeOwned + Serialize + 'static,
{
    match action {
        KnowledgeAction::Create | KnowledgeAction::Update => {
            let node: T = serde_json::from_value(body)?;
            graph.put_node(&node)?;
            Ok(serde_json::to_value(node)?)
        }
        KnowledgeAction::Read => {
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
        KnowledgeAction::Delete => {
            let read: ReadNode = serde_json::from_value(body)?;
            let labels =
                read.labels.iter().map(String::as_str).collect::<Vec<_>>();
            graph.delete_node::<T>(&labels, read.id)?;
            Ok(serde_json::json!({ "deleted": true }))
        }
    }
}

fn edge_crud(
    graph: &GrafeoDbContext,
    action: KnowledgeAction,
    body: serde_json::Value,
) -> Result<serde_json::Value, ServiceError> {
    match action {
        KnowledgeAction::Create | KnowledgeAction::Update => {
            let edge: Edge = serde_json::from_value(body)?;
            graph.put_edge(&edge)?;
            Ok(serde_json::to_value(edge)?)
        }
        KnowledgeAction::Delete => {
            let edge: Edge = serde_json::from_value(body)?;
            graph.delete_edge(&edge)?;
            Ok(serde_json::json!({ "deleted": true }))
        }
        KnowledgeAction::Read => {
            Err(ServiceError::bad_request("edge read is not available yet"))
        }
    }
}
