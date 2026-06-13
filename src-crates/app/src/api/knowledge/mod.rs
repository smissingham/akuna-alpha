//! Knowledge API routes and handlers.

use akuna_core::graph::{
    storage::grafeo::GrafeoDbContext,
    structs::{Edge, Node},
    traits::GraphDbContext,
};
use axum::{
    Json, Router,
    extract::{Path, Query},
    routing::{delete, get, post, put},
};
use serde::{Deserialize, Serialize};

use crate::api::error::{ApiErrorBody, ApiResult, ServiceError};

const GRAPH_DB_NAME: &str = "knowledge";

/// Registers knowledge API routes.
pub(crate) fn router() -> Router {
    Router::new()
        .route("/graph/nodes", post(create_node))
        .route(
            "/graph/nodes/{id}",
            get(read_node).put(update_node).delete(delete_node),
        )
        .route("/graph/edges", post(create_edge))
        .route("/graph/edges", put(update_edge))
        .route("/graph/edges", delete(delete_edge))
}

/// Query parameters for reading or deleting graph nodes.
#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub(crate) struct NodeQuery {
    /// Labels scoping node ID.
    labels: Vec<String>,
}

/// Query parameters identifying a graph edge.
#[derive(Deserialize, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
pub(crate) struct EdgeQuery {
    /// Source node labels.
    source_labels: Vec<String>,
    /// Stable source node identifier within its labels.
    source: String,
    /// Relationship type from source to target.
    predicate: String,
    /// Stable target node identifier within its labels.
    target: String,
    /// Target node labels.
    target_labels: Vec<String>,
}

/// Response returned after successful deletion.
#[derive(Serialize, utoipa::ToSchema)]
pub(crate) struct DeleteResponse {
    /// Whether entity was deleted.
    deleted: bool,
}

#[utoipa::path(
    post,
    path = "/graph/nodes",
    request_body = Node,
    responses(
        (status = 200, description = "Created node", body = Node),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn create_node(Json(node): Json<Node>) -> ApiResult<Node> {
    write_node(node).map(Json).map_err(Into::into)
}

#[utoipa::path(
    get,
    path = "/graph/nodes/{id}",
    params(("id" = String, Path, description = "Stable node ID"), NodeQuery),
    responses(
        (status = 200, description = "Node", body = Node),
        (status = 404, description = "Node not found", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn read_node(
    Path(id): Path<String>,
    Query(query): Query<NodeQuery>,
) -> ApiResult<Node> {
    read_graph_node(id, query).map(Json).map_err(Into::into)
}

#[utoipa::path(
    put,
    path = "/graph/nodes/{id}",
    params(("id" = String, Path, description = "Stable node ID")),
    request_body = Node,
    responses(
        (status = 200, description = "Updated node", body = Node),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn update_node(
    Path(id): Path<String>,
    Json(node): Json<Node>,
) -> ApiResult<Node> {
    if node.id != id {
        return Err(
            ServiceError::bad_request("path id must match node id").into()
        );
    }

    write_node(node).map(Json).map_err(Into::into)
}

#[utoipa::path(
    delete,
    path = "/graph/nodes/{id}",
    params(("id" = String, Path, description = "Stable node ID"), NodeQuery),
    responses(
        (status = 200, description = "Deleted node", body = DeleteResponse),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn delete_node(
    Path(id): Path<String>,
    Query(query): Query<NodeQuery>,
) -> ApiResult<DeleteResponse> {
    delete_graph_node(id, query).map(Json).map_err(Into::into)
}

#[utoipa::path(
    post,
    path = "/graph/edges",
    request_body = Edge,
    responses(
        (status = 200, description = "Created edge", body = Edge),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn create_edge(Json(edge): Json<Edge>) -> ApiResult<Edge> {
    write_edge(edge).map(Json).map_err(Into::into)
}

#[utoipa::path(
    put,
    path = "/graph/edges",
    request_body = Edge,
    responses(
        (status = 200, description = "Updated edge", body = Edge),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn update_edge(Json(edge): Json<Edge>) -> ApiResult<Edge> {
    write_edge(edge).map(Json).map_err(Into::into)
}

#[utoipa::path(
    delete,
    path = "/graph/edges",
    params(EdgeQuery),
    responses(
        (status = 200, description = "Deleted edge", body = DeleteResponse),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
pub(crate) async fn delete_edge(
    Query(query): Query<EdgeQuery>,
) -> ApiResult<DeleteResponse> {
    let graph = graph()?;
    graph
        .delete_edge(&Edge {
            source_labels: query.source_labels,
            source: query.source,
            predicate: query.predicate,
            target: query.target,
            target_labels: query.target_labels,
        })
        .map_err(ServiceError::from)?;

    Ok(Json(deleted()))
}

/// Stores a graph node.
fn write_node(node: Node) -> Result<Node, ServiceError> {
    let graph = graph()?;
    graph.put_node(&node)?;
    Ok(node)
}

/// Reads a graph node by ID and labels.
fn read_graph_node(id: String, query: NodeQuery) -> Result<Node, ServiceError> {
    let graph = graph()?;
    let labels = query.labels.iter().map(String::as_str).collect::<Vec<_>>();
    let Some(node) = graph.get_node::<Node>(&labels, id)? else {
        return Err(ServiceError::not_found("knowledge node not found"));
    };

    Ok(node)
}

/// Deletes a graph node by ID and labels.
fn delete_graph_node(
    id: String,
    query: NodeQuery,
) -> Result<DeleteResponse, ServiceError> {
    let graph = graph()?;
    let labels = query.labels.iter().map(String::as_str).collect::<Vec<_>>();
    graph.delete_node::<Node>(&labels, id)?;
    Ok(deleted())
}

/// Stores a graph edge.
fn write_edge(edge: Edge) -> Result<Edge, ServiceError> {
    let graph = graph()?;
    graph.put_edge(&edge)?;
    Ok(edge)
}

/// Opens graph database context for one API call.
fn graph() -> Result<GrafeoDbContext, ServiceError> {
    Ok(GrafeoDbContext::new(GRAPH_DB_NAME.to_string())?)
}

/// Builds standard delete response body.
fn deleted() -> DeleteResponse {
    DeleteResponse { deleted: true }
}
