//! Local HTTP REST API server.

use std::{net::SocketAddr, sync::Arc};

use akuna_core::graph::storage::grafeo::GrafeoDbContext;
use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Path, State},
    routing::post,
};
use serde::Deserialize;
use tokio::{net::TcpListener, sync::Mutex};
use utoipa::OpenApi;

use crate::api::{
    error::{ApiErrorBody, ApiResult},
    knowledge::{KnowledgeRequest, KnowledgeService},
};

const GRAPH_DB_NAME: &str = "knowledge";
const API_ADDRESS: &str = "127.0.0.1:9876";

type SharedGraph = Arc<Mutex<GrafeoDbContext>>;

#[derive(Clone)]
struct AppState {
    graph: SharedGraph,
}

#[derive(Deserialize)]
struct KnowledgePath {
    action: String,
    #[serde(rename = "type")]
    knowledge_type: String,
}

#[derive(utoipa::OpenApi)]
#[openapi(paths(knowledge), components(schemas(ApiErrorBody)))]
struct ApiDoc;

/// Runs the local REST API server.
pub async fn run() -> Result<()> {
    let address: SocketAddr = API_ADDRESS.parse()?;
    let listener = TcpListener::bind(address)
        .await
        .with_context(|| format!("Failed to bind API address {address}"))?;
    let graph = GrafeoDbContext::new(GRAPH_DB_NAME.to_string())?;
    let state = AppState {
        graph: Arc::new(Mutex::new(graph)),
    };
    let openapi = ApiDoc::openapi();
    let app = Router::new()
        .route("/knowledge/{action}/{type}", post(knowledge))
        .route(
            "/openapi.json",
            axum::routing::get(|| async { Json(openapi) }),
        )
        .with_state(state);

    akuna_core::ak_info!("serving REST API at http://{address}");
    axum::serve(listener, app)
        .await
        .context("REST API server failed")
}

#[utoipa::path(
    post,
    path = "/knowledge/{action}/{type}",
    params(
        ("action" = String, Path, description = "create, read, update, or delete"),
        ("type" = String, Path, description = "node, assertion, provenance, or edge"),
    ),
    request_body = serde_json::Value,
    responses(
        (status = 200, description = "Knowledge operation result"),
        (status = 400, description = "Invalid request", body = ApiErrorBody),
        (status = 404, description = "Knowledge entity not found", body = ApiErrorBody),
        (status = 500, description = "Graph operation failed", body = ApiErrorBody),
    )
)]
async fn knowledge(
    State(state): State<AppState>,
    Path(path): Path<KnowledgePath>,
    Json(body): Json<serde_json::Value>,
) -> ApiResult<serde_json::Value> {
    let graph = state.graph.lock().await;
    let output = KnowledgeService::new(&graph).execute(KnowledgeRequest {
        action: path.action,
        knowledge_type: path.knowledge_type,
        body,
    })?;

    Ok(Json(output.body))
}
