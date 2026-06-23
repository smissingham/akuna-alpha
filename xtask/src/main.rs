//! Repository maintenance tasks.

mod burn_models;

use anyhow::{Result, bail};
use tracing_subscriber::EnvFilter;

const XTASK_USAGE: &str = "usage: cargo run -p xtask -- burn_models build\n       cargo run -p xtask -- burn_models push";

/// Configure minimal log verbosity for local model generation runs.
///
/// Keeps CLI output clean while surfacing warnings and suppressing noisy parser
/// traces from ONNX/Burn internals.
fn configure_logging() {
    let env_filter = EnvFilter::new(
        "warn,onnx_ir::graph_state=error,onnx_ir=error,burn_onnx=error",
    );

    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_ansi(false)
        .without_time()
        .try_init();
}

/// Runs repository maintenance tasks.
///
/// Supports:
/// - `burn_models build`: regenerate all burn model artifacts
/// - `burn_models push`: upload all burn model artifacts to their HF repos
#[tokio::main]
async fn main() -> Result<()> {
    configure_logging();

    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    let args = raw_args.iter().map(|arg| arg.as_str()).collect::<Vec<_>>();
    match args.as_slice() {
        ["burn_models", "build"] => burn_models::build(),
        ["burn_models", "push"] => burn_models::push(),
        _ => bail!("{XTASK_USAGE}"),
    }
}

/// Returns repository root from xtask manifest location.
fn repo_root() -> Result<std::path::PathBuf> {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(std::path::Path::to_path_buf)
        .ok_or_else(|| {
            anyhow::anyhow!("xtask manifest directory has no parent")
        })
}
