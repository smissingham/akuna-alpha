//! Command-line application entry point.

mod api;
mod cli;
mod config;

use anyhow::Result;
use serde::Serialize;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    cli::run().await
}

pub(crate) fn print_json(value: &impl Serialize) -> Result<()> {
    serde_json::to_writer_pretty(std::io::stdout(), value)?;
    println!();
    Ok(())
}
