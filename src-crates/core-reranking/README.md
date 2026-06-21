[![License: MIT](https://img.shields.io/badge/license-MIT-0f766e?style=for-the-badge)](./LICENSE)
[![Crates.io](https://img.shields.io/crates/v/akuna-rerank?style=for-the-badge)](https://crates.io/crates/akuna-rerank)
[![Docs.rs](https://img.shields.io/docsrs/akuna-rerank?style=for-the-badge)](https://docs.rs/akuna-rerank)
[![Last Commit](https://img.shields.io/github/last-commit/akunasoftware/akuna-rerank?style=for-the-badge)](https://github.com/akunasoftware/akuna-rerank/commits/main)
[![CI](https://img.shields.io/github/actions/workflow/status/akunasoftware/akuna-rerank/ci.yml?label=ci&style=for-the-badge)](https://github.com/akunasoftware/akuna-rerank/actions/workflows/ci.yml)

# akuna-rerank

Simple pure-rust text reranking models built on [Burn](https://github.com/tracel-ai/burn).

- No external model runtimes, and native hardware acceleration.
- Super simple interface, give a query and documents, get ranked scores.
- Optionally, specify a model, and/or a Burn backend for different hardware execution.

## Usage

```rust
use akuna_core_reranking::TextReranker;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextReranker::try_new().await?;

    let score = model.score("Rust ML", "Burn is a Rust deep learning framework")?;
    println!("Relevance score: {score}");

    Ok(())
}
```

## Score Many Pairs

```rust
use akuna_core_reranking::TextReranker;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextReranker::try_new().await?;

    let scores = model.score_batch(
        &[
            ("Rust ML", "Burn is a Rust deep learning framework"),
            ("Rust ML", "Bananas are yellow"),
        ],
        None,
    )?;

    println!("{scores:?}");

    Ok(())
}
```

## Rerank Many Texts

```rust
use akuna_core_reranking::{RerankOptions, TextReranker};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextReranker::try_new().await?;

    let results = model.rerank_with_options(
        "Rust ML",
        &[
            "Burn is a Rust deep learning framework",
            "Bananas are yellow",
            "Rust has strong typing",
        ],
        RerankOptions {
            top_k: Some(2),
            normalize: true,
            batch_size: None,
        },
    )?;

    println!("Best document index: {}", results[0].index);
    println!("Best document: {}", results[0].document);

    Ok(())
}
```

`normalize: true` applies sigmoid to raw logits, mapping scores to `[0, 1]`.
Useful for probability-like interpretation across model families.

## Choose A Model

`RerankerModel::BgeRerankerBase` is the default.

Available models:

- `RerankerModel::BgeRerankerBase`
- `RerankerModel::JinaRerankerV2BaseMultilingual`

```rust,no_run
use akuna_core_reranking::{RerankerModel, TextReranker, TextRerankerOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = TextReranker::new(TextRerankerOptions {
        model: RerankerModel::JinaRerankerV2BaseMultilingual,
        ..Default::default()
    })
    .await?;

    let score = model.score("machine learning", "apprendimento automatico")?;
    assert!(score.is_finite());

    Ok(())
}
```

Future model families can add variants without changing the top-level API.

## Development

This project uses a Nix development shell.

```sh
nix develop
```

Run all checks with:

```sh
./scripts/check.sh
```

Tests compare Rust output with Python `transformers` reference scores
through `uv run scripts/reference_rerank.py`.
