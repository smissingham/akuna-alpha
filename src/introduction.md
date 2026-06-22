# akuna-core

`akuna-core` is the Akuna knowledge tooling library.

The crate exposes a single namespace with optional feature-gated modules, so
consumers depend on one crate and opt into only what they need.

## Modules

| Module       | Feature      | Purpose                                                |
| ------------ | ------------ | ------------------------------------------------------ |
| `chunking`   | `chunking`   | Text chunking via `memchunk` + optional `tree-sitter`  |
| `detection`  | `detection`  | File-type detection via Magika + Burn                  |
| `embedding`  | `embedding`  | Text embedding models (MiniLM, BGE, MPNet, BGE-M3)     |
| `extraction` | `extraction` | File content + metadata extraction                     |
| `reranking`  | `reranking`  | Cross-encoder text rerankers                           |
| `storage`    | `storage`    | Graph storage and retrieval                            |

The `full` feature enables every optional module above.

## Usage

Enable the features you want in `Cargo.toml`:

```toml
[dependencies]
akuna-core = { version = "0.2", features = ["extraction", "embedding"] }
```

Then import via the module path:

```rust,no_run
use akuna_core::extraction::{extract_file, ExtractionConfig};
use akuna_core::embedding::{TextEmbedding, TextEmbeddingOptions};
```

## API reference

For full type-level documentation, build rustdoc across the workspace.
