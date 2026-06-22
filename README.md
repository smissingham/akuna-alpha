[![License](https://img.shields.io/badge/license-MIT-0f766e?style=for-the-badge)](./LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/akunasoftware/akuna?style=for-the-badge)](https://github.com/akunasoftware/akuna/commits/main)

<h1>
  <img src="./assets/icon-gradient.svg" alt="" width="36" align="left">
  Akuna
</h1>

<p align="center">
  <strong><font color="#00bba7"><em>Knowledge</em></font> is about more than <font color="#155dfc"><em>memory</em></font>.</strong>
</p>

This project aims to service a gap in available semantic & context engineering tooling.

All of the below features, while preserving the following key values:

- **Permissive Core** and **Dependencies**
- **Zero External Runtimes** (no pytorch, onnx etc.)
- **Fully Platform Native**
- **Maximum Performance**

## Key Features

- Batteries Included
  - Rich defaults for painless start
  - Knowledge primitives included

- Sophisticated Tooling
  - Document extraction
  - Content chunking
  - Entity recognition & reification (WIP)
  - Hardware accelerated vector embedding
  - Graph storage & traversal (WIP)
  - Semantic search
  - Fulltext search
  - ML reranking (WIP)

## [Application](./src-crates/app/)

Command line application, currently implements minimal features. Much more coming here soon.

## [Core Library](./src-crates/core/)

See [`src-crates/core/Cargo.toml`](./src-crates/core/Cargo.toml) for available feature sets.
Use `full` to enable all feature-gated APIs.

`akuna-core` is a single crate with feature-gated modules.

| Module          | Cargo Feature | Description                                          |
| --------------- | ------------- | ---------------------------------------------------- |
| `extraction`    | `extraction`  | Extracts file metadata, text content, and chunks.    |
| `chunking`      | `chunking`    | Splits text using configured delimiters and size.    |
| `embedding`     | `embedding`   | Loads text embedding models and embeds text batches. |
| `storage`       | `storage`     | Provides graph primitives, types, and storage APIs.  |
| `reranking`     | `reranking`   | ML reranking of retrieved candidates.                |
| `detection`     | `detection`   | File type inference (Rust native Magika).            |

Module source lives under [`./src-crates/core/src/`](./src-crates/core/src/).

## Workspace Crates

| Crate          | Path                    | Purpose                                              |
| -------------- | ----------------------- | ---------------------------------------------------- |
| `akuna`        | `./src-crates/app/`     | Command line application binary.                     |
| `akuna-core`   | `./src-crates/core/`    | Knowledge tooling library with feature-gated modules.|

## Documentation

- `akdoc` — alias to `cargo doc` for rustdoc rendering of the workspace.
- `akbook` — alias to `mdbook` for viewing the project book.
