# Text Chunking

Text chunking strategies built on memchunk. Grammar-aware chunking via
tree-sitter is available behind the `tree-sitter` feature flag.

## Usage

Call `chunk_text` with an optional `ChunkingConfig`, the text content, and an
optional file extension. Without config, default chunk sizing and delimiter
behaviour are used. Use `delimiters_by_ft` when a file type needs its own
boundary rules; use `delimiters` as the fallback boundary rule for all other
content.

| Input            | Purpose                                               |
| ---------------- | ----------------------------------------------------- |
| `config`         | Optional target size and delimiter configuration.     |
| `content`        | Source text to split.                                 |
| `file_extension` | Extension used to pick file-type-specific delimiters. |

```rust
use std::collections::HashMap;

use akuna_core::chunking::{ChunkingConfig, chunk_text};

let mut delimiters_by_ft = HashMap::new();
delimiters_by_ft.insert("md".to_string(), b"\n\n".to_vec());

let config = ChunkingConfig {
    target_size: Some(512),
    delimiters_by_ft,
    delimiters: Some(b"\n".to_vec()),
};

let chunks = chunk_text(Some(&config), "hello\nworld", Some("md"));
```
