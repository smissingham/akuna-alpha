# File Extraction

Extracts text content and metadata from files on disk. File type is detected
via Magika and routed to a format-specific extractor — PDF, office documents,
EPUB, or a generic text and markup fallback through omniparse — with
`ExtractionConfig` selecting which of metadata, content, and chunks are
returned.

## Usage

```rust
use akuna_core::extraction::{extract_file, ExtractionConfig};

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = ExtractionConfig {
        return_metadata: true,
        return_content: true,
        ..ExtractionConfig::default()
    };
    let result = extract_file("path/to/file.pdf", &config).await?;
    println!("{}", result.content.unwrap().text.unwrap());
    Ok(())
}
```

## Configuration

`ExtractionConfig` controls which outputs are produced:

| Field              | Effect                                                     |
| ------------------ | ---------------------------------------------------------- |
| `return_metadata`  | Include inferred file metadata in the result.              |
| `return_content`   | Include extracted text in the result.                      |
| `return_chunking`  | Include text chunks in the result.                         |
| `text`             | Optional `TextExtractionConfig` for extractor behaviour.   |
| `chunking`         | Optional `ChunkingConfig` for chunk sizing and delimiters. |

Metadata inference does not read file contents.
Content is only read when `return_content` or `return_chunking` is enabled.

## Supported Formats

- PDF (`.pdf`) via `pdf_oxide`
- Word (`.doc`, `.docx`) and PowerPoint (`.pptx`) via `office_oxide`
- EPUB (`.epub`) via `rbook`
- Markdown, RTF, RSS, XHTML, XML, plain text via `omniparse`
- Source code: C, C++, C#, CSS, Go, HTML, Java, JavaScript, PHP, Python, Ruby, Rust, Shell, SQL, TOML, TypeScript, YAML

Unsupported types return `FileExtractionError::UnsupportedFileType`.
