# Graph Namespace

`akuna_core::graph` stores graph nodes and edges without API code depending on a storage backend.
Feature-gated behind `graph`.

| Module       | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| `structs`    | Shared graph node and edge types.                          |
| `storage`    | Backend adapters that implement the shared graph contract. |
| `traits`     | Backend-neutral storage contract.                          |

## How To Use It

Use `GraphNode` and `GraphEdge` for app-facing graph data.
Use `GraphDbContext` so storage remains swappable.
Node ids are scoped by label set, so read and delete calls must pass matching labels.

## Graph Types

`GraphNode` and `GraphEdge` are generic graph shapes for application data.

```rust
use akuna_core::graph::structs::{GraphEdge, GraphNode};
use serde_json::json;

let node = GraphNode {
    id: "rust".to_string(),
    labels: vec!["Concept".to_string(), "Language".to_string()],
    name: "Rust".to_string(),
    description: Some("Systems programming language".to_string()),
    metadata: Some(json!({ "first_release_year": 2015 })),
};

let edge = GraphEdge {
    source_labels: vec!["Concept".to_string(), "Language".to_string()],
    source: "rust".to_string(),
    predicate: "influenced_by".to_string(),
    target: "ml".to_string(),
    target_labels: vec!["Concept".to_string(), "Language".to_string()],
};
```

## Storage

Use `storage` when wiring the concrete backend.
Most domain code should depend on `GraphDbContext`, not a storage adapter directly.
Node ids are scoped by label set, so pass the same labels when reading or deleting nodes.
This example continues from the `node` and `edge` values above.

```rust
use akuna_core::graph::{
    storage::grafeo::GrafeoDbContext,
    traits::GraphDbContext,
};

let graph_db = GrafeoDbContext::new_in_memory();

graph_db.put_node(&node)?;
graph_db.put_edge(&edge)?;

let node = graph_db.get_node(&["Concept", "Language"], "rust")?;
```
