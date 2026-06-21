//! Storage backends for persistent application state.
//!
//! Currently provides graph storage via [`graph`]. Future backends
//! (vector databases, relational stores, etc.) will live alongside `graph`
//! as sibling modules.

pub mod graph;
