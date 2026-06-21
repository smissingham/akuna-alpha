//! Graph storage backend implementations.
//!
//! Each backend provides a concrete context that satisfies the
//! [`crate::graph::GraphDbContext`] trait. Modules here are crate-private;
//! callers obtain contexts via [`crate::graph::open_context`] or
//! [`crate::graph::in_memory_context`].

mod grafeo;

pub(crate) use grafeo::GrafeoDbContext;
