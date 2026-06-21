//! File-type detection using Magika and Burn.
//!
//! # Example
//!
//! ```rust,no_run
//! use akuna_core_detection::Session;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut session = Session::new_default()?;
//!
//!     let detected = session.identify_content_sync(b"fn main() { println!(\"hi\"); }")?;
//!     println!("{} {}", detected.info().label, detected.info().mime_type);
//!
//!     Ok(())
//! }
//! ```

mod config;
mod content {
    pub use crate::vendor::content::*;
}
mod detection;
mod file;
pub(crate) mod model;
mod preprocess;
mod session;
mod vendor;

pub use config::ModelConfig;
pub use detection::{Detection, RankedAlternative};
pub use file::{FileType, InferredType, OverwriteReason, TypeInfo};
pub use model::MagikaInferenceError;
pub use session::{DefaultSession, Session};
