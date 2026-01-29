# Tech Stack and Coding Conventions

## Tech Stack
- Language: Rust
- Framework: ndarray ecosystem
- Package Manager: Cargo
- Testing: cargo test

## Coding Conventions
- NEVER use wildcard exports: `pub use module::*;`
- ALWAYS use explicit exports: `pub use module::{a, b, c};`
- Use type aliases for NumPy types: `pub type int8 = i8;`
- Hide internal modules: `#[doc(hidden)] pub mod _internal;`
- Match NumPy API exactly - no more, no less
- Run verification after EVERY change
