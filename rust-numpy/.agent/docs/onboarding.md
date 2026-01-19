# Developer Onboarding Guide

## Prerequisites

- Rust (latest stable)
- `cargo`

## Setup

```bash
git clone <repo-url>
cd rust-numpy
cargo build
```

## Running Tests

```bash
# Run all library tests
cargo test

# Run specific module tests
cargo test modules::linalg
cargo test modules::random
```

## Key Modules

- `src/array.rs`: Core array structures and methods.
- `src/ufunc.rs`: Universal function engine and implementations.
- `src/linalg.rs`: Linear algebra operations (matrix multiplication, SVD, etc.).
- `src/dtype.rs`: NumPy-compatible data type system.

## Common Development Tasks

### Implementing a New Ufunc

1. Add function signature to `UfuncOps` trait in `src/ufunc.rs`.
2. Implement it in `src/ufunc_ops.rs` using the `UfuncEngine`.
3. Add tests in `tests/comprehensive_tests.rs`.

### Adding a Dtype Variant

1. Update `Dtype` enum in `src/dtype.rs`.
2. Update `itemsize()`, `alignment()`, and parsing logic in `dtype.rs`.
3. Add necessary trait implementations for the new type (e.g., `FromPrimitive`, `ToPrimitive`).

## Troubleshooting

- **Trait Bounds:** Most generic operations require `T: Clone + Default + 'static`. Some math operations require `num_traits::Float`.
- **Borrow Checker:** Be careful with `ArrayView` lifetimes; they are tied to the source `Array`.
