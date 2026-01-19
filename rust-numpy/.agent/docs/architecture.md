# rust-numpy Architecture

## System Overview

`rust-numpy` is a high-performance Rust port of core NumPy functionality. It provides a multi-dimensional array type with zero-copy views, broadcasting, and a modular universal function (ufunc) engine.

## Core Data Structures

### Array<T>

The primary owned array type.

```rust
pub struct Array<T> {
    data: Arc<MemoryManager<T>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,
    offset: usize,
}
```

- **Copy-on-Write (CoW):** Uses `Arc` and `MemoryManager` to allow efficient cloning (O(1)) and sharing of data until mutation.
- **Flexible Layout:** Supports C-contiguous, F-contiguous, and non-contiguous (strided) layouts.

### MemoryManager<T>

Manages the raw data buffer.

```rust
pub struct MemoryManager<T> {
    data: Vec<T>,
    ref_count: std::sync::atomic::AtomicUsize,
}
```

It provides thread-safe reference counting and aligned memory access for SIMD.

## Universal Functions (ufuncs)

The ufunc system is implemented via a central `UfuncEngine` which handles:

- Axis reduction
- Broadcasting
- Element-wise operations
- Skip-NA logic

## Key Patterns

### Stride-based Indexing

Instead of nested loops, many operations use linear indices calculated from strides:
`linear_idx = offset + Σ(indices[i] * strides[i])`

### Broadcasting

Broadcasting is achieved by setting strides to 0 for dimensions of size 1, allowing the same element to be accessed repeatedly without extra memory.

### Copy-on-Write Logic

When a mutable reference is needed, `Arc::make_mut` is used to ensure the data is unique, performing a copy only if necessary.
