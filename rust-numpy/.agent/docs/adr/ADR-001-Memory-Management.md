# ADR-001: Reference-Counted Memory Management with Copy-on-Write

## Status

Accepted

## Context

NumPy arrays are frequently cloned or sliced, which can lead to excessive memory copies if implemented naively. In Rust, we need a way to share data safely across threads while allowing for in-place modifications when an array has unique ownership.

## Decision

We use a combination of `Arc` (Atomic Reference Counting) and a custom `MemoryManager<T>` struct to manage array data.

- **Reference Counting:** Every `Array<T>` holds an `Arc<MemoryManager<T>>`. `clone()` is O(1) as it only increments the reference count.
- **Copy-on-Write (CoW):** When a mutable operation is performed (e.g., `get_mut`), we use `Arc::make_mut()` to check if the data is unique. If it is shared, it is cloned before modification.

## Consequences

- **Positive:**
  - Zero-copy array cloning and slicing.
  - Thread-safe data sharing out of the box.
  - Simplified API for creating views.
- **Negative:**
  - Small overhead for `Arc` management.
  - Complex implementation for mutable iterators (still partially implemented).

## Implementation

Visible in `src/array.rs` (struct definition) and `src/memory.rs` (`MemoryManager` definition).
