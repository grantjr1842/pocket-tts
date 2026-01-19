# ADR-002: Modular Universal Function (ufunc) Engine

## Status

Accepted

## Context

NumPy supports hundreds of element-wise operations (addition, multiplication, trigonometry, etc.) and reductions (sum, mean, max). Implementing each of these manually for every layout (C, F, strided) is error-prone and redundant.

## Decision

We implemented a `UfuncEngine` that abstracts the iteration logic.

- **Execution Trait:** A `Ufunc` trait defines how to apply an operation to elements.
- **Generic Engine:** The engine handles ndim recursion, broadcasting, and skipping NA values.
- **Operations:** Individual functions like `add`, `sin`, or `sum` are thin wrappers around calls to the engine.

## Consequences

- **Positive:**
  - Centralized logic for broadcasting and striding.
  - Adding new math functions is as simple as defining a closure or a simple struct.
  - Consistent behavior across all array operations.
- **Negative:**
  - Slight performance abstraction overhead compared to specialized 1D/2D loops.

## Implementation

Located in `src/ufunc.rs` (engine and traits) and `src/ufunc_ops.rs` (specific operations).
