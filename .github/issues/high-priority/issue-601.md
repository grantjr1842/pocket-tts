# [P1] Hide internal modules from public API with #[doc(hidden)]

**Issue #601**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/601  
**Created:** 2026-01-29T03:35:03Z  
**Updated:** 2026-01-29T03:35:03Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports  

---

## Summary
Multiple internal implementation modules are exposed in the public API. These should be hidden from documentation as they're not part of NumPy's public API.

## Internal Modules to Hide

```rust
// In src/lib.rs - Add #[doc(hidden)] to these:
pub mod advanced_broadcast;      // Internal implementation
pub mod advanced_reductions;     // Internal implementation
pub mod complex_simd;            // SIMD implementation detail
pub mod cpu_features;            // Platform detection
pub mod dynamic_kernel_registry; // Internal registry
pub mod kernel_api;              // Internal API
pub mod kernel_impls;            // Internal implementations
pub mod kernel_registry;         // Internal registry
pub mod layout_optimizer;        // Internal optimization
pub mod parallel;                // Internal parallelization
pub mod parallel_broadcasting;   // Internal parallelization
pub mod parallel_executor;       // Internal parallelization
pub mod parallel_ops;            // Internal parallelization
pub mod performance_metrics;     // Profiling (not NumPy API)
pub mod profiler;                // Profiling (not NumPy API)
pub mod simd;                    // SIMD implementation
pub mod simd_ops;                // SIMD implementation
pub mod strided_executor;        // Internal executor
```

## Fix
Add `#[doc(hidden)]` attribute before each internal module declaration:

```rust
#[doc(hidden)]
pub mod advanced_broadcast;

#[doc(hidden)]
pub mod complex_simd;

// ... etc
```

## Acceptance Criteria
- [ ] All internal modules marked `#[doc(hidden)]`
- [ ] Public API modules remain visible
- [ ] `cargo doc` only shows public API

