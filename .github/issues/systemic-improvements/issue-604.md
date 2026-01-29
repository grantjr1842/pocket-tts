# [P2] Create centralized export registry (exports.rs)

**Issue #604**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/604  
**Created:** 2026-01-29T04:19:00Z  
**Updated:** 2026-01-29T04:19:00Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports, ralph/numpy:impl  

---

## Summary
Create a centralized export registry to replace scattered `pub use` statements throughout the codebase. This eliminates the duplicate export problem at the architectural level.

## Current Problem
Exports are scattered across `src/lib.rs` with multiple import blocks:
- Line 186: `pub use crate::array_extra::exports::*;`
- Line 187: `pub use crate::comparison_ufuncs::exports::*;`
- Line 236: `pub use array_manipulation::{...};`
- Line 265: `pub use statistics::{...};`
- Line 333: `pub use statistics::{...};` (duplicate!)

This makes it impossible to see the full API surface at a glance.

## Proposed Solution

### Step 1: Create Central Registry
```rust
// src/exports.rs
#![allow(unused_imports)]

/// NumPy-compatible public API - Array Manipulation
pub mod array_api {
    pub use crate::array_manipulation::{
        append, apply_along_axis, apply_over_axes, atleast_1d, atleast_2d, atleast_3d,
        concatenate, delete, expand_dims, eye, flatten, flip, hstack, insert, moveaxis,
        pad, ravel, repeat, reshape, roll, rollaxis, rot90, split, squeeze, stack,
        swapaxes, tile, trim_zeros, vstack, Vectorize,
    };
}

/// NumPy-compatible public API - Math Functions
pub mod math_api {
    pub use crate::math_ufuncs::{
        abs, absolute, acos, acosh, angle, asin, asinh, atan, atan2, atanh,
        conj, conjugate, copysign, cos, cosh, exp, exp2, expm1, fabs, imag,
        log, log10, log1p, log2, real, real_if_close, sign, signbit, sin,
        sinh, sqrt, tan, tanh,
    };
}

/// NumPy-compatible public API - Statistics
pub mod stats_api {
    pub use crate::statistics::{
        amax, amin, average, bincount, corrcoef, cov, digitize, histogram,
        histogram2d, histogramdd, max_reduce, median, min_reduce, nanmax, nanmean,
        nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd, nansum,
        nanvar, percentile, ptp, quantile, std, var,
    };
}

// Re-export everything at crate root
pub use array_api::*;
pub use math_api::*;
pub use stats_api::*;
// ... etc
```

### Step 2: Simplify lib.rs
```rust
// src/lib.rs - Clean and minimal

// Central export registry - single source of truth
pub mod exports;
pub use exports::*;

// Only module declarations, no scattered exports
pub mod array;
pub mod linalg;
pub mod random;
// ... etc

// Internal modules (hidden)
#[doc(hidden)]
pub mod _kernels;
#[doc(hidden)]
pub mod _simd;
```

## Benefits

1. **Single Source of Truth**: All exports in one file
2. **No Duplicates**: Impossible to have duplicates in one location
3. **Clear API Surface**: See entire public API in one place
4. **Easy Auditing**: One file to check against NumPy
5. **Organized**: Grouped by functional area

## Acceptance Criteria
- [ ] Create `src/exports.rs` with organized sections
- [ ] Migrate all exports from `lib.rs` to `exports.rs`
- [ ] `lib.rs` only has `pub use exports::*;`
- [ ] All existing tests pass
- [ ] No change to public API (just reorganization)

## Related
- #599, #600 (duplicate issues)
- #602 (wildcard exports)

