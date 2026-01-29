# [CRITICAL] Merge duplicate statistics imports in src/lib.rs

**Issue #600**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/600  
**Created:** 2026-01-29T03:34:42Z  
**Updated:** 2026-01-29T03:34:42Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports  

---

## Summary
`src/lib.rs` has a duplicate import from the `statistics` module - the second import can be merged into the first.

## Location
File: `src/lib.rs`
- Line 265: First statistics import
- Line 333: Second statistics import (duplicate)

## Current Code
```rust
// Line 265
pub use statistics::{
    average, bincount, corrcoef, cov, digitize, histogram, histogram2d, histogramdd, median,
    nanmax, nanmean, nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd, nansum,
    nanvar, percentile, ptp, quantile, std, var,
};

// ... other code ...

// Line 333
pub use statistics::{amax, amin, max_reduce, min_reduce};
```

## Fix
Merge the second import into the first:

```rust
pub use statistics::{
    amax, amin,  // Add these
    average, bincount, corrcoef, cov, digitize, histogram, histogram2d, histogramdd, max_reduce,  // Add max_reduce
    median, min_reduce,  // Add min_reduce
    nanmax, nanmean, nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd, nansum,
    nanvar, percentile, ptp, quantile, std, var,
};
```

## Acceptance Criteria
- [ ] Single consolidated import statement
- [ ] All functions still accessible
- [ ] `cargo build` passes

