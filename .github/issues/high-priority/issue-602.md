# [P1] Convert wildcard exports to explicit exports in lib.rs

**Issue #602**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/602  
**Created:** 2026-01-29T03:35:36Z  
**Updated:** 2026-01-29T03:35:36Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports  

---

## Summary
Multiple wildcard exports (`*`) in `src/lib.rs` can accidentally expose internal items and make the API surface unclear.

## Wildcard Exports Found

```rust
// In src/lib.rs:
pub use crate::array_extra::exports::*;
pub use crate::comparison_ufuncs::exports::*;
pub use crate::fft::*;
pub use crate::matrix::exports::*;
pub use crate::modules::ma::exports::*;
pub use crate::modules::testing::exports::*;
pub use crate::set_ops::exports::*;
pub use bitwise::*;
```

## Risks
1. **Accidental Exposure:** Internal items may become public unintentionally
2. **API Clarity:** Hard to know what's in the public API
3. **Breaking Changes:** Adding items to source modules becomes breaking change
4. **Documentation:** Docs show "*" instead of explicit list

## Recommended Fix

Replace wildcards with explicit exports. Example:

```rust
// BEFORE:
pub use crate::array_extra::exports::*;

// AFTER:
pub use crate::array_extra::exports::{
    array_split, block, choose, column_stack, compress, concatenate, diag, 
    diagonal, diff, dsplit, dstack, ediff1d, gradient, hsplit, hstack, 
    place, put, put_along_axis, putmask, round, row_stack, split, stack, 
    trim_zeros, tril, triu, vander, vsplit, vstack,
};
```

## Acceptance Criteria
- [ ] All wildcard exports converted to explicit
- [ ] Public API unchanged (no breaking changes)
- [ ] `cargo build` passes
- [ ] `cargo doc` shows explicit exports

## Priority Note
This is P1 (not P0) because wildcards work correctly, they just risk future issues.

