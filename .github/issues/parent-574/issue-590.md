# [Sub-Issue] Fix linalg Exports (Explicit vs Wildcard) (#574)

**Issue #590**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/590  
**Created:** 2026-01-29T03:07:14Z  
**Updated:** 2026-01-29T03:24:14Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:linalg, ralph/numpy:impl  

---

## Parent Issue
Part of #574 - linalg module exports need verification

## Scope
Fix linalg Module Exports (explicit vs wildcard)

## Background
Current linalg uses wildcard exports (`pub use decompositions::*;`) which may export internal items.

## Tasks

### 1. Audit Current Exports
List everything currently exported and identify:
- Public API items (should be exported)
- Internal items (should NOT be exported)

### 2. Create Explicit Exports
```rust
// src/linalg/mod.rs - BEFORE (wildcard)
pub use decompositions::*;
pub use eigen::*;
pub use norms::*;
pub use products::*;
pub use solvers::*;

// src/linalg/mod.rs - AFTER (explicit)
pub use decompositions::{
    cholesky, qr, svd,
};
pub use eigen::{
    eig, eigh, eigvals, eigvalsh,
};
pub use einsum::{einsum, einsum_path};
pub use norms::{
    cond, det, matrix_rank, norm, slogdet,
};
pub use products::{
    cross, dot, inner, kron, matmul, matrix_power, multi_dot, 
    outer, tensordot, trace, vdot, vecdot,
};
pub use solvers::{
    inv, lstsq, pinv, solve, tensor_inv, tensor_solve,
};

// Remove from exports (keep internal):
// - diagonal, dot_nd, matrix_transpose from products
// - Any helper functions
```

### 3. Mark Internal Items
```rust
// In sub-modules, mark internal functions:
#[doc(hidden)]
pub fn internal_helper() { ... }

// Or make them private:
fn private_helper() { ... }
```

### 4. Verify Against NumPy API
Check NumPy linalg documentation and ensure:
- All NumPy public functions are exported
- No extra rust-numpy internal functions are exported

### 5. Update Documentation
```rust
/// Linear algebra operations
///
/// This module provides... [document what's available]
pub mod linalg {
    // ...
}
```

## Files to Modify
- `src/linalg/mod.rs` - Change to explicit exports
- Various sub-modules - Mark internal items

## Acceptance Criteria
- [ ] All exports are explicit (no wildcards)
- [ ] Matches NumPy linalg public API exactly
- [ ] Internal items marked #[doc(hidden)] or private
- [ ] Documentation lists all available functions

## Conflict Avoidance
⚠️ **COORDINATION:**
- Read-only audit first
- Coordinate with any active linalg work
- Don't change function implementations, just exports

