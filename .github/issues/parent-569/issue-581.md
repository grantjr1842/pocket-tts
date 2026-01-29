# [Sub-Issue] Comparison and Logical Ops at Root (#569)

**Issue #581**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/581  
**Created:** 2026-01-29T03:03:09Z  
**Updated:** 2026-01-29T03:23:48Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports, ralph/numpy:impl  

---

## Parent Issue
Part of #569 - Top-level lib.rs exports don't match NumPy public API

## Scope
Implement Comparison and Logical Operations at Root Level

## Background
NumPy exposes all comparison and logical operations at top level. These are essential for array filtering and masking.

## Functions to Implement (14 functions)

### Comparison Operations
- [ ] `greater(x1, x2, /, out=None, where=True, ...)` - x1 > x2
- [ ] `greater_equal(x1, x2, /, out=None, where=True, ...)` - x1 >= x2
- [ ] `less(x1, x2, /, out=None, where=True, ...)` - x1 < x2
- [ ] `less_equal(x1, x2, /, out=None, where=True, ...)` - x1 <= x2
- [ ] `equal(x1, x2, /, out=None, where=True, ...)` - x1 == x2
- [ ] `not_equal(x1, x2, /, out=None, where=True, ...)` - x1 != x2

### Logical Operations
- [ ] `logical_and(x1, x2, /, out=None, where=True, ...)`
- [ ] `logical_or(x1, x2, /, out=None, where=True, ...)`
- [ ] `logical_not(x, /, out=None, where=True, ...)`
- [ ] `logical_xor(x1, x2, /, out=None, where=True, ...)`

### Value Testing
- [ ] `isfinite(x, /, out=None, where=True, ...)` - Test for finite values
- [ ] `isinf(x, /, out=None, where=True, ...)` - Test for infinity
- [ ] `isnan(x, /, out=None, where=True, ...)` - Test for NaN
- [ ] `isnat(x, /, out=None, where=True, ...)` - Test for NaT (datetime)

## Implementation Notes
- Return boolean arrays
- Handle NaN correctly (NaN comparisons return False except `!=`)
- Support ufunc parameters
- Logical ops work on boolean arrays; nonzero values are True

## Files to Modify
- `rust-numpy/src/comparison_ufuncs.rs` - Ensure implementations exist
- `rust-numpy/src/lib.rs` - Export at ROOT
- `rust-numpy/tests/test_comparison_logical.rs` - Tests

## Acceptance Criteria
- [ ] All 14 functions at crate root
- [ ] Return boolean dtype arrays
- [ ] NaN handling matches NumPy
- [ ] Ufunc parameters supported

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #568 (Comparison and Logical Operations)
- This issue focuses on EXPORTS; #568 focuses on implementation
- May overlap - communicate with #568 owner

