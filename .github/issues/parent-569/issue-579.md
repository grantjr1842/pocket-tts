# [Sub-Issue] Array Manipulation Part 3: Add/Remove Elements (#569)

**Issue #579**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/579  
**Created:** 2026-01-29T03:01:58Z  
**Updated:** 2026-01-29T03:23:45Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports, ralph/numpy:impl  

---

## Parent Issue
Part of #569 - Top-level lib.rs exports don't match NumPy public API

## Scope
Implement Array Manipulation Functions Part 3: Adding/Removing Elements

## Background
NumPy provides functions to add and remove elements from arrays. These are different from stacking because they modify element counts.

## Functions to Implement (5 functions)

### Adding Elements
- [ ] `append(arr, values, axis=None)` - Append values to end
- [ ] `insert(arr, obj, values, axis=None)` - Insert values at indices

### Removing Elements
- [ ] `delete(arr, obj, axis=None)` - Delete sub-arrays along axis
- [ ] `trim_zeros(filt, trim='fb')` - Trim leading/trailing zeros
- [ ] `unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)` - Find unique elements

## Implementation Notes
- `append` is like concatenate but simpler API
- `insert` and `delete` require index handling
- `unique` is computationally more complex - may need sorting
- `trim_zeros` only works on 1-D arrays in NumPy

## Files to Modify
- `rust-numpy/src/array_manipulation.rs` - Implementations
- `rust-numpy/src/lib.rs` - Export at root
- `rust-numpy/tests/test_add_remove.rs` - Tests

## Acceptance Criteria
- [ ] All 5 functions implemented
- [ ] Exported at crate root
- [ ] `unique` supports all return options (indices, inverse, counts)
- [ ] Proper axis handling
- [ ] Tests cover edge cases

## Conflict Avoidance
⚠️ **COORDINATION:**
- Do NOT implement reshape/transpose functions (in #577)
- Do NOT implement stack/concatenate (in #576)
- Coordinate with #560 owner

