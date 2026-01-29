# [Sub-Issue] Array Manipulation Part 2: Reshape/Rearrange (#569)

**Issue #578**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/578  
**Created:** 2026-01-29T03:01:57Z  
**Updated:** 2026-01-29T03:23:44Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports, ralph/numpy:impl  

---

## Parent Issue
Part of #569 - Top-level lib.rs exports don't match NumPy public API

## Scope
Implement Array Manipulation Functions Part 2: Reshaping and Rearrangement

## Background
NumPy provides comprehensive array reshaping and rearrangement operations. These must be available at the top-level as free functions.

## Functions to Implement (12 functions)

### Reshaping
- [ ] `reshape(a, newshape, order='C')` - Free function version
- [ ] `ravel(a, order='C')` - Contiguous flattened array
- [ ] `flatten(a, order='C')` - Copy flattened (different semantics from ravel)

### Transpose and Axes
- [ ] `transpose(a, axes=None)` - Permute dimensions
- [ ] `moveaxis(a, source, destination)` - Move axes to new positions
- [ ] `rollaxis(a, axis, start=0)` - Roll axis backwards to given position
- [ ] `swapaxes(a, axis1, axis2)` - Interchange two axes

### Rearrangement
- [ ] `flip(m, axis=None)` - Reverse order along axis
- [ ] `fliplr(m)` - Flip left-right (horizontally)
- [ ] `flipud(m)` - Flip up-down (vertically)
- [ ] `roll(a, shift, axis=None)` - Roll elements along axis
- [ ] `rot90(m, k=1, axes=(0, 1))` - Rotate 90 degrees counter-clockwise

## Implementation Notes
- These are FREE FUNCTIONS (Array has methods, but we need top-level versions too)
- Coordinate with Array method implementations to share code
- Support all NumPy parameters including `order` ('C', 'F', 'A')

## Files to Modify
- `rust-numpy/src/array_manipulation.rs` - Add implementations
- `rust-numpy/src/lib.rs` - Add `pub use` exports
- `rust-numpy/tests/test_reshape_rearrange.rs` - Add tests

## Acceptance Criteria
- [ ] All 12 functions implemented
- [ ] Exported at crate root
- [ ] Support all NumPy parameters
- [ ] Order parameter ('C', 'F', 'A') works correctly
- [ ] Tests with various array shapes and axes

## Conflict Avoidance
⚠️ **COORDINATION:**
- Array may already have some methods - create free function wrappers
- Coordinate with #560 owner
- Do NOT implement joining/stacking functions (in #576)
- Do NOT implement insert/delete/unique (in other sub-issues)

