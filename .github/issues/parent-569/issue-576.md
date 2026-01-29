# [Sub-Issue] Array Manipulation Part 1: Join/Stack/Split (#569)

**Issue #576**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/576  
**Created:** 2026-01-29T03:00:55Z  
**Updated:** 2026-01-29T03:23:40Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports, ralph/numpy:impl  

---

## Parent Issue
Part of #569 - Top-level lib.rs exports don't match NumPy public API

## Scope  
Implement Array Manipulation Functions Part 1: Joining, Stacking, and Splitting

## Background
NumPy provides comprehensive array joining and splitting operations. These must be available at the top-level (`numpy.concatenate`, `numpy.stack`, etc.) not just as methods.

## Functions to Implement (15 functions)

### Joining Arrays
- [ ] `concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")`
- [ ] `concat(tup, axis=0, out=None, dtype=None, casting="same_kind")` - Alias

### Stacking Arrays
- [ ] `stack(arrays, axis=0, out=None)`
- [ ] `vstack(tup)` - Stack vertically (row-wise)
- [ ] `hstack(tup)` - Stack horizontally (column-wise)  
- [ ] `dstack(tup)` - Stack depth-wise (along third axis)
- [ ] `column_stack(tup)` - Stack 1-D arrays as columns
- [ ] `row_stack(tup)` - Alias for vstack
- [ ] `block(arrays)` - Assemble array from nested lists/blocks

### Splitting Arrays
- [ ] `split(ary, indices_or_sections, axis=0)`
- [ ] `array_split(ary, indices_or_sections, axis=0)`
- [ ] `hsplit(ary, indices_or_sections)`
- [ ] `vsplit(ary, indices_or_sections)`
- [ ] `dsplit(ary, indices_or_sections)`

### Tiling and Repeating
- [ ] `tile(A, reps)`
- [ ] `repeat(a, repeats, axis=None)`

## Implementation Notes
- These should be FREE FUNCTIONS exported at crate root, not just Array methods
- Support broadcasting and proper axis handling
- Return new Array instances
- Follow NumPy's error handling for shape mismatches

## Files to Modify
- `rust-numpy/src/array_manipulation.rs` - Add implementations (if not exists, create)
- `rust-numpy/src/lib.rs` - Add `pub use` exports
- `rust-numpy/tests/test_array_manipulation.rs` - Add tests

## Acceptance Criteria
- [ ] All 15 functions implemented
- [ ] Exported at crate root (`numpy::concatenate`, etc.)
- [ ] Support all NumPy parameters
- [ ] Broadcasting works correctly
- [ ] Tests pass with various array shapes
- [ ] Documentation follows rustdoc conventions

## Conflict Avoidance
⚠️ **COORDINATION:**
- Does NOT overlap with #560 (Add missing array manipulation functions) - that issue covers MORE functions
- Coordinate with #560 owner to avoid duplicate work
- This issue focuses ONLY on joining/stacking/splitting functions
- Do NOT implement: `append`, `insert`, `delete`, `resize`, `unique`, `flip`, etc. (those are in other sub-issues)

