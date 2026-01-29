# [Sub-Issue] Math Ufuncs: Trigonometric/Hyperbolic at Root (#569)

**Issue #577**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/577  
**Created:** 2026-01-29T03:01:57Z  
**Updated:** 2026-01-29T03:23:42Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports, ralph/numpy:impl  

---

## Parent Issue
Part of #569 - Top-level lib.rs exports don't match NumPy public API

## Scope
Implement Mathematical Ufuncs at Root Level: Trigonometric and Hyperbolic Functions

## Background
NumPy exposes ALL ufuncs at the top level (`numpy.sin`, `numpy.cos`, etc.). rust-numpy has these in `math_ufuncs` module but NOT at root.

## Functions to Implement (24 functions)

### Trigonometric
- [ ] `sin(x, /, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)`
- [ ] `cos(x, /, out=None, where=True, ...)`
- [ ] `tan(x, /, out=None, where=True, ...)`
- [ ] `arcsin(x, /, out=None, where=True, ...)`
- [ ] `arccos(x, /, out=None, where=True, ...)`
- [ ] `arctan(x, /, out=None, where=True, ...)`
- [ ] `arctan2(x1, x2, /, out=None, where=True, ...)`
- [ ] `hypot(x1, x2, /, out=None, where=True, ...)`

### Hyperbolic
- [ ] `sinh(x, /, out=None, where=True, ...)`
- [ ] `cosh(x, /, out=None, where=True, ...)`
- [ ] `tanh(x, /, out=None, where=True, ...)`
- [ ] `arcsinh(x, /, out=None, where=True, ...)`
- [ ] `arccosh(x, /, out=None, where=True, ...)`
- [ ] `arctanh(x, /, out=None, where=True, ...)`

### Angle Conversion
- [ ] `degrees(x, /, out=None, where=True, ...)` - Convert radians to degrees
- [ ] `radians(x, /, out=None, where=True, ...)` - Convert degrees to radians
- [ ] `deg2rad(x, /, out=None, where=True, ...)` - Alias for radians
- [ ] `rad2deg(x, /, out=None, where=True, ...)` - Alias for degrees
- [ ] `unwrap(p, discont=None, axis=-1, period=2*pi)` - Unwrap phase angle

### Rounding
- [ ] `around(a, decimals=0, out=None)` - Round to given decimals
- [ ] `round_(a, decimals=0, out=None)` - Alias for around
- [ ] `rint(x, /, out=None, where=True, ...)` - Round to nearest integer
- [ ] `trunc(x, /, out=None, where=True, ...)` - Truncate toward zero
- [ ] `fix(x, out=None)` - Round to nearest integer toward zero

## Implementation Notes
- These should be RE-EXPORTS from math_ufuncs module PLUS free functions
- Support ufunc parameters: `out`, `where`, `casting`, `order`, `dtype`
- May need to enhance existing math_ufuncs to support all parameters

## Files to Modify
- `rust-numpy/src/math_ufuncs.rs` - Ensure all functions exist
- `rust-numpy/src/lib.rs` - Add `pub use` to export at ROOT level
- `rust-numpy/tests/test_trig_ufuncs.rs` - Tests for root-level exports

## Acceptance Criteria
- [ ] All 24 functions exported at crate root
- [ ] Support ufunc parameters (out, where, casting)
- [ ] Tests verify root-level accessibility
- [ ] Documentation shows proper NumPy parity

## Conflict Avoidance
⚠️ **COORDINATION:**
- Focus on RE-EXPORTS and parameter support
- Do NOT rewrite math implementations (use existing)
- Coordinate with #543 (Special Mathematical Functions) owner

