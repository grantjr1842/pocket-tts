# [Sub-Issue] Statistical Functions at Root (#569)

**Issue #580**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/580  
**Created:** 2026-01-29T03:03:09Z  
**Updated:** 2026-01-29T03:23:46Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports, ralph/numpy:impl  

---

## Parent Issue
Part of #569 - Top-level lib.rs exports don't match NumPy public API

## Scope
Implement Statistical Functions at Root Level (NOT just in statistics module)

## Background
NumPy exposes statistical functions at ROOT level (`numpy.mean`, `numpy.std`, etc.) not just in `numpy.statistics`.

## Functions to Implement (26 functions)

### Order Statistics
- [ ] `amin(a, axis=None, out=None, keepdims=False, initial=None, where=True)` - Alias for min
- [ ] `amax(a, axis=None, out=None, keepdims=False, initial=None, where=True)` - Alias for max
- [ ] `nanmin(a, axis=None, out=None, keepdims=False)` - Min ignoring NaN
- [ ] `nanmax(a, axis=None, out=None, keepdims=False)` - Max ignoring NaN
- [ ] `percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False)`
- [ ] `nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False)`
- [ ] `quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False)`
- [ ] `nanquantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False)`

### Averages and Variances
- [ ] `median(a, axis=None, out=None, overwrite_input=False, keepdims=False)`
- [ ] `average(a, axis=None, weights=None, returned=False)`
- [ ] `mean(a, axis=None, dtype=None, out=None, keepdims=False, where=True)`
- [ ] `nanmean(a, axis=None, dtype=None, out=None, keepdims=False)`
- [ ] `std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True)`
- [ ] `nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False)`
- [ ] `var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True)`
- [ ] `nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False)`

### Sums and Products
- [ ] `sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True)`
- [ ] `nansum(a, axis=None, dtype=None, out=None, keepdims=False)`
- [ ] `prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True)`
- [ ] `nanprod(a, axis=None, dtype=None, out=None, keepdims=False)`
- [ ] `cumsum(a, axis=None, dtype=None, out=None)`
- [ ] `nancumsum(a, axis=None, dtype=None, out=None)`
- [ ] `cumprod(a, axis=None, dtype=None, out=None)`
- [ ] `nancumprod(a, axis=None, dtype=None, out=None)`

### Correlations
- [ ] `corrcoef(x, y=None, rowvar=True, bias=None, ddof=None)`
- [ ] `cov(m, y=None, rowvar=True, bias=None, ddof=None)`

## Implementation Notes
- RE-EXPORT from statistics module at ROOT level
- Support all NumPy parameters (axis, keepdims, dtype, where)
- NaN-aware versions skip NaN values

## Files to Modify
- `rust-numpy/src/statistics.rs` - Ensure implementations exist
- `rust-numpy/src/lib.rs` - Add ROOT-level exports
- `rust-numpy/tests/test_statistical_root.rs` - Verify root-level access

## Acceptance Criteria
- [ ] All 26 functions at crate root
- [ ] Support all NumPy parameters
- [ ] NaN handling works correctly
- [ ] Axis handling works for multi-dimensional arrays

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #561 (Add missing reduction functions)
- This issue focuses on ROOT-LEVEL EXPORTS
- #561 focuses on IMPLEMENTATION

