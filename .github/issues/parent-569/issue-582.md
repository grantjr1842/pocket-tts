# [Sub-Issue] Math Ufuncs: Exp/Log/Power at Root (#569)

**Issue #582**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/582  
**Created:** 2026-01-29T03:03:09Z  
**Updated:** 2026-01-29T03:23:50Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports, ralph/numpy:impl  

---

## Parent Issue
Part of #569 - Top-level lib.rs exports don't match NumPy public API

## Scope
Implement Mathematical Ufuncs at Root Level: Exponential, Logarithmic, and Power Functions

## Background
NumPy exposes exponential, logarithmic, and power functions at the top level.

## Functions to Implement (18 functions)

### Exponential
- [ ] `exp(x, /, out=None, where=True, ...)` - Exponential
- [ ] `expm1(x, /, out=None, where=True, ...)` - exp(x) - 1 (accurate for small x)
- [ ] `exp2(x, /, out=None, where=True, ...)` - 2**x
- [ ] `log(x, /, out=None, where=True, ...)` - Natural logarithm
- [ ] `log10(x, /, out=None, where=True, ...)` - Base-10 logarithm
- [ ] `log2(x, /, out=None, where=True, ...)` - Base-2 logarithm
- [ ] `log1p(x, /, out=None, where=True, ...)` - log(1 + x) (accurate for small x)

### Log-Sum-Exp
- [ ] `logaddexp(x1, x2, /, out=None, where=True, ...)` - Logarithm of sum of exponentiations
- [ ] `logaddexp2(x1, x2, /, out=None, where=True, ...)` - Base-2 log of sum of 2**x

### Roots and Powers
- [ ] `sqrt(x, /, out=None, where=True, ...)` - Square root
- [ ] `cbrt(x, /, out=None, where=True, ...)` - Cube root
- [ ] `square(x, /, out=None, where=True, ...)` - Square (x**2)
- [ ] `power(x1, x2, /, out=None, where=True, ...)` - First array raised to power of second
- [ ] `float_power(x1, x2, /, out=None, where=True, ...)` - Power with float promotion

### Reciprocal and Sign
- [ ] `reciprocal(x, /, out=None, where=True, ...)` - 1/x
- [ ] `sign(x, /, out=None, where=True, ...)` - Sign indicator (-1, 0, 1)
- [ ] `signbit(x, /, out=None, where=True, ...)` - True where signbit is set
- [ ] `copysign(x1, x2, /, out=None, where=True, ...)` - Copy sign of x2 to magnitude of x1

## Implementation Notes
- Focus on RE-EXPORTS at root level
- Ensure ufunc parameters are supported
- Special attention to `expm1` and `log1p` accuracy for small values

## Files to Modify
- `rust-numpy/src/math_ufuncs.rs` - Ensure implementations exist
- `rust-numpy/src/lib.rs` - Export at ROOT level
- `rust-numpy/tests/test_exp_log_ufuncs.rs` - Tests

## Acceptance Criteria
- [ ] All 18 functions at crate root
- [ ] Ufunc parameters supported
- [ ] Accuracy for edge cases (small values, overflow)

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #543 (Special Mathematical Functions)
- Do NOT implement trigonometric functions (in #578)
- Focus on exponential/logarithmic/power only

