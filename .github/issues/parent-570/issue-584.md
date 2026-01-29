# [Sub-Issue] Create emath, exceptions, ctypeslib Submodules (#570)

**Issue #584**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/584  
**Created:** 2026-01-29T03:04:04Z  
**Updated:** 2026-01-29T03:23:54Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:submodules, ralph/numpy:impl  

---

## Parent Issue
Part of #570 - Submodule structure doesn't match NumPy hierarchy

## Scope
Create Missing Submodules: `numpy.emath`, `numpy.exceptions`, `numpy.ctypeslib`

## Background
NumPy has several submodules that rust-numpy is missing entirely.

## Tasks

### 1. `numpy.emath` - Mathematical functions with automatic domain
Mathematical functions that handle domain errors gracefully.

- [ ] Create `src/emath.rs` or `src/emath/mod.rs`
- [ ] `sqrt(x)` - Square root with negative number handling
- [ ] `log(x)` - Log with negative/0 handling
- [ ] `log2(x)` - Base-2 log with domain handling
- [ ] `log10(x)` - Base-10 log with domain handling
- [ ] `logn(n, x)` - Base-n log
- [ ] `power(x, p)` - Power with domain handling
- [ ] `arccos(x)` - Arccos with domain handling
- [ ] `arcsin(x)` - Arcsin with domain handling
- [ ] `arctanh(x)` - Arctanh with domain handling

Implementation: Returns complex results when input is out of real domain.

### 2. `numpy.exceptions` - NumPy-specific exceptions (NumPy 2.0+)
New in NumPy 2.0, specific exception types.

- [ ] Create `src/exceptions.rs`
- [ ] `ComplexWarning` - Warning about complex numbers
- [ ] `VisibleDeprecationWarning` - Visible deprecation
- [ ] `RankWarning` - Rank deficiency warning
- [ ] `TooHardError` - Operation too hard
- [ ] `AxisError` - Invalid axis (may already exist in error.rs)
- [ ] `DTypePromotionError` - Dtype promotion error
- [ ] `UFuncTypeError` - Ufunc type error
- [ ] `UFuncOutputCastingError` - Output casting error
- [ ] `UFuncInputCastingError` - Input casting error

### 3. `numpy.ctypeslib` - C-interoperability
Functions for C FFI integration.

- [ ] Create `src/ctypeslib.rs`
- [ ] `load_library(libname, loader_path)` - Load shared library
- [ ] `ndpointer(dtype=None, ndim=None, shape=None, flags=None)` - Create ctypes pointer
- [ ] `as_array(obj, shape=None)` - Create array from ctypes array
- [ ] `as_ctypes(obj)` - Convert to ctypes array
- [ ] `c_intp` - ctypes integer type for indexing

## Files to Create
- `src/emath.rs` - Mathematical domain functions
- `src/exceptions.rs` - NumPy-specific exceptions
- `src/ctypeslib.rs` - C interoperability

## Files to Modify
- `src/lib.rs` - Add `pub mod emath; pub mod exceptions; pub mod ctypeslib;`

## Acceptance Criteria
- [ ] All three submodules created with proper structure
- [ ] Each submodule exports correct functions
- [ ] Tests for each submodule

## Conflict Avoidance
⚠️ **COORDINATION:**
- New files - minimal conflict risk
- Just add module declarations to lib.rs

