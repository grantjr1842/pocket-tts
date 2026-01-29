# [Sub-Issue] Implement Generator Methods (choice, shuffle, permutation) (#573)

**Issue #589**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/589  
**Created:** 2026-01-29T03:06:24Z  
**Updated:** 2026-01-29T03:24:12Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:random, ralph/numpy:impl  

---

## Parent Issue
Part of #573 - random module exports incomplete vs numpy.random

## Scope
Implement Missing Generator Methods (choice, shuffle, permutation, etc.)

## Background
NumPy's Generator class has methods for sampling and shuffling that rust-numpy is missing.

## Tasks

### 1. Implement choice()
```rust
impl Generator {
    /// Random sampling from input
    pub fn choice<T>(
        &mut self,
        a: Either<usize, &[T]>,  // size of range OR actual array
        size: Option<&[usize]>,
        replace: bool,
        p: Option<&[f64]>,  // probabilities
        axis: usize,
        shuffle: bool,
    ) -> Result<Array<T>, NumPyError>;
}
```

### 2. Implement shuffle()
```rust
impl Generator {
    /// Shuffle array in-place along given axis
    pub fn shuffle<T>(&mut self, x: &mut Array<T>, axis: Option<isize>) -> Result<(), NumPyError>;
}
```

### 3. Implement permutation()
```rust
impl Generator {
    /// Random permutation of sequence or range
    pub fn permutation<T>(
        &mut self,
        x: Either<usize, &[T]>,
        axis: isize,
    ) -> Result<Array<T>, NumPyError>;
}
```

### 4. Implement permuted()
```rust
impl Generator {
    /// Randomly permute elements along axis independently
    pub fn permuted<T>(&mut self, x: &Array<T>, axis: Option<isize>) -> Result<Array<T>, NumPyError>;
}
```

### 5. Implement multivariate_normal()
```rust
impl Generator {
    /// Multivariate normal distribution
    pub fn multivariate_normal(
        &mut self,
        mean: &[f64],
        cov: &[&[f64]],
        size: Option<&[usize]>,
        check_valid: &str,  // 'warn', 'raise', 'ignore'
        tol: f64,
    ) -> Result<Array<f64>, NumPyError>;
}
```

### 6. Add Module-Level Functions
```rust
// src/random/mod.rs
pub fn choice<T>(...) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().choice(...))
}
pub fn shuffle<T>(x: &mut Array<T>) -> Result<(), NumPyError> { ... }
pub fn permutation<T>(x: Either<usize, &[T]>) -> Result<Array<T>, NumPyError> { ... }
```

## Files to Modify
- `src/random/generator.rs` - Add methods
- `src/random/mod.rs` - Add module-level functions

## Acceptance Criteria
- [ ] choice() implemented
- [ ] shuffle() implemented
- [ ] permutation() implemented
- [ ] permuted() implemented
- [ ] multivariate_normal() implemented
- [ ] Module-level functions work
- [ ] Tests for all

## Conflict Avoidance
⚠️ **COORDINATION:**
- Focus on Generator methods only
- Don't modify BitGenerator trait
- May conflict with #549 (Complete Random Number Generation) - coordinate

