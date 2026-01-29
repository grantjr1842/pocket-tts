# [Sub-Issue] Implement Missing linalg Functions (#574)

**Issue #591**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/591  
**Created:** 2026-01-29T03:07:14Z  
**Updated:** 2026-01-29T03:24:16Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:linalg, ralph/numpy:impl  

---

## Parent Issue
Part of #574 - linalg module exports need verification

## Scope
Implement Missing linalg Functions (cholesky, cond, slogdet, matrix_rank)

## Background
NumPy's linalg module is missing several key functions in rust-numpy.

## Tasks

### 1. Implement cholesky()
```rust
// src/linalg/decompositions.rs
pub fn cholesky(a: &Array<f64>) -> Result<Array<f64>, NumPyError> {
    // Cholesky decomposition: A = L * L^T
    // A must be positive-definite Hermitian (symmetric for real)
    // Returns lower triangular L
}
```

### 2. Implement cond()
```rust
// src/linalg/norms.rs
pub fn cond(x: &Array<f64>, p: Option<&str>) -> Result<f64, NumPyError> {
    // Condition number: ||A|| * ||A^-1||
    // p can be None (2-norm), "fro" (Frobenius), or number
    // For p=None, use svd: cond = sigma_max / sigma_min
}
```

### 3. Implement slogdet()
```rust
// src/linalg/norms.rs
pub fn slogdet(a: &Array<f64>) -> Result<(f64, f64), NumPyError> {
    // Sign and log of determinant
    // Returns (sign, logdet) where det = sign * exp(logdet)
    // More numerically stable than det() for large matrices
}
```

### 4. Implement matrix_rank()
```rust
// src/linalg/norms.rs
pub fn matrix_rank(
    M: &Array<f64>, 
    tol: Option<f64>, 
    hermitian: bool
) -> Result<usize, NumPyError> {
    // Rank of matrix using SVD
    // Number of singular values > tolerance
    // If hermitian, use eigvalsh for efficiency
}
```

### 5. Add Parameters to Existing Functions
```rust
// Add hermitian parameter to svd
pub fn svd(
    a: &Array<f64>, 
    full_matrices: bool, 
    compute_uv: bool, 
    hermitian: bool  // NEW
) -> Result<...>;

// Add rcond parameter to lstsq
pub fn lstsq(
    a: &Array<f64>, 
    b: &Array<f64>, 
    rcond: Option<f64>  // NEW
) -> Result<...>;
```

## Files to Modify
- `src/linalg/decompositions.rs` - Add cholesky
- `src/linalg/norms.rs` - Add cond, slogdet, matrix_rank
- `src/linalg/solvers.rs` - Add rcond to lstsq
- `src/linalg/mod.rs` - Update exports

## Acceptance Criteria
- [ ] cholesky() works for positive-definite matrices
- [ ] cond() matches NumPy condition numbers
- [ ] slogdet() stable for large/small determinants
- [ ] matrix_rank() accurate
- [ ] Parameters added to existing functions
- [ ] All exported properly

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #540 (Complete Linear Algebra Operations) if still active
- Focus on MISSING functions
- Don't modify unrelated linalg code

