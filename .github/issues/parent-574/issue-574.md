# rust-numpy: linalg module exports need verification

**Issue #574**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/574  
**Created:** 2026-01-29T02:47:40Z  
**Updated:** 2026-01-29T03:24:13Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:linalg, ralph/numpy:audit  

---

## Summary
NumPy's `numpy.linalg` module provides linear algebra operations. rust-numpy has the module but needs verification that all exports match NumPy exactly.

## NumPy linalg Module Reference

### Matrix and Vector Products
```python
numpy.linalg.dot(a, b, out=None)                    # Matrix/dot product
numpy.linalg.vdot(a, b)                             # Vector dot product (conjugate for complex)
numpy.linalg.inner(a, b)                            # Inner product
numpy.linalg.outer(a, b, out=None)                  # Outer product
numpy.linalg.matmul(x1, x2, /, out=None, casting='same_kind')  # Matrix product
numpy.linalg.tensordot(a, b, axes=2)                # Tensor contraction
numpy.linalg.einsum(*operands, out=None, optimize=False)  # Einstein summation
numpy.linalg.einsum_path(*operands, optimize='greedy')    # Path for einsum
numpy.linalg.matrix_power(a, n)                     # Matrix to power n
numpy.linalg.kron(a, b)                             # Kronecker product
numpy.linalg.cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None)  # Cross product
numpy.linalg.multi_dot(arrays, *, out=None)         # Dot product of multiple arrays
numpy.linalg.vecdot(x1, x2, /, out=None, casting='same_kind')  # Vector dot (element-wise)
```

### Decompositions
```python
numpy.linalg.cholesky(a)                            # Cholesky decomposition
numpy.linalg.qr(a, mode='reduced')                  # QR decomposition
numpy.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)  # SVD
numpy.linalg.eig(a)                                 # Eigenvalues/vectors
numpy.linalg.eigh(a, UPLO='L')                      # Eigen for Hermitian/symmetric
numpy.linalg.eigvals(a)                             # Eigenvalues only
numpy.linalg.eigvalsh(a, UPLO='L')                  # Eigenvalues for Hermitian
```

### Matrix Eigenvalues
```python
# (eig, eigh, eigvals, eigvalsh are under Decompositions above)
```
### Norms and Other Numbers
```python
numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)   # Matrix/vector norm
numpy.linalg.cond(x, p=None)                        # Condition number
numpy.linalg.det(a)                                 # Determinant
numpy.linalg.matrix_rank(M, tol=None, hermitian=False)  # Matrix rank
numpy.linalg.slogdet(a)                             # Sign and log of determinant
```

### Solving Equations and Inverting Matrices
```python
numpy.linalg.solve(a, b)                            # Solve linear equations
numpy.linalg.tensorsolve(a, b, axes=None)           # Solve tensor equations
numpy.linalg.lstsq(a, b, rcond='warn')              # Least squares solution
numpy.linalg.inv(a)                                 # Matrix inverse
numpy.linalg.pinv(a, rcond=1e-15, hermitian=False)  # Moore-Penrose pseudoinverse
numpy.linalg.tensorinv(a, ind=2)                    # Tensor inverse
```

### Exceptions
```python
numpy.linalg.LinAlgError                            # Base exception
```

## Current rust-numpy linalg State

### In `src/linalg/mod.rs`
```rust
pub mod decompositions;
pub mod eigen;
pub mod einsum;
pub mod norms;
pub mod products;
pub mod solvers;

pub use decompositions::*;
pub use eigen::*;
pub use einsum::{einsum, einsum_path};
pub use norms::*;
pub use products::{
    cross, diagonal, dot, dot_nd, inner, kron, matmul, matrix_power, matrix_transpose, multi_dot,
    outer, tensordot, trace, vdot, vecdot,
};
pub use solvers::{det, inv, lstsq, pinv, solve, tensor_inv, tensor_solve};

pub use crate::error::NumPyError as LinAlgError;
```

## Problems Identified

### 1. Missing Functions
- ❌ `matrix_rank` - Not exported (may exist but not in re-exports)
- ❌ `slogdet` - Sign and log determinant
- ❌ `cond` - Condition number
- ❌ `cholesky` - Should be in decompositions

### 2. Name Mismatches
- Current: `dot_nd` - Should this be just `dot`?
- Current: `vecdot` - Correct
- Current: `matrix_transpose` - NumPy uses `.T` property, but no standalone function

### 3. Missing Parameters
Many functions missing NumPy parameters:
- `svd` missing `hermitian` parameter
- `eig` missing full implementation
- `lstsq` missing `rcond` parameter handling

### 4. Export Structure
Current exports everything with `*`, which may include private items. Should be explicit.

## Required Changes

### 1. Add Missing Functions
```rust
// src/linalg/norms.rs
pub fn cond(x: &Array<f64>, p: Option<&str>) -> Result<f64, NumPyError>;
pub fn slogdet(a: &Array<f64>) -> Result<(f64, f64), NumPyError>;  // (sign, logdet)
pub fn matrix_rank(M: &Array<f64>, tol: Option<f64>, hermitian: bool) -> Result<usize, NumPyError>;

// src/linalg/decompositions.rs
pub fn cholesky(a: &Array<f64>) -> Result<Array<f64>, NumPyError>;
```

### 2. Fix Parameter Compatibility
```rust
// Add hermitian parameter to svd
pub fn svd(a: &Array<f64>, full_matrices: bool, compute_uv: bool, hermitian: bool) -> Result<(...), NumPyError>;

// Add rcond parameter to lstsq
pub fn lstsq(a: &Array<f64>, b: &Array<f64>, rcond: Option<f64>) -> Result<(...), NumPyError>;
```

### 3. Explicit Exports
```rust
// src/linalg/mod.rs
pub use decompositions::{
    cholesky, qr, svd,
};
pub use eigen::{
    eig, eigh, eigvals, eigvalsh,
};
pub use einsum::{einsum, einsum_path};
pub use norms::{
    cond, det, matrix_rank, norm, slogdet,
};
pub use products::{
    cross, dot, inner, kron, matmul, matrix_power, multi_dot, outer, tensordot, vdot, vecdot,
};
pub use solvers::{
    inv, lstsq, pinv, solve, tensor_inv, tensor_solve,
};

// Remove diagonal, dot_nd, matrix_transpose from exports if not in NumPy API
```

### 4. LinAlgError at Root
```rust
// In src/lib.rs
pub use linalg::LinAlgError;
```

## Acceptance Criteria
- [ ] All NumPy linalg functions exported
- [ ] Function signatures match NumPy (parameters, defaults)
- [ ] `LinAlgError` available at crate root
- [ ] Explicit exports (no wildcard `*`)
- [ ] Missing functions implemented

## Affected Files
- `rust-numpy/src/linalg/mod.rs` - Fix exports
- `rust-numpy/src/linalg/decompositions.rs` - Add cholesky
- `rust-numpy/src/linalg/norms.rs` - Add cond, slogdet, matrix_rank
- `rust-numpy/src/lib.rs` - Re-export LinAlgError


---

## Sub-Issues (for parallel development)

| Issue | Title | Scope |
|-------|-------|-------|
| #591 | Implement Missing linalg Functions | cholesky, cond, slogdet, matrix_rank |
| #590 | Fix linalg Exports | Explicit exports vs wildcard |

**Coordination Notes:**
- #591: Implement missing functions
- #590: Audit and fix exports (can be done in parallel)

