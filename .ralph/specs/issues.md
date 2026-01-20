# Imported GitHub Issues

## Issue #40: [ufunc][core][kernels] Broadcast-aware binary exec path (elementwise add)

Goal: Make binary execution broadcast-correct.

Acceptance Criteria:
- [ ] add works for broadcasted shapes (e.g. (3,1)+(1,4)->(3,4))
- [ ] Tests cover mixed broadcasting patterns

Depends on: #34, #35, #39


---

## Issue #41: [api] Public Array facade + add() wired end-to-end

Goal: Expose minimal public API for Array + add.

Acceptance Criteria:
- [ ] api::Array supports basic construction (start with f64)
- [ ] api::ops::add(&Array,&Array)->Array
- [ ] Tests validate public API behavior

Depends on: #40


---

## Issue #42: [kernels][ufunc][api] Add mul ufunc (mirror add) + tests

Goal: Implement multiplication ufunc with the same pathway as add.

Acceptance Criteria:
- [ ] mul works for contiguous and broadcasted inputs
- [ ] Tests mirror add coverage for mul

Depends on: #41


---

## Issue #43: [ufunc][kernels][api] Global sum reduction for f64

Goal: Implement sum reduction over all axes.

Acceptance Criteria:
- [ ] sum(Array)->scalar (or 0-D Array) for f64
- [ ] Tests define and enforce empty-array policy

Depends on: #41


---

## Issue #44: [ufunc][api] sum(axis=..., keepdims=...): single axis first

Goal: Implement sum over a single axis with keepdims.

Acceptance Criteria:
- [ ] sum(axis=i, keepdims=bool) correct for small shapes
- [ ] Tests validate resulting shape + values

Depends on: #43


---

## Issue #45: [performance][core][ufunc] Dimension coalescing into fewer contiguous runs

Goal: Optimize iteration by coalescing dimensions into fewer 1-D kernel calls.

Acceptance Criteria:
- [ ] Outputs identical to baseline across existing test suite
- [ ] Basic perf sanity check (bench optional)

Depends on: #40, #44


---

## Issue #46: [performance][kernels] SIMD kernels (feature-gated) + runtime dispatch

Goal: Add optional SIMD-specialized kernels with runtime dispatch.

Acceptance Criteria:
- [ ] Baseline path remains default and passes all tests
- [ ] SIMD feature passes identical tests when enabled
- [ ] Dispatch chooses best available implementation safely

Depends on: #45


---

## Issue #47: [performance] Threading policy for safe kernels (no overlap/alias hazards)

Goal: Parallelize only where safe and deterministic under defined rules.

Acceptance Criteria:
- [ ] Threading is conditional and respects aliasing constraints
- [ ] Tests confirm correctness; determinism where required

Depends on: #45


---

## Issue #48: [rust-numpy] Phase 1: FFT Module Implementation

Complete all 6 stubbed FFT functions using rustfft crate. See `thoughts/plans/fft-module-implementation.md` for details.
Deliverables:
- fft(), ifft()
- rfft2(), irfft2()
- rfftn(), irfftn()
- hilbert()

---

## Issue #49: [rust-numpy] Phase 2: Advanced Array Methods

Implement advanced array manipulation methods. See `thoughts/plans/advanced-array-methods-and-nan-stats.md` for details.
Deliverables:
- flatten(), ravel(), squeeze()
- repeat(), tile()
- swapaxes(), rollaxis(), moveaxis()
- atleast_1d/2d/3d()

---

## Issue #50: [rust-numpy] Phase 3: NaN-Aware Statistics

Implement statistical functions that properly handle NaN values. See `thoughts/plans/advanced-array-methods-and-nan-stats.md` for details.
Deliverables:
- nanmean(), nanstd(), nanvar()
- nanmedian(), nansum(), nanprod()

---

## Issue #51: [rust-numpy] Phase 4: Advanced Indexing & Slicing

Implement fancy indexing, boolean masking, and ellipsis indexing. See `thoughts/plans/advanced-indexing-and-additional-gaps.md` for details.
Deliverables:
- Fancy indexing, Boolean masking
- Ellipsis indexing
- Field access, NewAxis support

---

## Issue #52: [rust-numpy] Phase 5: Missing Utility Ufuncs

Add utility functions for NaN/Inf/Finite checking and angle conversions. See `thoughts/plans/advanced-indexing-and-additional-gaps.md` for details.
Deliverables:
- isnan(), isinf(), isfinite()
- deg2rad(), rad2deg()
- Array versions of above

---

## Issue #53: [rust-numpy] Phase 6: Polynomial Module Comparison

Audit and complete polynomial module. See `thoughts/plans/advanced-indexing-and-additional-gaps.md` for details.
Deliverables:
- Audit existing functions
- Implement missing: roots(), poly(), polyval(), polyfit()
- Comprehensive tests

---

## Issue #54: [rust-numpy] Phase 7: Test Coverage Expansion

Expand test coverage to include NumPy conformance testing. See `thoughts/plans/rust-numpy-port-master-execution-plan.md` for details.
Deliverables:
- Conformance tests against NumPy
- Edge case testing
- Performance benchmarks

---

## Issue #55: [rust-numpy] C-ABI Integration & Python Shim

Integrate Rust library via C-ABI and rewrite Python shim. See `thoughts/plans/rust_numpy_parity_gaps.md` for details.
Deliverables:
- Export all required functions via C-ABI
- Rewrite numpy_rs.py to use ctypes
- Remove PyO3 dependencies

---

## Issue #56: [rust-numpy] Implement Eigen-decomposition

### A. Technical Specification

**What this feature does:**

- Computes eigenvalues and eigenvectors of square matrices
- Supports both real and complex matrices
- Returns eigenvectors as columns of unitary matrix

**Technical requirements:**

- Use Francis double-shift QR iteration algorithm (industry standard)
- Reduce matrix to upper Hessenberg form first
- Handle defective matrices (repeated eigenvalues)
- Sort eigenvalues and eigenvectors consistently
- Return complex eigenvalues even for real matrices

**NumPy API requirements:**

```python
np.linalg.eig(a)  # Returns (eigenvalues, eigenvectors)
```

- Input: Square matrix (n×n)
- Output: Tuple of (eigenvalues array, eigenvectors matrix)
- Eigenvectors as columns: `eigenvectors[:, i]` is eigenvector for `eigenvalues[i]`
- Eigenvalue ordering: Unspecified (typically not sorted)

**Special cases to handle:**

- Non-square matrices (error)
- Singular matrices (ill-conditioned, may produce large errors)
- Defective matrices (incomplete eigenvector basis)
- Empty matrices
- 1×1 matrices

**Algorithms:**

1. Francis double-shift QR iteration with Hessenberg reduction
2. For real matrices: Complex eigenvalue detection
3. Wilkinson shift for convergence

### B. Code Structure

**Files to create/modify:**

- `src/linalg/eigen.rs` - Create eigen-decomposition module (extend existing)
- `src/linalg/mod.rs` - Export eig() function
- `tests/eigen_tests.rs` - Add eigen tests

**Function signatures:**

```rust
pub fn eig<T>(
    a: &Array<T>,
) -> Result<(Array<Complex64>, Array<Complex64>), NumPyError>
where T: LinalgScalar;

pub fn hessenberg_reduction<T>(
    a: &Array<T>,
) -> Array<T>
where T: LinalgScalar;

pub fn francis_double_shift<T>(
    h: &Array<T>,
    max_iterations: usize,
    eps: f64,
) -> (Array<Complex64>, Array<Complex64>)
where T: LinalgScalar;
```

**Data structures to define:**

```rust
// Hessenberg matrix representation
struct HessenbergMatrix<T> {
    data: Array<T>,
    is_upper: bool,
}

// Francis QR iteration state
struct FrancisState {
    iteration: usize,
    converged: bool,
}
```

**Integration points:**

- QR decomposition - Use within Francis iteration
- Complex arithmetic - Use for eigenvalue computation
- Existing `linalg/mod.rs` - Export new eig() function

### C. Implementation Steps

**Phase 1: Foundation (8-10 hours)**

1. [ ] Review existing `src/linalg/eigen.rs` (stub only)
2. [ ] Study Francis double-shift QR algorithm (reference papers)
3. [ ] Implement Wilkinson shift computation
4. [ ] Implement Hessenberg reduction function
5. [ ] Add data structures: `HessenbergMatrix`, `FrancisState`

**Phase 2: Core Algorithm (15-20 hours)** 6. [ ] Implement Francis QR iteration: - Initialize with Hessenberg matrix - Iteratively apply QR steps with Wilkinson shift - Detect convergence (subdiagonal elements below threshold) - Extract 2×2 submatrices for eigenvalues 7. [ ] Handle complex eigenvalues for real matrices: - Detect 2×2 blocks with complex eigenvalues - Convert to complex form 8. [ ] Implement eigenvector back-transformation: - Track accumulated Q matrices from QR steps - Apply to identity to get eigenvectors 9. [ ] Add max iteration limit (e.g., 1000) 10. [ ] Add convergence detection (subdiagonal < eps)

**Phase 3: Integration & Testing (7-10 hours)** 11. [ ] Create `eig()` public function with proper error handling 12. [ ] Validate input is square matrix 13. [ ] Handle edge cases: 1×1, singular, empty 14. [ ] Write unit tests with known examples 15. [ ] Benchmark vs NumPy for correctness and performance 16. [ ] Add documentation with examples 17. [ ] Export from linalg module

**Phase 4: Optimization (Optional, 8-12 hours)** 18. [ ] Optimize Hessenberg reduction (avoid unnecessary computations) 19. [ ] Cache QR results where possible 20. [ ] Add early termination for small matrices (n < 10)

### D. Testing Requirements

**Unit tests in `tests/eigen_tests.rs`:**

```rust
#[test]
fn test_eig_2x2_real()
#[test]
fn test_eig_3x3_real()
#[test]
fn test_eig_complex_matrix()
#[test]
fn test_eig_symmetric()
#[test]
fn test_eig_defective_matrix()
#[test]
fn test_eig_1x1()
#[test]
fn test_non_square_error()
#[test]
fn test_eigenvalue_accuracy()
#[test]
fn test_eigenvector_orthogonality()
#[test]
fn test_convergence_large_matrix()
```

**Integration tests:**

- Test eig() @ eigenvectors ≈ identity (within tolerance)
- Test with real matrices that have complex eigenvalues
- Verify NumPy conformance

**Performance benchmarks:**

- Benchmark eigen-decomposition vs NumPy
- Test scaling with matrix size (10, 50, 100, 500)

**Edge case tests:**

- Singular matrices (near-singular, very ill-conditioned)
- Repeated eigenvalues
- Complex input matrices

### E. Success Criteria

- [ ] All eigen tests pass
- [ ] Eigenvalues match NumPy (within 1e-6 tolerance for well-conditioned)
- [ ] Eigenvectors are orthogonal (for real symmetric matrices)
- [ ] Performance within 3x of NumPy for typical sizes
- [ ] Handles complex eigenvalues correctly
- [ ] No unsafe code
- [ ] Documentation complete with examples
- [ ] Exported in linalg module


---

## Issue #57: [rust-numpy] Implement QR Decomposition

### A. Technical Specification

**What this feature does:**

- Decomposes matrix A into Q (orthonormal/unitary) and R (upper triangular)
- Supports 4 NumPy modes: 'reduced', 'complete', 'r', 'raw'
- Handles rectangular matrices (both m>n and m<n)

**Technical requirements:**

- Use Householder reflections algorithm (standard method)
- Orthogonal/unitary Q matrix via Householder vectors
- Upper triangular R matrix via Gaussian elimination
- Handle complex matrices (unitary Q)
- Return Q and R in correct shapes based on mode

**NumPy API requirements:**

```python
np.linalg.qr(a, mode='reduced')
```

**Modes:**

- `mode='reduced'` (default): Q is m×k, R is k×n where k=min(m,n)
- `mode='complete'`: Q is m×m square, R is m×n (zero-padded)
- `mode='r'`: Returns only R (k×n)
- `mode='raw'`: Returns (h, tau) where h contains Householder vectors, tau contains scaling factors

**Special cases to handle:**

- Non-2D arrays (error)
- Empty arrays
- 1×N matrices (Q is 1×1, R is 1×N)
- Complex matrices

**Algorithm:**

1. Householder reflections for QR factorization
2. Accumulate Q matrix or track Householder vectors
3. Upper triangular R via successive elimination

### B. Code Structure

**Files to create/modify:**

- `src/linalg/decompositions.rs` - Extend existing with qr()
- `src/linalg/mod.rs` - Export qr() function
- `tests/qr_tests.rs` - Add QR tests

**Function signatures:**

```rust
#[derive(Debug, Clone, Copy)]
pub enum QrMode {
    Reduced,
    Complete,
    R,
    Raw,
}

pub fn qr<T>(
    a: &Array<T>,
    mode: QrMode,
) -> Result<QrResult<T>, NumPyError>
where T: LinalgScalar;

pub struct QrResult<T> {
    q: Option<Array<T>>,
    r: Array<T>,
    h: Option<Array<T>>,      // For raw mode
    tau: Option<Array<T>>,     // For raw mode
}

// Householder reflection
fn householder_reflection<T>(
    x: &[T],
) -> (Vec<T>, T)
where T: LinalgScalar;
```

**Data structures to define:**

```rust
pub enum QrResult<T> {
    Reduced(Array<T>, Array<T>),
    Complete(Array<T>, Array<T>),
    R(Array<T>),
    Raw(Array<T>, Array<T>),
}
```

**Integration points:**

- Matrix multiplication - Use in Q accumulation
- Existing decompositions module - Extend with qr()
- Array creation functions - Use for Q, R matrices

### C. Implementation Steps

**Phase 1: Householder Reflections (6-8 hours)**

1. [ ] Review existing `src/linalg/decompositions.rs`
2. [ ] Implement `householder_reflection()`:
   - Compute Householder vector v = x - 2\*(x·u)u where u = x/||x||
   - Return (v, tau) where tau = 2/(v·v)
3. [ ] Implement `apply_householder()` to apply reflection to matrix

**Phase 2: QR Factorization (10-12 hours)** 4. [ ] Implement `qr_reduced()`: - Apply Householder reflections column by column - Stop after k=min(m,n) reflections - Accumulate Q matrix (m×k) - Compute R (k×n upper triangular) 5. [ ] Implement `qr_complete()`: - Continue reflections for all m columns - Q is m×m unitary/square - R is m×n with zeros below diagonal 6. [ ] Implement `qr_r_only()`: - Compute only R matrix (k×n) - Skip Q accumulation (performance) 7. [ ] Implement `qr_raw()`: - Return h (Householder vectors) and tau (scaling factors) - Q can be reconstructed by caller if needed

**Phase 3: Integration & Testing (4-5 hours)** 8. [ ] Create public `qr()` function with mode parameter 9. [ ] Validate input is 2D matrix 10. [ ] Handle m>n and m<n cases 11. [ ] Add unit tests for all modes 12. [ ] Verify NumPy conformance 13. [ ] Document QR modes with examples

### D. Testing Requirements

**Unit tests in `tests/qr_tests.rs`:**

```rust
#[test]
fn test_qr_reduced_square()
#[test]
fn test_qr_reduced_tall()
#[test]
fn test_qr_reduced_wide()
#[test]
fn test_qr_complete()
#[test]
fn test_qr_r_only()
#[test]
fn test_qr_raw()
#[test]
fn test_qr_complex()
#[test]
fn test_qr_orthogonality()
#[test]
fn test_qr_accuracy_a_approx_qr()
```

**Integration tests:**

- Verify Q is orthonormal (Q^T @ Q ≈ I)
- Verify R is upper triangular
- Test A ≈ Q @ R (within tolerance)
- Compare outputs with NumPy.linalg.qr

**Performance benchmarks:**

- Benchmark QR vs NumPy for various sizes (10, 50, 100, 500)

### E. Success Criteria

- [ ] All QR tests pass
- [ ] Q is orthonormal/unitary (within tolerance)
- [ ] R is upper triangular
- [ ] All 4 modes implemented correctly
- [ ] Performance within 2x of NumPy
- [ ] NumPy conformance tests pass
- [ ] No unsafe code
- [ ] Documentation complete with mode examples


---

## Issue #58: [rust-numpy] Implement SVD (Singular Value Decomposition)

### A. Technical Specification

**What this feature does:**

- Decomposes matrix A into U @ Σ @ V^T where U and V are unitary, Σ is diagonal
- Computes singular values (always non-negative, sorted descending)
- Supports full_matrices and compute_uv parameters

**Technical requirements:**

- Use Golub-Kahan bidiagonalization algorithm
- Implicit QR algorithm for bidiagonal SVD
- Singular values ALWAYS returned as f64 (even for complex input)
- Handle rank-deficient matrices
- Support complex matrices (unitary transformations)

**NumPy API requirements:**

```python
np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False)
```

- Input: 2D matrix (m×n)
- Output: (u, s, vh) where:
  - u: (m, m) unitary matrix
  - s: (k,) array of singular values where k=min(m,n)
  - vh: (n, n) unitary matrix (V^T)

**Parameters:**

- `full_matrices=True`: u is (m,m), vh is (n,n)
- `full_matrices=False`: u is (m,k), vh is (k,n) where k=min(m,n)
- `compute_uv=False`: Returns only s
- `hermitian=True`: Optimized for Hermitian matrices (not in initial scope)

**Special cases to handle:**

- Non-2D arrays (error)
- Empty arrays
- Rank-deficient matrices (zero singular values)
- Complex matrices

**Algorithm:**

1. Golub-Kahan bidiagonalization (A = U @ B @ V^T where B is bidiagonal)
2. Implicit QR for bidiagonal SVD (Wilkinson shift)
3. Divide and conquer for large matrices (optional optimization)

### B. Code Structure

**Files to create/modify:**

- `src/linalg/decompositions.rs` - Extend existing with svd()
- `src/linalg/mod.rs` - Export svd() function
- `tests/svd_tests.rs` - Add SVD tests

**Function signatures:**

```rust
pub fn svd<T>(
    a: &Array<T>,
    full_matrices: bool,
    compute_uv: bool,
) -> Result<SvdResult, NumPyError>
where T: LinalgScalar;

pub struct SvdResult {
    u: Option<Array<Complex64>>,
    s: Array<f64>,  // Always f64, even for complex input
    vh: Option<Array<Complex64>>,
}

// Bidiagonalization
pub fn golub_kahan_bidiagonal<T>(
    a: &Array<T>,
) -> (Array<T>, Array<T>, Array<T>)  // (U, B, V)
where T: LinalgScalar;

// Bidiagonal SVD (implicit QR)
pub fn bidiagonal_svd<T>(
    b: &Array<T>,
) -> (Array<T>, Array<f64>, Array<T>)  // (U, s, V)
where T: LinalgScalar;
```

**Data structures to define:**

```rust
// Bidiagonal matrix representation
struct BidiagonalMatrix<T> {
    upper: Vec<T>,  // Superdiagonal
    main: Vec<T>,   // Diagonal
    lower: Vec<T>,  // Subdiagonal
}

// Wilkinson shift state
struct WilkinsonShift {
    shift: f64,
    converged: bool,
}
```

**Integration points:**

- QR decomposition - Use in bidiagonalization
- Matrix multiplication - Use for verification
- Existing decompositions module - Extend with svd()

### C. Implementation Steps

**Phase 1: Bidiagonalization (10-12 hours)**

1. [ ] Review existing `src/linalg/decompositions.rs`
2. [ ] Implement Householder reflections for bidiagonalization:
   - Left Householder (zero above diagonal in columns)
   - Right Householder (zero below diagonal in rows)
3. [ ] Implement `golub_kahan_bidiagonal()`:
   - Apply left reflections to zero above diagonal
   - Apply right reflections to zero below diagonal
   - Return U (orthonormal), B (bidiagonal), V (orthonormal)

**Phase 2: Bidiagonal SVD (12-15 hours)** 4. [ ] Implement `bidiagonal_svd()`: - Initialize with bidiagonal matrix - Apply Wilkinson shift for 2×2 submatrix at bottom - Perform Givens rotations to eliminate subdiagonal - Iterate until subdiagonal < eps (converged) 5. [ ] Extract singular values from diagonal 6. [ ] Back-transform singular vectors: - Accumulate left Givens rotations into U - Accumulate right Givens rotations into V 7. [ ] Handle complex singular values

**Phase 3: Integration & Testing (8-10 hours)** 8. [ ] Create public `svd()` function with parameters 9. [ ] Implement full_matrices flag: - True: Return full U (m×m) and full V (n×n) - False: Return reduced U (m×k) and V (k×n) 10. [ ] Implement compute_uv flag: - True: Return (u, s, vh) - False: Return only s 11. [ ] Validate singular values are f64 (convert if needed) 12. [ ] Handle rank-deficient matrices (zero singular values) 13. [ ] Add unit tests with known matrices 14. [ ] Verify NumPy conformance (compare U, s, V) 15. [ ] Benchmark vs NumPy 16. [ ] Document API with examples

### D. Testing Requirements

**Unit tests in `tests/svd_tests.rs`:**

```rust
#[test]
fn test_svd_square()
#[test]
fn test_svd_tall()
#[test]
fn test_svd_wide()
#[test]
fn test_svd_rank_deficient()
#[test]
fn test_svd_singular_values_sorted()
#[test]
fn test_svd_complex()
#[test]
fn test_svd_compute_uv_false()
#[test]
fn test_svd_full_matrices()
#[test]
fn test_svd_accuracy_a_approx_usv()
#[test]
fn test_svd_unitarity()
```

**Integration tests:**

- Verify U is unitary (U @ U^H ≈ I)
- Verify V is unitary (V @ V^H ≈ I)
- Test A ≈ U @ diag(s) @ V^H (within tolerance)
- Compare with NumPy.linalg.svd

**Performance benchmarks:**

- Benchmark SVD vs NumPy for various sizes (10, 50, 100, 500)

### E. Success Criteria

- [ ] All SVD tests pass
- [ ] Singular values are non-negative f64, sorted descending
- [ ] U and V are unitary (within tolerance)
- [ ] A ≈ U @ diag(s) @ V^T (within tolerance)
- [ ] Handles rank-deficient matrices correctly
- [ ] NumPy conformance tests pass
- [ ] Performance within 3x of NumPy
- [ ] No unsafe code
- [ ] Documentation complete with examples


---

## Issue #59: [rust-numpy] Implement Multi-dimensional Dot Products

### A. Technical Specification

**What this feature does:**

- Implements dot(), matmul(), and tensordot() for N-D arrays
- Supports broadcasting between multi-dimensional arrays
- Optimizes for common cases (2D×2D matrix multiplication)

**Technical requirements:**

- Support N-D arrays with broadcasting
- Implement different dimension combinations:
  - 1D × 1D → inner product (scalar)
  - 1D × 2D → matrix-vector multiplication
  - 2D × 1D → vector-matrix multiplication
  - 2D × 2D → matrix multiplication
  - N-D × N-D → tensor contraction
- Proper broadcasting rules (NumPy semantics)
- Clear error messages for shape mismatches

**NumPy API requirements:**

```python
np.dot(a, b)
np.matmul(a, b)  # Python @ operator
np.tensordot(a, b, axes=2)
```

**Shape rules:**

- `dot(a, b)`: Last dim of a must match second-to-last dim of b
- `matmul(a, b)`: (..., n, k) @ (..., k, m) → (..., n, m)

**Special cases to handle:**

- Scalar × Array or Array × Scalar
- Shape mismatches
- Empty arrays
- Large dimensions (>10)

### B. Code Structure

**Files to create/modify:**

- `src/linalg/products.rs` - Extend existing with N-D support
- `src/linalg/mod.rs` - Export enhanced dot(), matmul(), tensordot()
- `tests/tensor_dot_tests.rs` - Add tensor multiplication tests

**Function signatures:**

```rust
pub fn dot<T>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

pub fn matmul<T>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

pub fn tensordot<T>(
    a: &Array<T>,
    b: &Array<T>,
    axes: usize,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

// Optimized 2D×2D case
pub fn matrix_multiply_2d<T>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;
```

**Data structures to define:**

```rust
// Tensor contraction specification
struct TensorContraction {
    a_axes: Vec<usize>,
    b_axes: Vec<usize>,
    output_shape: Vec<usize>,
}
```

**Integration points:**

- Broadcasting module - Use for shape validation and broadcasting
- Existing dot() - Extend for N-D support
- Array operations - Use for efficient computation

### C. Implementation Steps

**Phase 1: Foundation (6-8 hours)**

1. [ ] Review existing `src/linalg/products.rs` (only 2D supported)
2. [ ] Implement shape validation for N-D arrays
3. [ ] Implement broadcasting compatibility check
4. [ ] Create error messages for shape mismatches

**Phase 2: Core Operations (8-10 hours)** 5. [ ] Implement `dot()` for 1D×1D: - Compute inner product - Return scalar (0-D array) 6. [ ] Implement `dot()` for 1D×2D and 2D×1D: - Validate shapes - Compute matrix-vector product 7. [ ] Implement `dot()` for 2D×2D: - Validate (m, k) @ (k, n) → (m, n) - Use optimized loop or call `matrix_multiply_2d()` 8. [ ] Implement `dot()` for N-D×N-D: - Identify contraction dimensions (last dim of a, second-to-last of b) - Contract along specified axes - Compute output shape

**Phase 3: Additional Functions (4-5 hours)** 9. [ ] Implement `matmul()` with same logic as dot() 10. [ ] Implement `tensordot()`: - Generalized tensor contraction - Support arbitrary axes parameter - Contract along specified dimensions

**Phase 4: Testing & Optimization (3-4 hours)** 11. [ ] Add unit tests for all dimension combinations 12. [ ] Add SIMD optimization for 2D×2D (if beneficial) 13. [ ] Benchmark vs NumPy for various sizes 14. [ ] Verify NumPy conformance 15. [ ] Document functions with examples

### D. Testing Requirements

**Unit tests in `tests/tensor_dot_tests.rs`:**

```rust
#[test]
fn test_dot_1d_1d()
#[test]
fn test_dot_1d_2d()
#[test]
fn test_dot_2d_1d()
#[test]
fn test_dot_2d_2d()
#[test]
fn test_dot_nd_nd()
#[test]
fn test_matmul()
#[test]
fn test_tensordot()
#[test]
fn test_shape_error()
#[test]
fn test_scalar_multiplication()
#[test]
fn test_empty_arrays()
#[test]
fn test_large_dimensions()
```

**Integration tests:**

- Test with NumPy.dot() for various shapes
- Verify broadcasting rules match NumPy
- Test numerical accuracy (within tolerance)

**Performance benchmarks:**

- Benchmark dot() vs NumPy for various sizes (100, 1000, 10000)
- Test SIMD optimization impact

### E. Success Criteria

- [ ] All tensor dot tests pass
- [ ] All dimension combinations work correctly
- [ ] Broadcasting matches NumPy exactly
- [ ] Shape mismatches produce clear errors
- [ ] Performance within 2x of NumPy for 2D×2D
- [ ] NumPy conformance tests pass
- [ ] No unsafe code
- [ ] Documentation complete with examples


---

## Issue #60: [rust-numpy] Implement Least Squares Solver

### A. Technical Specification

**What this feature does:**

- Solves linear least squares problem: minimize ||Ax - b||^2
- Returns x, residuals, rank, singular values
- Handles both full-rank and rank-deficient matrices

**Technical requirements:**

- QR method for full-rank matrices (faster)
- SVD method for rank-deficient matrices (Moore-Penrose pseudo-inverse)
- Support multiple right-hand sides (2D b matrix)
- Compute residuals (sum of squared errors)
- Determine matrix rank from singular values

**NumPy API requirements:**

```python
x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
```

- Input: a (m×n matrix), b (m,) or (m, k) array
- Output: Tuple of (x, residuals, rank, s) where:
  - x: (n,) or (n, k) solution
  - residuals: (k,) or () sum of squared errors
  - rank: integer matrix rank
  - s: (min(m,n),) singular values of A

**Parameters:**

- `rcond`: Cutoff for singular values (values < rcond \* s_max are treated as zero)
- Default: max(m,n) \* eps where eps is machine precision

**Special cases to handle:**

- Non-2D arrays (error)
- Empty arrays
- Over-determined (m > n)
- Under-determined (m < n)
- Rank-deficient matrices
- Multiple right-hand sides (2D b)

**Algorithms:**

1. QR-based: Full-rank case, faster, x = R^{-1} @ Q^T @ b
2. SVD-based: Rank-deficient case, x = V @ Σ^+ @ U^T @ b where Σ^+ is pseudo-inverse

### B. Code Structure

**Files to create/modify:**

- `src/linalg/solvers.rs` - Extend existing with lstsq()
- `src/linalg/mod.rs` - Export lstsq() function
- `tests/lstsq_tests.rs` - Add least squares tests

**Function signatures:**

```rust
pub fn lstsq<T>(
    a: &Array<T>,
    b: &Array<T>,
    rcond: Option<f64>,
) -> Result<LstsqResult<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

pub struct LstsqResult<T> {
    x: Array<T>,
    residuals: Array<f64>,
    rank: usize,
    s: Array<f64>,
}

// QR-based least squares
pub fn lstsq_qr<T>(
    a: &Array<T>,
    b: &Array<T>,
) -> Result<Array<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;

// SVD-based least squares (pseudo-inverse)
pub fn lstsq_svd<T>(
    a: &Array<T>,
    b: &Array<T>,
    rcond: f64,
) -> Result<LstsqResult<T>, NumPyError>
where T: LinalgScalar + Clone + 'static;
```

**Data structures to define:**

```rust
// SVD pseudo-inverse helper
struct PseudoInverse {
    v: Array<Complex64>,
    sigma_plus: Array<f64>,
    u_h: Array<Complex64>,
}
```

**Integration points:**

- QR decomposition - Use in QR-based solving
- SVD decomposition - Use in SVD-based solving
- Matrix multiplication - Use in both methods
- Existing solvers module - Extend with lstsq()

### C. Implementation Steps

**Phase 1: QR-based Solver (8-10 hours)**

1. [ ] Review existing `src/linalg/solvers.rs`
2. [ ] Implement `lstsq_qr()`:
   - Compute QR: A = Q @ R
   - Solve R @ x = Q^T @ b (triangular solve)
   - Return x (n,) or (n, k)
3. [ ] Compute residuals:
   - Residuals = ||Ax - b||^2 for each column of b
   - Sum to get total residual
4. [ ] Handle multiple right-hand sides (2D b)

**Phase 2: SVD-based Solver (10-12 hours)** 5. [ ] Compute SVD: A = U @ Σ @ V^T 6. [ ] Implement `lstsq_svd()`: - Compute Σ^+ (pseudo-inverse of Σ) - Set 1/s for s_i > rcond _ s_max, 0 otherwise - Compute x = V @ Σ^+ @ U^T @ b 7. [ ] Compute rank: - Count singular values > rcond _ s_max - Return integer rank 8. [ ] Compute residuals (same as QR method)

**Phase 3: Integration & Testing (4-6 hours)** 9. [ ] Create public `lstsq()` function: - Check condition number of A - Use QR if well-conditioned (fast path) - Use SVD if ill-conditioned or rank-deficient - Default rcond = max(m,n) \* eps 10. [ ] Return (x, residuals, rank, s) tuple 11. [ ] Handle 1D b and 2D b correctly 12. [ ] Add unit tests: - Full-rank cases - Rank-deficient cases - Multiple right-hand sides 13. [ ] Verify NumPy conformance (compare x, residuals, rank, s) 14. [ ] Benchmark vs NumPy 15. [ ] Document API with examples

### D. Testing Requirements

**Unit tests in `tests/lstsq_tests.rs`:**

```rust
#[test]
fn test_lstsq_overdetermined()
#[test]
fn test_lstsq_underdetermined()
#[test]
fn test_lstsq_rank_deficient()
#[test]
fn test_lstsq_multiple_rhs()
#[test]
fn test_lstsq_rcond()
#[test]
fn test_lstsq_residuals()
#[test]
fn test_lstsq_rank()
#[test]
fn test_lstsq_accuracy()
#[test]
fn test_lstsq_qr_vs_svd()
```

**Integration tests:**

- Verify Ax approximates b (within tolerance)
- Test with NumPy.linalg.lstsq for various problems
- Test both QR and SVD paths

**Performance benchmarks:**

- Benchmark lstsq vs NumPy for various sizes (10, 50, 100, 500)

### E. Success Criteria

- [ ] All least squares tests pass
- [ ] QR method works for full-rank matrices
- [ ] SVD method handles rank-deficient correctly
- [ ] Residuals computed correctly
- [ ] Rank matches expected
- [ ] Both 1D and 2D b supported
- [ ] NumPy conformance tests pass
- [ ] Performance within 2x of NumPy
- [ ] No unsafe code
- [ ] Documentation complete with examples


---

## Issue #61: [rust-numpy] Complete Unique Implementation

### A. Technical Specification

**What this feature does:**

- Finds unique elements in array with optional additional outputs
- Returns sorted unique elements (already done)
- Adds return_index, return_inverse, return_counts, axis parameters

**Technical requirements:**

- Return sorted unique elements
- Add return_index parameter (indices of first occurrences)
- Add return_inverse parameter (indices to reconstruct original)
- Add return_counts parameter (counts of each unique element)
- Add axis parameter support (find unique along axis)

**NumPy API requirements:**

```python
np.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)
```

### B. Code Structure

**Files to modify:**

- `src/set_ops.rs` - Complete implementation
- `tests/set_ops_tests.rs` - Add tests

### C. Implementation Steps

1. [ ] Review current unique() implementation
2. [ ] Implement return_index parameter
3. [ ] Implement return_inverse parameter
4. [ ] Implement return_counts parameter
5. [ ] Implement axis parameter support
6. [ ] Add comprehensive tests

### D. Testing Requirements

Test all parameter combinations, axis handling, NaN values, edge cases.

### E. Success Criteria

- [ ] All tests pass
- [ ] 100% NumPy parity
- [ ] Performance acceptable


---

## Issue #62: Identify Gaps

**Task ID:** task-1
**Prompt:** Identify all remaining todos required for a 100% port to Rust with 100% parity.

---

## Issue #64: [High] Run `./ralph/loop.sh plan` to populate this plan from `ralph/specs/*`.

Created from ralph/IMPLEMENTATION_PLAN.md.

---

## Issue #65: [rust-numpy] Add PARITY.md and link checklist to tests

Create PARITY.md as authoritative progress ledger with sections:
- Layout
- Broadcast
- DType
- UFunc
- Reduce

Each section must link to a test file path (stubs allowed initially).

Acceptance Criteria:
- PARITY.md exists with all sections
- Each section links to test file
- Can track progress against NumPy parity

---

## Issue #66: [TTS] Fix generation skipping parts of sentences

Fix split_into_best_sentences() in pocket_tts/models/tts_model.py (lines 875-956) which causes TTS to skip parts of long text.

Root causes:
1. tokenizer.sp.decode() may not preserve whitespace when decoding partial sequences
2. Comma-separated clauses not handled by end-of-sentence detection
3. Token boundary issues only looking for .!? tokens

Proposed fix:
- Use regex-based sentence splitting (preserves original text exactly)
- Only use tokenizer for counting, not text reconstruction
- Add strict validation that no content is lost
- Don't split on commas, only sentence terminators

Acceptance Criteria:
- Tale of Two Cities test case passes (all clauses spoken)
- No content loss for any input text
- Unit tests added for sentence splitting
- All existing tests pass

---

## Issue #67: [TTS] Verify docker-compose.yaml exists and is complete

Investigate whether docker-compose.yaml already exists in the repo (appears in README.md lines 87-100).

If it exists:
- Verify it meets all requirements from issue #4
- Update issue #4 status

If incomplete or missing:
- Create docker-compose.yaml with:
  - Available environment variables (POCKET_TTS_ALLOWED_ORIGINS)
  - Port exposure (port 8000)
  - Volume for model persistence

Acceptance Criteria:
- docker-compose.yaml exists and works
- Can run: docker compose up -d
- Models persist in volume
- All environment variables documented

---

## Issue #68: [TTS] Reduce memory usage - Part 1: Slice kv-cache to voice length

When caching model state for a given voice, we store full kv cache while we could slice it and only keep the first N elements.

N = number of frames in voice sample.

Files to modify:
- pocket_tts/models/tts_model.py (kv-cache handling)

Acceptance Criteria:
- kv-cache is sliced to voice sample length
- Memory usage reduced for voice caching
- All tests pass

Depends on: None

---

## Issue #69: [TTS] Reduce memory usage - Part 2: Keep weights in bfloat16

We load weights in bfloat16 but convert to float32 and do all computation in float32.

Implement:
- Option to keep weights in bfloat16
- Perform computation in bfloat16 or float32 where beneficial


Files to modify:
- pocket_tts/models/ (model weight loading and computation)

Acceptance Criteria:
- Weights can be kept in bfloat16
- Configurable computation dtype
- Memory usage reduced
- Accuracy preserved

Depends on: Memory usage Part 1

---

## Issue #70: [TTS] Reduce memory usage - Part 3: Dynamically size kv-cache

Currently kv-cache size is hardcoded at 1000 steps for both flowlm and mimi.

Implement:
- Calculate max kv-cache size from text token count
- Use educated guess (lower than 1000 in typical cases)

Files to modify:
- pocket_tts/models/tts_model.py
- pocket_tts/models/flow_lm.py
- pocket_tts/models/mimi.py

Acceptance Criteria:
- kv-cache sized based on input text length
- Memory usage reduced for typical inputs
- All tests pass

Depends on: Memory usage Part 1, Part 2

---

## Issue #71: [TTS] Fuse MimiStreamingMultiheadAttention and StreamingMultiheadAttention

We have two nearly identical transformer implementations:
- MimiStreamingMultiheadAttention (pocket_tts/modules/mimi_transformer.py line 65)
- StreamingMultiheadAttention (pocket_tts/modules/transformer.py line 34)

Goal:
- Keep only one implementation
- Or fuse them together to reduce duplication

Acceptance Criteria:
- Single transformer implementation
- No logic duplication
- All existing functionality preserved
- Tests pass

---

## Issue #72: [TTS] Integrate Rust-numpy via C-ABI

Export all required functions via C-ABI and rewrite numpy_rs.py to use ctypes.

See thoughts/plans/rust_numpy_parity_gaps.md for details.

Deliverables:
- Export all required functions via C-ABI
- Rewrite numpy_rs.py to use ctypes
- Remove PyO3 dependencies

Acceptance Criteria:
- C-ABI exports work correctly
- Python shim uses ctypes successfully
- All tests pass
- No PyO3 dependencies remaining

---

## Issue #73: [rust-numpy] Derive contiguity (C/F) and layout invariants

Implement derived contiguity checks and core layout invariants.

Acceptance Criteria:
- Layout::is_c_contiguous()
- Layout::is_f_contiguous()
- Tests cover: 0-D, 1-D, and canonical 2-D cases

Important: Contiguity must be derived from shape/strides (do not store flags blindly).

---

## Issue #74: [rust-numpy] Transpose-as-view (stride permutation)

Implement transpose as a pure view (no copy).

Acceptance Criteria:
- Layout::transpose(axes: Option<&[usize]>) -> Layout
- Tests validate shape/strides correctness for default and explicit axes

---

## Issue #75: [rust-numpy] Broadcast layout (stride=0 where dim=1)

Add broadcast_layout that produces stride=0 broadcasted views.

Acceptance Criteria:
- broadcast_layout(layout, out_shape) -> Layout
- Tests validate stride=0 behavior and errors on incompatible shapes

---

## Issue #76: [rust-numpy] Minimal correct N-D iterator offsets

Implement a correct baseline N-D iterator/planner.

Acceptance Criteria:
- Given broadcasted layouts, iterator yields correct per-operand element offsets
- Tests verify offsets for small shapes against expected sequences

---

## Issue #77: [rust-numpy] Numeric promotion rules for Add/Sub/Mul/TrueDiv/Comparison/Bitwise

Expand dtype promotion into an explicit, auditable rule set.

Acceptance Criteria:
- promote(left,right,op) supports Bool/Int/UInt/Float/Complex for:
  Add, Sub, Mul, TrueDiv, Comparison, Bitwise
- Tests include:
  - int + float -> float
  - float + complex -> complex
  - bitwise rejects float/complex with typed error

Use fixed-width dtypes only (no platform int).

---

## Issue #78: [rust-numpy] Casting policy skeleton (Safe/SameKind/Unsafe)

Implement can_cast(from,to,safety) metadata rules.

Acceptance Criteria:
- can_cast implements Safe/SameKind/Unsafe categories
- Tests cover representative pairs

---

## Issue #79: [rust-numpy] UFunc registry + kernel lookup by signature

Implement registry and kernel selection by dtype signature.

Acceptance Criteria:
- Registry registers and retrieves UFuncs by name
- Kernel selection by exact signature works
- Tests cover registry + selection

---

## Issue #80: [rust-numpy] Minimal execution engine (contiguous baseline) for binary ufunc

Execute a selected 1-D kernel over planned runs (start contiguous).

Acceptance Criteria:
- Given kernel + layouts + buffers, exec succeeds for contiguous case
- Tests: add_f64 contiguous correctness

---

## Issue #81: [rust-numpy] Broadcast-aware binary exec path (elementwise add)

Make binary execution broadcast-correct.

Acceptance Criteria:
- add works for broadcasted shapes (e.g. (3,1)+(1,4)->(3,4))
- Tests cover mixed broadcasting patterns

---

## Issue #82: [rust-numpy] Public Array facade + add() wired end-to-end

Expose minimal public API for Array + add.

Acceptance Criteria:
- api::Array supports basic construction (start with f64)
- api::ops::add(&Array,&Array)->Array
- Tests validate public API behavior

---

## Issue #83: [rust-numpy] Add mul ufunc (mirror add) + tests

Implement multiplication ufunc with the same pathway as add.

Acceptance Criteria:
- mul works for contiguous and broadcasted inputs
- Tests mirror add coverage for mul

---

## Issue #84: [rust-numpy] Global sum reduction for f64

Implement sum reduction over all axes.

Acceptance Criteria:
- sum(Array)->scalar (or 0-D Array) for f64
- Tests define and enforce empty-array policy

---

## Issue #85: [rust-numpy] sum(axis=..., keepdims=...): single axis first

Implement sum over a single axis with keepdims.

Acceptance Criteria:
- sum(axis=i, keepdims=bool) correct for small shapes
- Tests validate resulting shape + values

---

## Issue #86: [rust-numpy] Dimension coalescing into fewer contiguous runs

Optimize iteration by coalescing dimensions into fewer 1-D kernel calls.

Acceptance Criteria:
- Outputs identical to baseline across existing test suite
- Basic perf sanity check (bench optional)

---

## Issue #87: [rust-numpy] SIMD kernels (feature-gated) + runtime dispatch

Add optional SIMD-specialized kernels with runtime dispatch.

Acceptance Criteria:
- Baseline path remains default and passes all tests
- SIMD feature passes identical tests when enabled
- Dispatch chooses best available implementation safely

---

## Issue #88: [rust-numpy] Threading policy for safe kernels (no overlap/alias hazards)

Parallelize only where safe and deterministic under defined rules.

Acceptance Criteria:
- Threading is conditional and respects aliasing constraints
- Tests confirm correctness; determinism where required

---

## Issue #89: [rust-numpy] Identify all remaining todos for 100% Rust port with 100% parity

Identify all remaining todos required for a 100% port to Rust with 100% parity.

This is task-1 from the current plan.

Deliverables:
- Complete audit of all NumPy functionality
- Gap analysis of what's missing
- Prioritized list of remaining work

Acceptance Criteria:
- Comprehensive gap analysis complete
- All missing functionality identified
- Prioritization done

Depends on: All core rust-numpy issues

---

## Issue #90: [TTS] Investigate text input pauses (silence generation)

Research if it's possible to generate pauses by adding tags in text input without re-training.

From issue #8: "We don't know if it's possible, but it would be nice to have a way to generate pauses by adding some tag in text. If someone finds a way to do this without re-training, that would be cool :)"

Acceptance Criteria:
- Research complete
- Feasibility documented
- If feasible: Implementation plan
- If not feasible: Documentation of why not possible

---

## Issue #91: [TTS] Research quantization to int8

Research and implement int8 quantization for model parts that handle it well.

From issue #7: "Some parts of the model might handle quantization better than others, so it's possible we have to run some parts of the model in int8 and some other in bfloat16 or even float32."

Notes:
- int8 unlikely to be useful (rare CPU support)
- Stick to int8
- Investigate which parts quantize well

Acceptance Criteria:
- Quantization feasibility analysis complete
- Parts identified for int8 vs bfloat16 vs float32
- If feasible: Implementation
- Memory savings measured
- Accuracy impact measured

---

## Issue #92: Resolve README.md merge conflict

Fix git merge conflict markers (lines 49-54) in README.md. The conflict appears to be between HEAD and commit 7a4142b regarding torch.compile feature documentation.

---

## Issue #93: Fix generation skipping parts of sentences (High Priority)

Fix split_into_best_sentences() in pocket_tts/models/tts_model.py (lines 875-956) which causes TTS to skip parts of long text. Root causes: 1) tokenizer.sp.decode() may not preserve whitespace when decoding partial sequences 2) Comma-separated clauses not handled by end-of-sentence detection 3) Token boundary issues only looking for .!? tokens. Proposed fix: Use regex-based sentence splitting (preserves original text exactly), only use tokenizer for counting (not text reconstruction), add strict validation that no content is lost, don't split on commas (only sentence terminators). Acceptance Criteria: Tale of Two Cities test case passes (all clauses spoken), no content loss for any input text, unit tests added for sentence splitting, all existing tests pass.

---

## Issue #94: Verify docker-compose.yaml exists and is complete

Investigate whether docker-compose.yaml already exists in repo (appears in README.md lines 87-100). If it exists: verify it meets all requirements from issue #4 and update issue #4 status. If incomplete or missing: create docker-compose.yaml with available environment variables (POCKET_TTS_ALLOWED_ORIGINS), port exposure (port 8000), volume for model persistence. Acceptance Criteria: docker-compose.yaml exists and works, can run: docker compose up -d, models persist in volume, all environment variables documented.

---

## Issue #95: Reduce memory usage - Part 1: Slice kv-cache to voice length

When caching model state for a given voice, we store full kv cache while we could slice it and only keep first N elements. N = number of frames in voice sample. Files to modify: pocket_tts/models/tts_model.py (kv-cache handling). Acceptance Criteria: kv-cache is sliced to voice sample length, memory usage reduced for voice caching, all tests pass. Depends on: None.

---

## Issue #96: Reduce memory usage - Part 2: Keep weights in bfloat16

We load weights in bfloat16 but convert to float32 and do all computation in float32. Implement: option to keep weights in bfloat16, perform computation in bfloat16 or float32 where beneficial. Files to modify: pocket_tts/models/ (model weight loading and computation). Acceptance Criteria: Weights can be kept in bfloat16, configurable computation dtype, memory usage reduced, accuracy preserved. Depends on: Memory usage Part 1.

---

## Issue #97: Reduce memory usage - Part 3: Dynamically size kv-cache

Currently kv-cache size is hardcoded at 1000 steps for both flowlm and mimi. Implement: calculate max kv-cache size from text token count, use educated guess (lower than 1000 in typical cases). Files to modify: pocket_tts/models/tts_model.py, pocket_tts/models/flow_lm.py, pocket_tts/models/mimi.py. Acceptance Criteria: kv-cache sized based on input text length, memory usage reduced for typical inputs, all tests pass. Depends on: Memory usage Part 1, Part 2.

---

## Issue #98: Fuse MimiStreamingMultiheadAttention and StreamingMultiheadAttention

We have two nearly identical transformer implementations: MimiStreamingMultiheadAttention (pocket_tts/modules/mimi_transformer.py line 65) and StreamingMultiheadAttention (pocket_tts/modules/transformer.py line 34). Goal: Keep only one implementation or fuse them together to reduce duplication. Acceptance Criteria: Single transformer implementation, no logic duplication, all existing functionality preserved, tests pass.

---

## Issue #99: Integrate Rust-numpy via C-ABI

Export all required functions via C-ABI and rewrite numpy_rs.py to use ctypes. See thoughts/plans/rust_numpy_parity_gaps.md for details. Deliverables: Export all required functions via C-ABI, rewrite numpy_rs.py to use ctypes, remove PyO3 dependencies. Acceptance Criteria: C-ABI exports work correctly, Python shim uses ctypes successfully, all tests pass, no PyO3 dependencies remaining.

---

## Issue #100: Implement tensor operations with axes support

Several tensor operations in linalg.rs currently return not_implemented() errors when called with axes parameters. Affected Functions: tensor_solve (lines 515-521) - Full tensor solve with axes, tensor_inv (lines 589, 604) - Tensor inverse with axes, tensor_dot (line 409) - Higher-dim (>2D) tensor dot, diagonal_enhanced (line 744) - Custom axes for diagonal. Proposed Implementation: 1) Use reshape + matmul + reshape strategy for tensor_dot 2) Permute axes, reshape to 2D, solve/invert, reshape back for tensor_solve/inv 3) Implement axis permutation for diagonal_enhanced. Files to Modify: rust-numpy/src/linalg.rs. Acceptance Criteria: tensor_solve works with axes parameter, tensor_inv works with axes parameter, tensor_dot works with >2D arrays, diagonal_enhanced works with custom axes, all tests pass.

---

## Issue #101: Implement nuclear and L-p norms in linalg::norm

The norm() function in linalg.rs is missing nuclear norm and general L-p norm implementations. Current Status: ✅ Frobenius norm ('fro'), ✅ Infinity norm ('inf'), ✅ Negative infinity norm ('-inf'), ✅ L-1 norm ('1'), ✅ L-2 norm ('2'), ❌ Nuclear norm ('nuc') - returns not_implemented(), ❌ L-p norms for p > 2 - returns not_implemented(). Proposed Implementation: Nuclear norm using SVD and sum of singular values, L-p norms using power function. Files to Modify: rust-numpy/src/linalg.rs - Lines 203, 231. Acceptance Criteria: Nuclear norm returns correct value, L-p norms work for any positive integer p, tests added for both.

---

## Issue #102: Implement sorting functions: argpartition and partition with array kth

argpartition and partition in sorting.rs return not_implemented() when called with an array-based kth parameter. Current Status: argpartition works with scalar kth but fails with array kth, partition works with scalar kth but fails with array kth. Proposed Implementation: Iterate over all kth values and use quickselect for each. Files to Modify: rust-numpy/src/sorting.rs - Lines 489, 523, 1117, 1166. Acceptance Criteria: argpartition works with array kth, partition works with array kth, tests added.

---

## Issue #103: Implement set operations in set_ops.rs

Several set operations in set_ops.rs return not_implemented() errors. Affected Functions: Set difference, Symmetric difference, Additional set utilities. Files to Modify: rust-numpy/src/set_ops.rs - Lines 162, 199, 220. Proposed Implementation: Use HashSet-based algorithms with proper handling for empty arrays. Acceptance Criteria: All set operations implemented, edge cases for empty arrays handled, tests added.

---

## Issue #104: Verify and implement bitwise operations

Bitwise universal functions and their corresponding tests are currently missing or unverified. Next steps: Implement bitwise_and, bitwise_or, bitwise_xor, invert, left_shift, right_shift, add comprehensive tests in src/bitwise.rs and tests/conformance/.

---

## Issue #105: Complete Dtype system (intp, uintp, IEEE 754 f16)

The current Dtype system in rust-numpy is missing critical support for: intp/uintp (platform-dependent indexing types used throughout NumPy), IEEE 754 f16 (current implementation uses simple bit-shifts which leads to incorrect results for many half-precision operations). Next steps: Use 'half' crate to implement a proper f16, add Intp and Uintp variants to Dtype enum, update parsing and itemsize logic in src/dtype.rs.

---

## Issue #106: Workspace lint/format enforcement (fmt + clippy -D warnings)

Goal: Enforce consistent formatting and linting. Acceptance Criteria: CI runs cargo fmt --check, CI runs cargo clippy --workspace -- -D warnings. Depends on: #29 (recommended, not required).

---

## Issue #107: Slicing-as-view (ranges + step; support negative step)

Goal: Implement slicing as views with steps. Acceptance Criteria: Slice spec supports full range ':', start..end, step (including negative), tests cover positive and negative strides. Depends on: #31.

---

## Issue #108: Derive contiguity (C/F) and layout invariants

Implement derived contiguity checks and core layout invariants. Acceptance Criteria: Layout::is_c_contiguous(), Layout::is_f_contiguous(), tests cover 0-D, 1-D, and canonical 2-D cases. Important: Contiguity must be derived from shape/strides (do not store flags blindly).

---

## Issue #109: Transpose-as-view (stride permutation)

Implement transpose as a pure view (no copy). Acceptance Criteria: Layout::transpose(axes: Option<&[usize]>) -> Layout, tests validate shape/strides correctness for default and explicit axes.

---

## Issue #110: Broadcast layout (stride=0 where dim=1)

Add broadcast_layout that produces stride=0 broadcasted views. Acceptance Criteria: broadcast_layout(layout, out_shape) -> Layout, tests validate stride=0 behavior and errors on incompatible shapes.

---

## Issue #111: Minimal correct N-D iterator offsets

Implement a correct baseline N-D iterator/planner. Acceptance Criteria: Given broadcasted layouts, iterator yields correct per-operand element offsets, tests verify offsets for small shapes against expected sequences. Depends on: #34.

---

## Issue #112: Numeric promotion rules for Add/Sub/Mul/TrueDiv/Comparison/Bitwise

Expand dtype promotion into an explicit, auditable rule set. Acceptance Criteria: promote(left,right,op) supports Bool/Int/UInt/Float/Complex for Add, Sub, Mul, TrueDiv, Comparison, Bitwise, tests include int + float -> float, float + complex -> complex, bitwise rejects float/complex with typed error. Use fixed-width dtypes only (no platform int).

---

## Issue #113: Casting policy skeleton (Safe/SameKind/Unsafe)

Implement can_cast(from,to,safety) metadata rules. Acceptance Criteria: can_cast implements Safe/SameKind/Unsafe categories, tests cover representative pairs. Depends on: #36 (recommended).

---

## Issue #114: UFunc registry + kernel lookup by signature

Implement registry and kernel selection by dtype signature. Acceptance Criteria: Registry registers and retrieves UFuncs by name, kernel selection by exact signature works, tests cover registry + selection. Depends on: #36 (promotion integration next).

---

## Issue #115: Minimal execution engine (contiguous baseline) for binary ufunc

Execute a selected 1-D kernel over planned runs (start contiguous). Acceptance Criteria: Given kernel + layouts + buffers, exec succeeds for contiguous case, tests: add_f64 contiguous correctness. Depends on: #35, #38.

---

## Issue #116: Broadcast-aware binary exec path (elementwise add)

Make binary execution broadcast-correct. Acceptance Criteria: add works for broadcasted shapes (e.g. (3,1)+(1,4)->(3,4)), tests cover mixed broadcasting patterns. Depends on: #34, #35, #39.

---

## Issue #117: Public Array facade + add() wired end-to-end

Expose minimal public API for Array + add. Acceptance Criteria: api::Array supports basic construction (start with f64), api::ops::add(&Array,&Array)->Array, tests validate public API behavior. Depends on: #40.

---

## Issue #118: Add mul ufunc (mirror add) + tests

Implement multiplication ufunc with same pathway as add. Acceptance Criteria: mul works for contiguous and broadcasted inputs, tests mirror add coverage for mul. Depends on: #41.

---

## Issue #119: Global sum reduction for f64

Implement sum reduction over all axes. Acceptance Criteria: sum(Array)->scalar (or 0-D Array) for f64, tests define and enforce empty-array policy. Depends on: #41.

---

## Issue #120: sum(axis=..., keepdims=...): single axis first

Implement sum over a single axis with keepdims. Acceptance Criteria: sum(axis=i, keepdims=bool) correct for small shapes, tests validate resulting shape + values. Depends on: #43.

---

## Issue #121: Dimension coalescing into fewer contiguous runs

Optimize iteration by coalescing dimensions into fewer 1-D kernel calls. Acceptance Criteria: Outputs identical to baseline across existing test suite, basic perf sanity check (bench optional). Depends on: #40, #44.

---

## Issue #122: SIMD kernels (feature-gated) + runtime dispatch

Add optional SIMD-specialized kernels with runtime dispatch. Acceptance Criteria: Baseline path remains default and passes all tests, SIMD feature passes identical tests when enabled, dispatch chooses best available implementation safely. Depends on: #45.

---

## Issue #123: Threading policy for safe kernels (no overlap/alias hazards)

Parallelize only where safe and deterministic under defined rules. Acceptance Criteria: Threading is conditional and respects aliasing constraints, tests confirm correctness and determinism where required. Depends on: #45.

---

## Issue #124: Phase 1: FFT Module Implementation

Complete all 6 stubbed FFT functions using rustfft crate. See thoughts/plans/fft-module-implementation.md for details. Deliverables: fft(), ifft(), rfft2(), irfft2(), rfftn(), irfftn(), hilbert().

---

## Issue #125: Phase 2: Advanced Array Methods

Implement advanced array manipulation methods. See thoughts/plans/advanced-array-methods-and-nan-stats.md for details. Deliverables: flatten(), ravel(), squeeze(), repeat(), tile(), swapaxes(), rollaxis(), moveaxis(), atleast_1d/2d/3d().

---

## Issue #126: Phase 3: NaN-Aware Statistics

Implement statistical functions that properly handle NaN values. See thoughts/plans/advanced-array-methods-and-nan-stats.md for details. Deliverables: nanmean(), nanstd(), nanvar(), nanmedian(), nansum(), nanprod().

---

## Issue #127: Phase 4: Advanced Indexing & Slicing

Implement fancy indexing, boolean masking, and ellipsis indexing. See thoughts/plans/advanced-indexing-and-additional-gaps.md for details. Deliverables: Fancy indexing, Boolean masking, Ellipsis indexing, Field access, NewAxis support.

---

## Issue #128: Phase 5: Missing Utility Ufuncs

Add utility functions for NaN/Inf/Finite checking and angle conversions. See thoughts/plans/advanced-indexing-and-additional-gaps.md for details. Deliverables: isnan(), isinf(), isfinite(), deg2rad(), rad2deg(), Array versions of above.

---

## Issue #129: Phase 6: Polynomial Module Comparison

Audit and complete polynomial module. See thoughts/plans/advanced-indexing-and-additional-gaps.md for details. Deliverables: Audit existing functions, implement missing: roots(), poly(), polyval(), polyfit(), comprehensive tests.

---

## Issue #130: Phase 7: Test Coverage Expansion

Expand test coverage to include NumPy conformance testing. See thoughts/plans/rust-numpy-port-master-execution-plan.md for details. Deliverables: Conformance tests against NumPy, edge case testing, performance benchmarks.

---

## Issue #131: Add PARITY.md and link checklist to tests

Create PARITY.md as authoritative progress ledger with sections: Layout, Broadcast, DType, UFunc, Reduce. Each section must link to a test file path (stubs allowed initially). Acceptance Criteria: PARITY.md exists with all sections, each section links to test file, can track progress against NumPy parity.

---

## Issue #132: Implement Eigen-decomposition

Computes eigenvalues and eigenvectors of square matrices using Francis double-shift QR iteration algorithm with Hessenberg reduction. Supports both real and complex matrices. NumPy API: np.linalg.eig(a) returns (eigenvalues, eigenvectors). Handle non-square matrices (error), singular matrices, defective matrices, empty matrices, 1×1 matrices. Implementation in 4 phases: Foundation (8-10h), Core Algorithm (15-20h), Integration & Testing (7-10h), Optimization (8-12h optional). Files: rust-numpy/src/linalg/eigen.rs, rust-numpy/src/linalg/mod.rs, rust-numpy/tests/eigen_tests.rs.

---

## Issue #133: Implement QR Decomposition

Decomposes matrix A into Q (orthonormal/unitary) and R (upper triangular). Supports 4 NumPy modes: 'reduced', 'complete', 'r', 'raw'. Handles rectangular matrices (both m>n and m<n). Use Householder reflections algorithm. NumPy API: np.linalg.qr(a, mode='reduced'). Handle non-2D arrays (error), empty arrays, 1×N matrices, complex matrices. Implementation in 3 phases: Householder Reflections (6-8h), QR Factorization (10-12h), Integration & Testing (4-5h). Files: rust-numpy/src/linalg/decompositions.rs, rust-numpy/src/linalg/mod.rs, rust-numpy/tests/qr_tests.rs.

---

## Issue #134: Implement SVD (Singular Value Decomposition)

Decomposes matrix A into U @ Σ @ V^T where U and V are unitary, Σ is diagonal. Computes singular values (always non-negative, sorted descending). Supports full_matrices and compute_uv parameters. Use Golub-Kahan bidiagonalization algorithm with implicit QR for bidiagonal SVD. NumPy API: np.linalg.svd(a, full_matrices=True, compute_uv=True, hermitian=False). Handle non-2D arrays (error), empty arrays, rank-deficient matrices, complex matrices. Implementation in 3 phases: Bidiagonalization (10-12h), Bidiagonal SVD (12-15h), Integration & Testing (8-10h). Files: rust-numpy/src/linalg/decompositions.rs, rust-numpy/src/linalg/mod.rs, rust-numpy/tests/svd_tests.rs.

---

## Issue #135: Implement Multi-dimensional Dot Products

Implements dot(), matmul(), and tensordot() for N-D arrays with broadcasting. Optimizes for common cases (2D×2D matrix multiplication). Support different dimension combinations: 1D × 1D → inner product (scalar), 1D × 2D → matrix-vector multiplication, 2D × 1D → vector-matrix multiplication, 2D × 2D → matrix multiplication, N-D × N-D → tensor contraction. Proper broadcasting rules (NumPy semantics). Implementation in 4 phases: Foundation (6-8h), Core Operations (8-10h), Additional Functions (4-5h), Testing & Optimization (3-4h). Files: rust-numpy/src/linalg/products.rs, rust-numpy/src/linalg/mod.rs, rust-numpy/tests/tensor_dot_tests.rs.

---

## Issue #136: Implement Least Squares Solver

Solves linear least squares problem: minimize ||Ax - b||^2. Returns x, residuals, rank, singular values. Handles both full-rank and rank-deficient matrices. QR method for full-rank matrices (faster), SVD method for rank-deficient matrices (Moore-Penrose pseudo-inverse). Support multiple right-hand sides (2D b matrix). NumPy API: x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None). Handle over-determined (m > n), under-determined (m < n), rank-deficient matrices, multiple right-hand sides. Implementation in 3 phases: QR-based Solver (8-10h), SVD-based Solver (10-12h), Integration & Testing (4-6h). Files: rust-numpy/src/linalg/solvers.rs, rust-numpy/src/linalg/mod.rs, rust-numpy/tests/lstsq_tests.rs.

---

## Issue #137: Complete Unique Implementation

Finds unique elements in array with optional additional outputs. Currently returns sorted unique elements. Add return_index parameter (indices of first occurrences), return_inverse parameter (indices to reconstruct original), return_counts parameter (counts of each unique element), axis parameter support (find unique along axis). NumPy API: np.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None). Implementation steps: Review current unique() implementation, implement return_index parameter, implement return_inverse parameter, implement return_counts parameter, implement axis parameter support, add comprehensive tests. Files: rust-numpy/src/set_ops.rs, rust-numpy/tests/set_ops_tests.rs.

---

## Issue #138: Identify all remaining todos for 100% Rust port with 100% parity

Task ID: task-1. Identify all remaining todos required for a 100% port to Rust with 100% parity. This involves comprehensive gap analysis across all NumPy modules and features.

---

## Issue #139: Investigate text input pauses (silence generation)

Research and implement the ability to add silence in text input to generate pauses. This is a feature explicitly mentioned in the README as unsupported but desired. Reference: https://github.com/kyutai-labs/pocket-tts/issues/6.

---

## Issue #140: Research quantization to int8

Investigate and implement quantization to run the computation in int8. This is a feature explicitly mentioned in the README as unsupported but desired. Reference: https://github.com/kyutai-labs/pocket-tts/issues/7. Benefits include reduced memory usage and potentially faster inference.

---

