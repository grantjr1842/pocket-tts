# Rust-NumPy vs NumPy 2.x Gap Analysis

**Date:** 2026-01-24
**Scope:** Rust-NumPy API parity against NumPy 2.0 reference docs.

## Sources Reviewed

- `rust-numpy/INVENTORY.md` (function inventory)
- `rust-numpy/PARITY.md` (parity status + known gaps)
- NumPy 2.0 reference docs
  - https://numpy.org/doc/2.0/reference/routines.array-creation.html
  - https://numpy.org/doc/2.0/reference/routines.array-manipulation.html
  - https://numpy.org/doc/2.0/reference/routines.math.html
  - https://numpy.org/doc/2.0/reference/routines.sort.html
  - https://numpy.org/doc/2.0/reference/routines.statistics.html
  - https://numpy.org/doc/2.0/reference/routines.linalg.html

## Executive Summary

- **Overall parity estimate:** ~82% (per `rust-numpy/PARITY.md`).
- **Strong coverage:** array creation/manipulation, ufunc catalog, sorting, statistics, FFT API surface, random distributions, string ops.
- **Critical remaining gaps:** linear algebra completeness, ufunc execution engine, advanced broadcasting edge cases, SIMD/parallel tuning, test coverage.
- **Project posture:** Rust-first library with no Python bindings (per `rust-numpy/AGENTS.md`).

## Coverage by NumPy 2.0 Module

### Array Creation
**NumPy reference:** `routines.array-creation.html`.

**Status:** Nearly complete (95% per `PARITY.md`).

**Implemented (examples from inventory):**
- `arange`, `array`, `full`, `empty`, `zeros`, `ones`, `linspace`, `logspace`, `geomspace`, `vander`, `frombuffer`, `fromiter`, `fromfunction`.

**Open gaps:**
- No explicit gaps documented; parity file indicates near completion. Audit recommended for record arrays (`rec.*`) and char array builders (`numpy.char`).

### Array Manipulation
**NumPy reference:** `routines.array-manipulation.html`.

**Status:** Complete for core operations; some edge cases remain.

**Implemented (examples):**
- `reshape`, `ravel`, `flatten`, `squeeze`, `expand_dims`, `transpose`, `swapaxes`, `moveaxis`, `broadcast_to`, `concatenate`, `stack`, `split`, `tile`, `repeat`, `pad`, `rot90`, `flip`, `roll`, `unique`.

**Open gaps:**
- Advanced indexing edge cases (integer/boolean/fancy indexing completeness).
- `choose`, `compress` flagged as missing in `PARITY.md`.

### Mathematical Functions (ufuncs)
**NumPy reference:** `routines.math.html`.

**Status:** Function catalog complete; execution engine incomplete.

**Implemented (examples):**
- Trig: `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`
- Hyperbolic: `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh`
- Exponent/log: `exp`, `expm1`, `exp2`, `log`, `log10`, `log2`, `log1p`, `logaddexp`, `logaddexp2`
- Rounding: `round`, `around`, `rint`, `floor`, `ceil`, `trunc`, `fix`
- Misc: `sqrt`, `cbrt`, `fabs`, `absolute`, `clip`, `convolve`, `interp`

**Open gaps (from `PARITY.md`):**
- Ufunc execution engine: strided execution, kernel registry, dtype-specific kernels.
- Reduction ops with full axis/keepdims support.
- SIMD kernels and runtime dispatch (performance parity).

### Sorting, Searching, Counting
**NumPy reference:** `routines.sort.html`.

**Status:** Complete per inventory and parity.

**Implemented:**
- Sorting: `sort`, `argsort`, `lexsort`, `partition`, `argpartition`, `sort_complex`.
- Searching: `argmax`, `argmin`, `argwhere`, `nonzero`, `flatnonzero`, `where`, `searchsorted`, `extract`.
- Counting: `count_nonzero`.

**Open gaps:**
- No explicit gaps documented.

### Statistics
**NumPy reference:** `routines.statistics.html`.

**Status:** Complete per inventory and parity update.

**Implemented:**
- `mean`, `std`, `var`, `median`, `average`, `ptp`.
- `percentile`, `quantile`, `nan*` variants.
- `corrcoef`, `cov`, `histogram`, `histogram2d`, `histogramdd`, `bincount`, `digitize`.

**Open gaps:**
- No explicit gaps documented; verify NaN variants and axis handling coverage.

### Linear Algebra (numpy.linalg)
**NumPy reference:** `routines.linalg.html`.

**Status:** Partial (65% per `PARITY.md`).

**Implemented (inventory):**
- Products: `dot`, `vdot`, `inner`, `outer`, `matmul`, `kron`, `einsum`, `einsum_path`.
- Decompositions: `svd`, `qr`, `cholesky`.
- Eigen: `eig`, `eigh`, `eigvals`, `eigvalsh`.
- Norms: `norm`, `matrix_rank`, `cond`, `slogdet`, `det`.
- Solvers: `solve`, `lstsq`, `inv`, `pinv`, `tensorinv`, `tensorsolve`.

**Open gaps (from `PARITY.md`):**
- Parity document flags multiple missing linalg functions despite inventory entries. Needs reconciliation:
  - `qr`, `cholesky`, `eig`, `lstsq`, `inv`, `pinv`, `matrix_power`, `matrix_rank`, `det`, `trace`, `solve`, `tensorsolve`, `tensorinv`.
- Action: audit implementation completeness and API fidelity (parameters, broadcasting, error messages).

### FFT (numpy.fft)
**NumPy reference:** `routines.fft` (not fully linked in this review).

**Status:** Partial per `PARITY.md` (50%).

**Implemented (inventory):**
- `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfftn`, `irfftn`, `rfft2`, `irfft2`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`.

**Open gaps (from `PARITY.md`):**
- Multi-dimensional and real FFT variants flagged as incomplete; needs verification of API parity and coverage of all parameters.

### Random (numpy.random)
**Status:** ~90% complete per `PARITY.md` update.

**Implemented (examples):**
- Core: `rand`, `randn`, `random`, `random_sample`, `seed`, `choice`, `permutation`.
- Distributions: `beta`, `binomial`, `chisquare`, `gamma`, `normal`, `poisson`, `uniform`, etc.

**Open gaps:**
- Remaining distributions and generator compatibility; validate trait bounds and API parity.

### Strings / Char
**Status:** ~95% complete per `PARITY.md` update.

**Open gaps:**
- Unlisted functions in `numpy.char` should be audited against NumPy 2.0 reference.

### Datetime / Timedelta
**Status:** ~75% complete per `PARITY.md`.

**Open gaps:**
- Additional datetime utilities and full behavior parity.

### Masked Arrays (numpy.ma)
**Status:** ~75% complete per `PARITY.md`.

**Open gaps:**
- `choose` still missing; additional masked operations likely absent.

### Typing / Testing
**Status:** Typing missing; testing utilities partial.

**Open gaps:**
- `numpy.typing` equivalents.
- Conformance tests and coverage >80%.

## Cross-Reference Notes

- The function inventory lists many implementations that the parity file marks as missing. This suggests the primary gap is **API fidelity and completeness** (parameter behavior, broadcasting, dtype handling, error messages, and tests) rather than raw function presence.
- Recommendation: treat missing items as *parity gaps* until verified against NumPy 2.0 reference behavior.

## Priority Gap List (Actionable)

1. **Linear Algebra parity audit**
   - Confirm every `numpy.linalg.*` function matches signatures, broadcasting, and error semantics.

2. **Ufunc execution engine**
   - Strided execution + dtype kernel registry + reduction axis support.

3. **FFT completeness verification**
   - Validate multidimensional FFTs and real FFT parameter coverage.

4. **Advanced indexing completeness**
   - Fancy indexing edge cases, `newaxis`, and ellipsis behavior.

5. **Testing and conformance**
   - Establish parity tests for each module, including NaN/Inf edge cases.

## Output Status

This report consolidates the current Rust-NumPy parity view against NumPy 2.0 reference categories using existing project documentation and inventory data. Further verification should focus on behavioral parity gaps rather than raw function presence.
