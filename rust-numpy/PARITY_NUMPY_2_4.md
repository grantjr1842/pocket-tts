# Rust-NumPy Parity Report (NumPy 2.4)

**Last Updated:** 2026-01-24
**Scope:** Structural parity (public API surface) and implementation coverage vs NumPy 2.4.0
**Status:** Consolidated source of truth for parity and gaps

## Sources of Truth (inputs)

- `rust-numpy/NUMPY_2_4_FUNCTIONS.md` - list of public NumPy 2.4.0 callables
- `rust-numpy/PARITY_BASELINE_NUMPY_2_4.csv` - structural parity matrix (present / namespace_mismatch / missing)
- `rust-numpy/INVENTORY.md` - rust-numpy implementation inventory (generated)
- `rust-numpy/ENUM_PARITY_NUMPY_2_4.csv` - NumPy enum parity (generated)

## Definitions

- **Structural parity**: Symbol exists and is exported in the expected namespace.
- **Behavioral parity**: Signatures, broadcasting, dtype rules, and error behavior match NumPy.
- **present**: Symbol exported as expected.
- **namespace_mismatch**: Symbol exists but is not exported in the expected namespace.
- **missing**: Symbol not found in the module/codebase.

## Executive Summary

- rust-numpy has broad internal coverage, but structural parity is limited by export/binding gaps.
- The baseline CSV is authoritative for the public API surface; inventory reflects internal implementations.
- Enum parity is tracked separately (see Enum Parity section and CSV).

## Structural Parity Summary (from PARITY_BASELINE_NUMPY_2_4.csv)

| Module                   | Present | Namespace Mismatch | Missing | Audit Priority |
| ------------------------ | ------- | ------------------ | ------- | -------------- |
| numpy                    | 155/462 | 93                 | 214     | critical       |
| numpy.linalg             | 22/32   | 0                  | 10      | critical       |
| numpy.fft                | 18/18   | 0                  | 0       | high           |
| numpy.random             | 43/60   | 0                  | 17      | high           |
| numpy.polynomial         | 6/7     | 0                  | 1       | medium         |
| numpy.ma                 | 13/219  | 0                  | 206     | high           |
| numpy.char               | 41/53   | 0                  | 12      | medium         |
| numpy.testing            | 4/36    | 0                  | 32      | medium         |
| numpy.typing             | 0/2     | 0                  | 2       | low            |
| numpy.ndarray Methods    | 4/53    | 0                  | 49      | critical       |
| numpy.ndarray Properties | 6/17    | 0                  | 11      | critical       |

## Enum Parity (NumPy 2.4)

Enum parity is derived from `ENUM_PARITY_NUMPY_2_4.csv`. That file lists public Enum classes
found in NumPy 2.4 and matches them (by name) against rust-numpy `pub enum` definitions.

Summary (see CSV for full list):

- Total NumPy enums: 0
- Present in rust-numpy: 0
- Missing in rust-numpy: 0

**Audit note:** Enum audit executed with NumPy 2.4.0 using a Python 3.11 venv.

## Primary Gaps (prioritized)

1. **FFI export / namespace alignment**
   - Many functions exist internally but are not exported in the expected namespace.
   - Structural parity will remain low until exports and namespace wiring are complete.

2. **Behavioral parity validation**
   - Structural presence does not imply correctness (signatures, broadcasting, dtype rules, errors).
   - Use the checklist below for correctness-first audits.

3. **Conformance test coverage**
   - Add golden tests that compare against NumPy for representative datasets.

## Correctness-First Parity Checklist (NumPy 2.4)

### Global gates

- [ ] Signature parity: positional/keyword arguments, defaults, and return shapes match NumPy 2.4.
- [ ] Broadcasting parity: shape rules, stride-0 behavior, mixed-dim broadcasting.
- [ ] Dtype parity: promotions, casting rules, dtype-specific outputs.
- [ ] Error parity: error type and message semantics aligned.
- [ ] View semantics: non-contiguous arrays, slicing views, offsets.
- [ ] where/mask parity: masked execution and `where=` semantics.
- [ ] NaN/Inf parity: NaN propagation and edge behavior.

### numpy (top-level)

- [ ] Array creation (`array`, `arange`, `linspace`, `logspace`, `geomspace`) matches NumPy defaults.
- [ ] `dot`, `inner`, `outer`, `matmul`, `kron`, `vdot`, `trace`, `cross` semantics match NumPy.
- [ ] `where` broadcasting and scalar/array combinations match NumPy.
- [ ] `arg*`/`sort*` outputs match NumPy defaults.
- [ ] Top-level wrappers call correct ufunc kernels.

### numpy.ndarray Methods / Properties

- [ ] Implement missing methods per baseline, matching NumPy signatures and return types.
- [ ] In-place vs out-of-place behavior matches NumPy (`sort`, `resize`, `fill`, etc.).
- [ ] `T`, `real`, `imag`, `itemsize`, `nbytes`, `flags`, `base`, `flat` parity validated.

### numpy.linalg

- [ ] Matrix shape rules (1D/2D/batched) and broadcasting match NumPy.
- [ ] Error parity: singular matrices, non-square, invalid shapes, convergence failures.
- [ ] Numerical accuracy vs NumPy on representative cases (including ill-conditioned matrices).

### numpy.fft

- [ ] `n`, `s`, `axes`, `norm` parameters behave like NumPy.
- [ ] Real/complex output types and shapes match NumPy (`rfft*`, `irfft*`).
- [ ] `fftshift`/`ifftshift` behavior matches NumPy for odd/even sizes.

### numpy.random

- [ ] Generator API parity (`Generator`, `BitGenerator`, `default_rng`, `SeedSequence`).
- [ ] Reproducibility and seeding match NumPy reference behavior.
- [ ] Distribution parameter validation and dtype behavior match NumPy.

### numpy.ma

- [ ] Mask propagation rules match NumPy (binary ops, reductions, comparisons).
- [ ] `filled`, `fill_value`, `mask` semantics match NumPy.
- [ ] Masked reductions match NumPy defaults (axis, keepdims).

### numpy.char

- [ ] Unicode case behavior matches NumPy (`upper`, `lower`, `title`, `swapcase`).
- [ ] Encoding/decoding behavior matches NumPy output types.
- [ ] Vectorized split/join semantics match NumPy for arrays of strings.

### numpy.testing / numpy.typing

- [ ] Assertion helper parity (error types, error messages, tolerance behavior).
- [ ] `NDArray` and `NBitBase` definitions implemented and documented.

## Dataset Catalog (shared inputs for parity tests)

| Dataset ID  | Description             | Data Summary                       | Intended Coverage            |
| ----------- | ----------------------- | ---------------------------------- | ---------------------------- |
| DS-ARRAY-1  | Small integer vector    | [0, 1, -2, 3, 4, -5, 6]            | basic array ops, reductions  |
| DS-ARRAY-2  | 2D int matrix           | [[1, 2, 3], [4, 5, 6]]             | reshape/transpose/stack      |
| DS-ARRAY-3  | Float vector w/ NaN/Inf | [0.0, -1.5, 2.25, NaN, Inf, -Inf]  | nan-aware stats, comparisons |
| DS-ARRAY-4  | Boolean mask            | [true, false, true, false, true]   | masking, selection           |
| DS-ARRAY-5  | String vector           | "alpha", "Beta", "gamma", ""       | string/char ops              |
| DS-ARRAY-6  | Datetime sample         | 2024-01-01, 2024-06-30, 2025-01-01 | datetime/timedelta           |
| DS-ARRAY-7  | Complex vector          | [1+2i, -3+0.5i, 0-1i]              | complex math, fft            |
| DS-ARRAY-8  | Sorted vector           | [1, 1, 2, 3, 5, 8, 13]             | sorting/search/unique        |
| DS-ARRAY-9  | Set ops pair            | A:[1,2,3,4] B:[3,4,5,6]            | union/intersect/diff         |
| DS-ARRAY-10 | Linear system           | A:[[3,1],[1,2]] b:[9,8]            | linalg solve/inv             |
| DS-ARRAY-11 | FFT signal              | 64-sample sine wave, 1 Hz          | fft/rfft/freq                |
| DS-ARRAY-12 | Polynomial coeffs       | p(x)=1 -2x +3x^2                   | poly eval/roots              |
| DS-RAND-1   | Seeded RNG              | seed=1234, size=10                 | random distributions         |
| DS-IO-1     | Simple CSV              | 3x3 numeric matrix                 | load/save/loadtxt            |

## Parity Table

| Module | Function | Status | Dataset | Parity | Notes | Evidence |
| ------ | -------- | ------ | ------- | ------ | ----- | -------- |

## Update Workflow

1. Update `NUMPY_2_4_FUNCTIONS.md` if the NumPy reference list changes.
2. Refresh `PARITY_BASELINE_NUMPY_2_4.csv` against rust-numpy exports.
3. Regenerate `INVENTORY.md` and `ENUM_PARITY_NUMPY_2_4.csv`.
4. Update this report with new counts and dates.
