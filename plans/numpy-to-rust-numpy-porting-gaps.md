# NumPy to Rust-NumPy Porting Gaps Analysis

**Generated:** 2026-01-26  
**Source:** rust-numpy/PARITY_BASELINE_NUMPY_2_4.csv  
**Target:** NumPy 2.4.0 API parity

## Executive Summary

Rust-NumPy has made significant progress but still has substantial gaps across multiple modules. The project has **676 total NumPy 2.4 callables** to port, with **284 present (42.0%)**, **93 namespace mismatches (13.8%)**, and **299 missing (44.2%)**.

### Module Parity Overview

| Module | Present | Namespace Mismatch | Missing | Total | Parity % | Priority |
|---------|----------|-------------------|---------|--------|-----------|----------|
| numpy.fft | 18 | 0 | 0 | 18/18 (100%) | ✅ COMPLETE |
| numpy.polynomial | 6 | 0 | 1 | 6/7 (85.7%) | Medium |
| numpy.char | 41 | 0 | 12 | 41/53 (77.4%) | High |
| numpy.random | 43 | 0 | 17 | 43/60 (71.7%) | High |
| numpy.linalg | 22 | 0 | 10 | 22/32 (68.8%) | High |
| numpy | 155 | 93 | 214 | 155/462 (33.5%) | Critical |
| numpy.ndarray Properties | 6 | 0 | 11 | 6/17 (35.3%) | Critical |
| numpy.ndarray Methods | 4 | 0 | 49 | 4/53 (7.5%) | Critical |
| numpy.testing | 4 | 0 | 32 | 4/36 (11.1%) | Medium |
| numpy.ma | 13 | 0 | 206 | 13/219 (5.9%) | Critical |
| numpy.typing | 0 | 0 | 2 | 0/2 (0%) | Low |

---

## Critical Gaps

### 1. numpy.ma (Masked Arrays) - 206 Missing Functions

**Status:** Only 13/219 functions present (5.9%)  
**Priority:** CRITICAL - Masked arrays are fundamental to NumPy's data handling

#### Missing Core Functions:
- Array creation: `array`, `asanyarray`, `asarray`, `empty`, `empty_like`, `ones`, `ones_like`, `zeros`, `zeros_like`
- Mathematical operations: All ufuncs (`abs`, `add`, `subtract`, `multiply`, `divide`, `sin`, `cos`, `exp`, `log`, etc.)
- Reduction operations: `max`, `min`, `mean`, `std`, `var`, `sum`, `prod`, `cumsum`, `cumprod`
- Mask operations: `compress`, `compress_cols`, `compress_nd`, `compress_rowcols`, `compress_rows`, `compressed`
- Masked value functions: `masked_all`, `masked_all_like`, `masked_equal`, `masked_greater`, `masked_less`, `masked_inside`, `masked_outside`, `masked_where`, etc.
- Utility functions: `getdata`, `getmask`, `getmaskarray`, `is_mask`, `is_masked`, `make_mask`, etc.

#### Present Functions:
- `compress`, `filled`, `mean`, `median`, `ndim`, `set_fill_value`, `shape`, `size`, `std`, `sum`, `unique`, `var`

---

### 2. numpy.ndarray Methods - 49 Missing Methods

**Status:** Only 4/53 methods present (7.5%)  
**Priority:** CRITICAL - ndarray methods are core to array manipulation

#### Present Methods:
- `dot`, `reshape`, `take`, `transpose`

#### Missing Methods:
**Mathematical/Reduction Methods:**
- `all`, `any`, `argmax`, `argmin`, `max`, `min`, `mean`, `sum`, `prod`, `std`, `var`

**Shape Manipulation:**
- `flatten`, `ravel`, `reshape`, `resize`, `squeeze`, `swapaxes`, `transpose`

**Array Operations:**
- `choose`, `clip`, `compress`, `conj`, `conjugate`, `copy`, `cumprod`, `cumsum`, `diagonal`, `fill`, `partition`, `put`, `repeat`, `round`, `searchsorted`, `sort`

**Data Access:**
- `byteswap`, `dump`, `dumps`, `getfield`, `item`, `setfield`, `setflags`, `tobytes`, `tofile`, `tolist`, `to_device`, `view`

---

### 3. numpy.ndarray Properties - 11 Missing Properties

**Status:** Only 6/17 properties present (35.3%)  
**Priority:** CRITICAL

#### Present Properties:
- `data`, `dtype`, `ndim`, `shape`, `size`, `strides`

#### Missing Properties:
- `T`, `base`, `ctypes`, `device`, `flags`, `flat`, `imag`, `itemsize`, `mT`, `nbytes`, `real`

---

### 4. numpy (Top-level) - 214 Missing Functions

**Status:** Only 155/462 functions present (33.5%)  
**Priority:** CRITICAL

#### Missing Functions by Category:

**Array Creation & Conversion (40+ missing):**
- `asanyarray`, `asarray`, `asarray_chkfinite`, `ascontiguousarray`, `asfortranarray`, `asmatrix`
- `array2string`, `array_repr`, `array_str`
- `from_dlpack`, `frompyfunc`, `fromregex`

**Type System (30+ missing):**
- All dtype constructors: `bool`, `bool_`, `byte`, `bytes_`, `complex128`, `complex256`, `complex64`, `complexfloating`, `cdouble`, `clongdouble`, `csingle`, `double`, `float16`, `float32`, `float64`, `float128`, `floating`, `generic`, `half`, `int16`, `int32`, `int64`, `int8`, `int_`, `intc`, `integer`, `intp`, `long`, `longdouble`, `longlong`, `number`, `object_`, `signedinteger`, `single`, `str_`, `uint`, `uint16`, `uint32`, `uint64`, `uint8`, `uintc`, `uintp`, `ulong`, `ulonglong`, `unsignedinteger`, `ushort`, `void`

**Mathematical Functions (namespace mismatch - need exports):**
- `acos`, `acosh`, `asin`, `asinh`, `atan`, `atan2`, `atanh`, `cos`, `cosh`, `exp`, `exp2`, `expm1`, `floor`, `hypot`, `i0`, `log`, `log10`, `log1p`, `log2`, `logaddexp`, `logaddexp2`, `sin`, `sinc`, `sinh`, `tan`, `tanh`, `unwrap`
- Ufuncs: `add`, `all`, `any`, `around`, `ceil`, `divide`, `equal`, `exp`, `fix`, `floor`, `greater`, `greater_equal`, `isfinite`, `isinf`, `isnan`, `isneginf`, `isposinf`, `less`, `less_equal`, `log`, `max`, `maximum`, `min`, `minimum`, `multiply`, `negative`, `not_equal`, `ones`, `prod`, `radians`, `result_type`, `rint`, `round`, `sign`, `sin`, `sqrt`, `square`, `subtract`, `take`, `tan`, `tanh`, `trunc`, `zeros`

**Array Manipulation (20+ missing):**
- `broadcast`, `broadcast_shapes`, `concat`, `copyto`, `indices`, `may_share_memory`, `ndenumerate`, `ndindex`, `nditer`, `nested_iters`, `require`, `resize`, `shares_memory`, `unstack`, `where`

**Bitwise Operations (4 missing):**
- `bitwise_count`, `bitwise_invert`, `bitwise_left_shift`, `bitwise_right_shift`

**Indexing & Slicing (10+ missing):**
- `diag_indices`, `diag_indices_from`, `diagflat`, `indices`, `ix_`, `mask_indices`, `ravel_multi_index`, `take_along_axis`, `tril_indices`, `tril_indices_from`, `triu_indices`, `triu_indices_from`

**Matrix Operations (5+ missing):**
- `bmat`, `matvec`, `matrix`, `matrix_transpose`, `tensordot`

**Special Functions (10+ missing):**
- `cbrt`, `deg2rad`, `divmod`, `float_power`, `format_float_positional`, `format_float_scientific`, `frexp`, `ldexp`, `mod`, `modf`, `nan_to_num`, `nanargmax`, `nanargmin`, `nextafter`, `piecewise`, `positive`, `pow`, `power`, `rad2deg`, `reciprocal`, `remainder`, `spacing`, `sort_complex`, `tri`

**String/Char (4 missing):**
- `base_repr`, `binary_repr`, `typename`

**IO & Serialization (10+ missing):**
- `genfromtxt`, `load`, `loadtxt`, `save`, `savetxt`, `savez`, `savez_compressed`

**Error Handling (3 missing):**
- `errstate`, `geterr`, `geterrcall`, `seterr`, `seterrcall`

**Utility Functions (15+ missing):**
- `busdaycalendar`, `common_type`, `finfo`, `get_include`, `get_printoptions`, `getbufsize`, `iinfo`, `info`, `iscomplex`, `iscomplexobj`, `isdtype`, `isfortran`, `isnat`, `isreal`, `isrealobj`, `isscalar`, `issubdtype`, `iterable`, `min_scalar_type`, `mintypecode`, `printoptions`, `set_printoptions`, `setbufsize`, `show_config`, `show_runtime`, `test`

**Unique Functions (4 missing):**
- `unique_all`, `unique_counts`, `unique_inverse`, `unique_values`

**Polynomial Functions (8+ missing):**
- `poly`, `poly1d`, `polyadd`, `polyder`, `polydiv`, `polyfit`, `polyint`, `polymul`, `polysub`, `polyval`, `roots`

**Window Functions (namespace mismatch - need exports):**
- `bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`

---

## High Priority Gaps

### 5. numpy.random - 17 Missing Functions

**Status:** 43/60 present (71.7%)  
**Priority:** HIGH - Random number generation is widely used

#### Missing Functions:

**Core Classes (6 missing):**
- `BitGenerator`, `Generator`, `MT19937`, `PCG64`, `PCG64DXSM`, `Philox`, `SFC64`

**Distributions (8 missing):**
- `bytes`, `f`, `logistic`, `multivariate_normal`, `noncentral_chisquare`, `noncentral_f`, `power`, `standard_t`, `vonmises`

**Utility Functions (3 missing):**
- `default_rng`

---

### 6. numpy.linalg - 10 Missing Functions

**Status:** 22/32 present (68.8%)  
**Priority:** HIGH - Linear algebra is fundamental to scientific computing

#### Missing Functions:

**Matrix Operations (6 missing):**
- `diagonal`, `matrix_norm`, `matrix_transpose`, `svdvals`, `tensorinv`, `tensorsolve`

**Utility Functions (4 missing):**
- `LinAlgError`, `tensordot`, `vecdot`, `vector_norm`

---

### 7. numpy.char - 12 Missing Functions

**Status:** 41/53 present (77.4%)  
**Priority:** HIGH - String operations are commonly used

#### Missing Functions:

**Core Functions (10 missing):**
- `array`, `asarray`, `chararray`, `compare_chararrays`, `decode`, `encode`, `isdecimal`, `islower`, `istitle`, `isupper`, `translate`

**Utility Functions (2 missing):**
- `mod`

---

## Medium Priority Gaps

### 8. numpy.testing - 32 Missing Functions

**Status:** 4/36 present (11.1%)  
**Priority:** MEDIUM - Testing framework is important but not core functionality

#### Missing Functions:

**Exception Classes (3 missing):**
- `IgnoreException`, `KnownFailureException`, `SkipTest`, `TestCase`

**Assertion Functions (24 missing):**
- `assert_`, `assert_almost_equal`, `assert_approx_equal`, `assert_array_almost_equal_nulp`, `assert_array_compare`, `assert_array_max_ulp`, `assert_equal`, `assert_no_gc_cycles`, `assert_no_warnings`, `assert_raises`, `assert_raises_regex`, `assert_string_equal`, `assert_warns`

**Utility Functions (5 missing):**
- `break_cycles`, `build_err_msg`, `check_support_sve`, `clear_and_catch_warnings`, `decorate_methods`, `jiffies`, `measure`, `memusage`, `print_assert_equal`, `run_threaded`, `rundocs`, `runstring`, `suppress_warnings`, `tempdir`, `temppath`

---

### 9. numpy.typing - 2 Missing Types

**Status:** 0/2 present (0%)  
**Priority:** LOW - Type hints are important for IDE support but not runtime functionality

#### Missing Types:
- `NBitBase`
- `NDArray`

---

### 10. numpy.polynomial - 1 Missing Function

**Status:** 6/7 present (85.7%)  
**Priority:** MEDIUM

#### Missing Function:
- `set_default_printstyle`

---

## Namespace Mismatch Issues (93 Functions)

**Status:** 93 functions exist internally but are not exported in the expected namespace  
**Priority:** HIGH - This is a quick win that can improve parity without new implementations

### Affected Modules:
- **numpy:** 93 functions have namespace mismatches (mostly ufuncs and utility functions that need to be re-exported)

### Examples:
- `acos` - exists in `math_ufuncs` but not exported as `numpy.acos`
- `add` - exists in `ufunc_ops` but not exported as `numpy.add`
- `all` - exists in `ufunc_ops` but not exported as `numpy.all`
- `arange` - exists in `array_creation` but not exported as `numpy.arange`
- `zeros` - exists in `array` but not exported as `numpy.zeros`

### Action Required:
Review [`lib.rs`](rust-numpy/src/lib.rs) and add appropriate re-exports to match NumPy's public API.

---

## Implementation Recommendations

### Phase 1: Quick Wins (Namespace Exports)
1. **Fix namespace mismatches** - Export 93 functions that already exist internally
2. **Export ndarray properties** - Add `T`, `base`, `flags`, `flat`, `imag`, `itemsize`, `mT`, `nbytes`, `real`
3. **Export ndarray methods** - Add 49 methods that are already implemented internally

### Phase 2: Critical Core Functionality
1. **Implement numpy.ndarray methods** - 49 missing methods are critical for array manipulation
2. **Implement numpy.ma (Masked Arrays)** - Start with core functions: array creation, basic ufuncs, reductions
3. **Add dtype constructors** - Implement all missing dtype types (bool_, int8, float64, etc.)

### Phase 3: High-Value Modules
1. **Complete numpy.random** - Add missing BitGenerator, Generator, and distributions
2. **Complete numpy.linalg** - Add missing matrix operations and utility functions
3. **Complete numpy.char** - Add missing string operations

### Phase 4: Supporting Infrastructure
1. **Implement numpy.testing** - Add testing framework functions
2. **Implement numpy.typing** - Add type hints for IDE support
3. **Add utility functions** - IO, error handling, and miscellaneous utilities

### Phase 5: Advanced Features
1. **Complete numpy.ma** - Finish all 206 missing masked array functions
2. **Add special functions** - Implement advanced mathematical functions
3. **Add polynomial functions** - Complete polynomial module

---

## Gap Summary by Category

| Category | Missing Count | Priority | Estimated Complexity |
|-----------|----------------|----------|---------------------|
| Masked Arrays (numpy.ma) | 206 | Critical | High |
| ndarray Methods | 49 | Critical | Medium |
| Top-level numpy functions | 214 | Critical | Mixed |
| ndarray Properties | 11 | Critical | Low |
| Random (numpy.random) | 17 | High | Medium |
| Linear Algebra (numpy.linalg) | 10 | High | Medium |
| String Operations (numpy.char) | 12 | High | Low |
| Testing (numpy.testing) | 32 | Medium | Low |
| Typing (numpy.typing) | 2 | Low | Low |
| Polynomial (numpy.polynomial) | 1 | Medium | Low |
| **TOTAL** | **299** | - | - |

---

## Next Steps

1. **Prioritize namespace exports** - This is the fastest way to improve parity
2. **Focus on ndarray methods** - These are core to array manipulation
3. **Implement dtype system** - Required for many other features
4. **Build masked arrays incrementally** - Start with core functionality
5. **Complete high-value modules** - Random, linalg, char
6. **Add testing infrastructure** - Important for long-term maintainability

---

## Notes

- The project has extensive internal implementation (see [`INVENTORY.md`](rust-numpy/INVENTORY.md))
- Many functions exist but are not properly exported (namespace mismatch)
- The main challenge is export organization and API surface alignment, not implementation
- Behavioral parity (correctness) is separate from structural parity (API presence)
- See [`PARITY_NUMPY_2_4.md`](rust-numpy/PARITY_NUMPY_2_4.md) for correctness validation checklist
