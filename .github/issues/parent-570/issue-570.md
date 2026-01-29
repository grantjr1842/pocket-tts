# rust-numpy: Submodule structure doesn't match NumPy hierarchy

**Issue #570**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/570  
**Created:** 2026-01-29T02:47:02Z  
**Updated:** 2026-01-29T03:23:51Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:submodules, ralph/numpy:audit  

---

## Summary
NumPy's module hierarchy must be precisely mirrored in rust-numpy. Currently, submodules are either missing, incorrectly structured, or don't match NumPy's organization.

## NumPy Module Structure Reference

```
numpy/
├── __init__.py          # Top-level exports (lib.rs equivalent)
├── array_api/           # Array API standard compatibility (NEW in NumPy 2.0)
├── core/                # Core array functionality
│   ├── __init__.py
│   ├── _multiarray_umath
│   ├── arrayprint
│   ├── defchararray     # → numpy.char
│   ├── einsumfunc       # → numpy.linalg
│   ├── fromnumeric      # Array methods as functions
│   ├── function_base    # Core functions
│   ├── getlimits        # finfo, iinfo
│   ├── multiarray
│   ├── numeric          # Array creation, fromnumeric
│   ├── numerictypes     # Type objects
│   ├── records          # → numpy.rec
│   ├── shape_base       # Array shape manipulation
│   └── umath            # Universal functions
├── char/                # Character array operations (numpy.char)
├── ctypeslib/           # C-interoperability
├── distutils/           # Build support (DEPRECATED)
├── exceptions/          # NumPy-specific exceptions (NEW in NumPy 2.0)
├── f2py/                # Fortran to Python
├── fft/                 # Fast Fourier Transform
├── lib/                 # Additional utilities
│   ├── __init__.py
│   ├── arraypad         # → numpy.pad
│   ├── arraysetops      # → numpy.set_ops
│   ├── arrayterator     # Array iterator
│   ├── format           # Array file format
│   ├── function_base    # Misc functions
│   ├── histograms       # → numpy.histogram
│   ├── index_tricks     # → numpy.r_, numpy.c_, numpy.ix_
│   ├── mixins           # NDArrayOperatorsMixin
│   ├── nanfunctions     # nan-aware functions
│   ├── npyio            # → numpy.io
│   ├── polynomial       # → numpy.polynomial
│   ├── scimath          # → numpy.emath
│   ├── shape_base       # vstack, hstack, etc.
│   ├── stride_tricks    # as_strided, sliding_window
│   ├── twodim_base      # 2D array creation
│   ├── type_check       # type checking functions
│   ├── ufunclike        # ufunc-like functions
│   └── utils            # utility functions
├── linalg/              # Linear algebra
├── ma/                  # Masked arrays
├── matlib/              # Matrix library (deprecated)
├── matrixlib/           # Matrix-specific (deprecated)
├── polynomial/          # Polynomial package
├── random/              # Random number generation
├── rec/                 # Record arrays
├── strings/             # String operations (NEW in NumPy 2.0)
├── testing/             # Test utilities
└── typing/              # Type annotations
```

## Current rust-numpy Structure vs Expected

### ✅ Correctly Implemented
```
rust-numpy/src/
├── lib.rs              # Top-level (needs fixes, see other issue)
├── fft/                # FFT operations ✓
├── linalg/             # Linear algebra ✓ (structure matches)
├── polynomial/         # Polynomial operations ✓
├── random/             # Random numbers ✓ (structure matches)
├── modules/
│   ├── ma.rs           # Masked arrays ✓ (structure matches)
│   └── testing.rs      # Testing utilities ✓
└── typing/             # Type annotations ✓
```

### ❌ MISSING Submodules
1. **`numpy.char`** - Character string operations
   - NumPy: `numpy.char.add`, `numpy.char.multiply`, `numpy.char.capitalize`, etc.
   - Current: Partial implementation in `src/char.rs` but NOT as a proper submodule
   - **Action**: Create `src/char/mod.rs` with proper structure

2. **`numpy.emath`** - Mathematical functions with automatic domain
   - NumPy: `numpy.emath.sqrt`, `numpy.emath.log`, `numpy.emath.arcsin`, etc.
   - Current: ❌ NOT IMPLEMENTED
   - **Action**: Create `src/emath.rs` or `src/emath/mod.rs`

3. **`numpy.exceptions`** - NumPy-specific exceptions (NumPy 2.0+)
   - NumPy: `ComplexWarning`, `VisibleDeprecationWarning`, `RankWarning`, etc.
   - Current: Generic error handling in `src/error.rs`
   - **Action**: Create `src/exceptions.rs` with specific exception types

4. **`numpy.lib`** - Library utilities
   - NumPy: `numpy.lib.stride_tricks`, `numpy.lib.scimath`, etc.
   - Current: ❌ NOT IMPLEMENTED
   - **Action**: Create `src/lib_module/` directory (named to avoid conflict with lib.rs)

5. **`numpy.strings`** - String operations (NumPy 2.0+)
   - NumPy: `numpy.strings.add`, `numpy.strings.multiply`, etc.
   - Current: ❌ NOT IMPLEMENTED
   - **Action**: Create `src/strings/mod.rs`

6. **`numpy.ctypeslib`** - C-interoperability
   - NumPy: `numpy.ctypeslib.as_array`, `numpy.ctypeslib.as_ctypes`, etc.
   - Current: ❌ NOT IMPLEMENTED
   - **Action**: Create `src/ctypeslib.rs`

7. **`numpy.dtypes`** - Data type utilities (NumPy 2.0+)
   - NumPy: `numpy.dtypes.StringDType`, etc.
   - Current: ❌ NOT IMPLEMENTED
   - **Action**: Create `src/dtypes.rs` or extend `src/dtype.rs`

### ❌ INCORRECTLY Structured Submodules

1. **`src/set_ops.rs`** → Should be `src/lib/arraysetops.rs` or `src/set_ops/mod.rs`
   - Currently: Flat file
   - Expected: Module with proper sub-organization
   - **Action**: Reorganize to match NumPy's `numpy.lib.arraysetops`

2. **`src/char.rs`** → Should be `src/char/mod.rs`
   - Currently: Single file with exports
   - Expected: Module directory with sub-modules
   - **Action**: Convert to proper module structure

3. **`src/utils.rs`** - Mixes multiple NumPy modules
   - Currently: Contains items from `numpy.core`, `numpy.lib`, `numpy.testing`
   - Expected: Split into proper modules
   - **Action**: 
     - Move type-related to `src/dtype.rs` or `src/dtypes.rs`
     - Move IO-related to `src/io.rs` (already exists)
     - Move test-related to `src/modules/testing.rs`
     - Keep only true `numpy.lib.utils` functions

4. **`src/io.rs`** - Missing many NumPy I/O functions
   - NumPy: `numpy.save`, `numpy.load`, `numpy.savez`, `numpy.savez_compressed`, `numpy.genfromtxt`
   - Current: Only `from_txt`, `to_txt`, `from_npy`, `to_npy`
   - **Action**: Add missing functions

5. **`src/modules/`** - Directory structure doesn't match
   - Currently: `ma.rs`, `testing.rs` directly in folder
   - Expected: Each as a proper module directory
   - **Action**: Create `src/ma/mod.rs`, keep `src/testing/mod.rs`

## Required Module Reorganization

### Proposed Structure
```
rust-numpy/src/
├── lib.rs                      # Crate root
├── lib/
│   └── ...                     # Internal utilities (not public modules)
├── core/                       # numpy.core equivalent (if needed)
│   └── ...
├── char/                       # numpy.char
│   ├── mod.rs
│   └── ...
├── ctypeslib.rs                # numpy.ctypeslib
├── dtypes.rs                   # numpy.dtypes (NumPy 2.0+)
├── emath.rs                    # numpy.emath
├── exceptions.rs               # numpy.exceptions (NumPy 2.0+)
├── fft/
│   └── mod.rs
├── io.rs                       # numpy.io / numpy.lib.npyio
├── linalg/
│   └── mod.rs
├── ma/                         # numpy.ma (was modules/ma.rs)
│   ├── mod.rs
│   ├── core.rs
│   └── extras.rs
├── polynomial/
│   └── mod.rs
├── random/
│   └── mod.rs
├── rec.rs                      # numpy.rec
├── set_ops.rs                  # numpy.lib.arraysetops (OR lib/arraysetops.rs)
├── strings/                    # numpy.strings (NumPy 2.0+)
│   └── mod.rs
├── testing/                    # numpy.testing (was modules/testing.rs)
│   └── mod.rs
└── typing/
    └── mod.rs
```

## Acceptance Criteria
- [ ] All NumPy submodules represented in rust-numpy
- [ ] Each submodule has proper `mod.rs` with `pub use` exports
- [ ] Sub-module hierarchy matches NumPy exactly
- [ ] No mixing of different NumPy modules in single files
- [ ] `pub mod` declarations in `lib.rs` match NumPy's module structure
- [ ] Internal modules marked with `#[doc(hidden)]`

## Affected Files
- `rust-numpy/src/lib.rs` - Module declarations
- `rust-numpy/src/modules/` - Reorganize to `ma/`, `testing/`
- `rust-numpy/src/char.rs` - Convert to `char/mod.rs`
- `rust-numpy/src/utils.rs` - Split into proper modules
- New files: `exceptions.rs`, `emath.rs`, `ctypeslib.rs`, `dtypes.rs`, `strings/`


---

## Sub-Issues (for parallel development)

| Issue | Title | Scope |
|-------|-------|-------|
| #583 | Create numpy.char Submodule Structure | Convert char.rs to char/ module |
| #584 | Create emath, exceptions, ctypeslib Submodules | New submodules |

**Coordination Notes:**
- #583 creates module structure for char operations
- #584 creates three new submodules from scratch
- Minimal overlap between these issues

