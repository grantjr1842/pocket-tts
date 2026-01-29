# [P1] Reorganize module structure to match NumPy hierarchy

**Issue #603**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/603  
**Created:** 2026-01-29T03:36:09Z  
**Updated:** 2026-01-29T03:36:09Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:submodules  

---

## Summary
Comprehensive audit comparing rust-numpy module structure to NumPy's structure reveals multiple structural misalignments.

## Structural Mismatches

### 1. Misplaced Modules (Wrong Directory)

| Module | Current | NumPy Location | Action |
|--------|---------|----------------|--------|
| `ma` | `src/modules/ma.rs` | `numpy.ma` | Move to `src/ma/mod.rs` |
| `testing` | `src/modules/testing.rs` | `numpy.testing` | Move to `src/testing/mod.rs` |
| `fft` | `src/fft.rs` | `numpy.fft` | Convert to `src/fft/mod.rs` |

### 2. Missing Core Submodules

NumPy has these container modules that rust-numpy lacks:

```
numpy/                    rust-numpy/src/ (current)
├── core/                 ❌ Missing
│   ├── defchararray      → Partial in src/char.rs
│   ├── einsumfunc        → Partial in src/linalg/einsum.rs
│   ├── fromnumeric       → Missing
│   ├── function_base     → Missing
│   ├── getlimits         → Partial in src/utils.rs
│   ├── numeric.py        → Partial in src/array_creation.rs
│   ├── numerictypes.py   → Partial in src/dtype.rs
│   ├── records.py        → Partial in src/rec.rs
│   └── shape_base.py     → Partial in src/array_manipulation.rs
├── lib/                  ❌ Missing
│   ├── arraypad.py       → Partial
│   ├── arraysetops.py    → Partial in src/set_ops.rs
│   ├── histograms.py     → Partial in src/statistics.rs
│   ├── index_tricks.py   → Missing
│   └── scimath.py        → Missing (emath equivalent)
├── compat/               ❌ Missing
├── exceptions/           ❌ Missing (NumPy 2.0+)
├── dtypes/               ❌ Missing (NumPy 2.0+)
└── strings/              ❌ Missing (NumPy 2.0+)
```

### 3. Extra Internal Modules (Not in NumPy)

These modules should be marked `#[doc(hidden)]`:
- `advanced_broadcast`, `advanced_reductions`
- `complex_simd`, `simd`, `simd_ops`
- `cpu_features`
- `dynamic_kernel_registry`, `kernel_api`, `kernel_impls`, `kernel_registry`
- `layout_optimizer`
- `parallel`, `parallel_broadcasting`, `parallel_executor`, `parallel_ops`
- `performance_metrics`, `profiler`
- `strided_executor`

## Recommended Module Reorganization

```
rust-numpy/src/
├── lib.rs                    # Minimize - just re-exports
│
├── core/                     # NEW - Core functionality
│   ├── mod.rs
│   ├── defchararray.rs       # Move from char.rs
│   ├── fromnumeric.rs        # NEW
│   ├── function_base.rs      # NEW
│   ├── numeric.rs            # Merge from array_creation
│   └── shape_base.rs         # Merge from array_manipulation
│
├── lib/                      # NEW - Library utilities
│   ├── mod.rs
│   ├── arraysetops.rs        # Move from set_ops.rs
│   ├── scimath.rs            # NEW (emath alternative)
│   └── index_tricks.py       # NEW
│
├── ma/                       # Move from modules/ma.rs
│   ├── mod.rs
│   └── ...
│
├── testing/                  # Move from modules/testing.rs
│   ├── mod.rs
│   └── ...
│
├── fft/                      # Convert from fft.rs
│   ├── mod.rs
│   └── ...
│
├── exceptions/               # NEW - NumPy 2.0+
│   └── mod.rs
│
├── dtypes/                   # NEW - NumPy 2.0+
│   └── mod.rs
│
├── strings/                  # NEW - NumPy 2.0+
│   └── mod.rs
│
└── internal/                 # NEW - Hide implementation
    ├── mod.rs
    ├── advanced_broadcast.rs
    ├── simd/
    ├── parallel/
    └── kernels/
```

## Related Issues
- #570 - Submodule structure audit
- #583 - Create numpy.char submodule
- #584 - Create emath/exceptions/ctypeslib
- #601 - Hide internal modules

## Acceptance Criteria
- [ ] Modules in correct locations
- [ ] Internal modules hidden
- [ ] NumPy module hierarchy matched
- [ ] All existing tests pass

