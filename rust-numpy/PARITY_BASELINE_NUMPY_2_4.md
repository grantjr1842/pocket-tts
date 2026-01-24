# NumPy 2.4 Parity Baseline (Structural)

Generated from `rust-numpy/NUMPY_2_4_FUNCTIONS.md` against current rust-numpy exports.

## Status Definitions
- `present`: symbol exported in the expected namespace.
- `namespace_mismatch`: symbol exists in module/codebase but is not exported at the expected namespace.
- `missing`: symbol not found in the module/codebase.

**Correctness-first note:** This is **structural** parity only. Every `present` item still requires correctness audits (signatures, broadcasting, dtype behavior, error parity).

## Module Summary
- **numpy**: 155/462 present, 93 namespace_mismatch, 214 missing (audit_priority: critical)
- **numpy.linalg**: 22/32 present, 0 namespace_mismatch, 10 missing (audit_priority: critical)
- **numpy.fft**: 18/18 present, 0 namespace_mismatch, 0 missing (audit_priority: high)
- **numpy.random**: 43/60 present, 0 namespace_mismatch, 17 missing (audit_priority: high)
- **numpy.polynomial**: 6/7 present, 0 namespace_mismatch, 1 missing (audit_priority: medium)
- **numpy.ma**: 13/219 present, 0 namespace_mismatch, 206 missing (audit_priority: high)
- **numpy.char**: 41/53 present, 0 namespace_mismatch, 12 missing (audit_priority: medium)
- **numpy.testing**: 4/36 present, 0 namespace_mismatch, 32 missing (audit_priority: medium)
- **numpy.typing**: 0/2 present, 0 namespace_mismatch, 2 missing (audit_priority: low)
- **numpy.ndarray Methods**: 4/53 present, 49 missing (audit_priority: critical)
- **numpy.ndarray Properties**: 6/17 present, 11 missing (audit_priority: critical)

## Outputs
- Full matrix: `rust-numpy/PARITY_BASELINE_NUMPY_2_4.csv`
- This summary: `rust-numpy/PARITY_BASELINE_NUMPY_2_4.md`
