# Diagnostics: rust-numpy/src/advanced_reductions.rs

**Errors:** 2 | **Warnings:** 0

---

## Errors

### Line 478 (Col 13)
**ERROR[E0433]**: failed to resolve: use of undeclared type `Complex32`

```
error[E0433]: failed to resolve: use of undeclared type `Complex32`
   --> src/advanced_reductions.rs:478:13
    |
478 |             Complex32::new(1.0, 2.0),
    |             ^^^^^^^^^ use of undeclared type `Complex32`
    |
    = note: type alias `crate::complex_simd::tests::Complex32` exists but is inaccessible
```

### Line 479 (Col 13)
**ERROR[E0433]**: failed to resolve: use of undeclared type `Complex32`

```
error[E0433]: failed to resolve: use of undeclared type `Complex32`
   --> src/advanced_reductions.rs:479:13
    |
479 |             Complex32::new(3.0, 4.0),
    |             ^^^^^^^^^ use of undeclared type `Complex32`
    |
    = note: type alias `crate::complex_simd::tests::Complex32` exists but is inaccessible
```
