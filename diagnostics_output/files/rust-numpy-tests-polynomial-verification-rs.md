# Diagnostics: rust-numpy/tests/polynomial_verification.rs

**Errors:** 0 | **Warnings:** 3

---

## Warnings

### Line 20 (Col 5)
**WARNING[unused_imports]**: warning: unused import: `num_complex::Complex`

```
warning: unused import: `num_complex::Complex`
  --> tests/polynomial_verification.rs:20:5
   |
20 | use num_complex::Complex;
   |     ^^^^^^^^^^^^^^^^^^^^
   |
   = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default
```

### Line 27 (Col 7)
**WARNING[dead_code]**: warning: constant `ILL_CONDITIONED_TOLERANCE` is never used

```
warning: constant `ILL_CONDITIONED_TOLERANCE` is never used
  --> tests/polynomial_verification.rs:27:7
   |
27 | const ILL_CONDITIONED_TOLERANCE: f64 = 1e-6;
   |       ^^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: `#[warn(dead_code)]` (part of `#[warn(unused)]`) on by default
```

### Line 56 (Col 9)
**WARNING[unused_variables]**: warning: unused variable: `coeffs`

```
warning: unused variable: `coeffs`
  --> tests/polynomial_verification.rs:56:9
   |
56 |     let coeffs = p.coeffs();
   |         ^^^^^^ help: if this is intentional, prefix it with an underscore: `_coeffs`
   |
   = note: `#[warn(unused_variables)]` (part of `#[warn(unused)]`) on by default
```
