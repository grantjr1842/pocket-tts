# Diagnostics: rust-numpy/tests/kernel_registry_tests.rs

**Errors:** 1 | **Warnings:** 0

---

## Errors

### Line 307 (Col 1)
**ERROR[Click for full compiler diagnostic]**: error: unexpected closing delimiter: `}`

```
error: unexpected closing delimiter: `}`
   --> tests/kernel_registry_tests.rs:307:1
    |
268 |     fn test_multiple_kernel_types_performance() {
    |                                                 - the nearest open delimiter
...
273 |         let b_f64: Vec<f64> = (0..size).map(|i| i as f64).collect());
    |                                                                    - missing open `(` for this delimiter
...
307 | }
    | ^ unexpected closing delimiter
```
