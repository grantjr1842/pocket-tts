# Diagnostics: rust-numpy/tests/ndarray_conversion_tests.rs

**Errors:** 0 | **Warnings:** 1

---

## Warnings

### Line 40 (Col 9)
**WARNING[non_snake_case]**: warning: variable `cArray` should have a snake case name

```
warning: variable `cArray` should have a snake case name
  --> tests/ndarray_conversion_tests.rs:40:9
   |
40 |     let cArray = Array::from_vec(vec![c1]);
   |         ^^^^^^ help: convert the identifier to snake case: `c_array`
   |
   = note: `#[warn(non_snake_case)]` (part of `#[warn(nonstandard_style)]`) on by default
```
