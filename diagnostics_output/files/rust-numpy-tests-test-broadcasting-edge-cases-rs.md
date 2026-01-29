# Diagnostics: rust-numpy/tests/test_broadcasting_edge_cases.rs

**Errors:** 0 | **Warnings:** 2

---

## Warnings

### Line 9 (Col 9)
**WARNING[unused_variables]**: warning: unused variable: `matrix`

```
warning: unused variable: `matrix`
  |
9 |     let matrix = Array::from_vec(vec![1i32, 2, 3, 4])
  |         ^^^^^^ help: if this is intentional, prefix it with an underscore: `_matrix`
  |
  = note: `#[warn(unused_variables)]` (part of `#[warn(unused)]`) on by default
```

### Line 74 (Col 9)
**WARNING[unused_variables]**: warning: unused variable: `a`

```
warning: unused variable: `a`
  --> tests/test_broadcasting_edge_cases.rs:74:9
   |
74 |     let a = Array::from_vec(vec![1i32, 2, 3, 4])
   |         ^ help: if this is intentional, prefix it with an underscore: `_a`
```
