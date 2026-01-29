# Diagnostics: rust-numpy/tests/type_promotion_rules_tests.rs

**Errors:** 0 | **Warnings:** 1

---

## Warnings

### Line 78 (Col 9)
**WARNING[unused_variables]**: warning: unused variable: `f64_type`

```
warning: unused variable: `f64_type`
  --> tests/type_promotion_rules_tests.rs:78:9
   |
78 |     let f64_type = Dtype::Float64 { byteorder: None };
   |         ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_f64_type`
   |
   = note: `#[warn(unused_variables)]` (part of `#[warn(unused)]`) on by default
```
