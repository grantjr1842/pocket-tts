# Diagnostics: rust-numpy/examples/typing_examples.rs

**Errors:** 0 | **Warnings:** 1

---

## Warnings

### Line 25 (Col 9)
**WARNING[unused_mut]**: warning: variable does not need to be mutable

```
warning: variable does not need to be mutable
  --> examples/typing_examples.rs:25:9
   |
25 |     let mut result = floats.clone();
   |         ----^^^^^^
   |         |
   |         help: remove this `mut`
   |
   = note: `#[warn(unused_mut)]` (part of `#[warn(unused)]`) on by default
```
