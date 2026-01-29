# Diagnostics: rust-numpy/src/char_tests.rs

**Errors:** 1 | **Warnings:** 0

---

## Errors

### Line 108 (Col 22)
**ERROR[E0425]**: cannot find function `strip_chars` in this scope

```
error[E0425]: cannot find function `strip_chars` in this scope
   --> src/char_tests.rs:108:22
    |
108 |         let result = strip_chars(&a, "xyz").unwrap();
    |                      ^^^^^^^^^^^ not found in this scope
```
