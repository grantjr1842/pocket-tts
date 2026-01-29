# Diagnostics: rust-numpy/tests/random_tests.rs

**Errors:** 4 | **Warnings:** 0

---

## Errors

### Line 26 (Col 18)
**ERROR[E0283]**: type annotations needed

```
error[E0283]: type annotations needed
   --> tests/random_tests.rs:26:18
    |
 26 |     let result = geometric(1.5, &[10]);
    |                  ^^^^^^^^^ cannot infer type of the type parameter `T` declared on the function `geometric`
    |
    = note: cannot satisfy `_: Clone`
```

### Line 29 (Col 18)
**ERROR[E0283]**: type annotations needed

```
error[E0283]: type annotations needed
   --> tests/random_tests.rs:29:18
    |
 29 |     let result = geometric(-0.1, &[10]);
    |                  ^^^^^^^^^ cannot infer type of the type parameter `T` declared on the function `geometric`
    |
    = note: multiple `impl`s satisfying `_: std::convert::From<f64>` found in the following crates: `ciborium`, `core`, `qd`, `serde_json`:
            - impl std::convert::From<f64> for ciborium::Value;
            - impl std::convert::From<f64> for f128;
            - impl std::convert::From<f64> for qd::Quad;
            - impl std::convert::From<f64> for serde_json::Value;
```

### Line 200 (Col 5)
**ERROR[E0425]**: cannot find function `shuffle` in this scope

```
error[E0425]: cannot find function `shuffle` in this scope
   --> tests/random_tests.rs:200:5
    |
200 |     shuffle(&mut arr).unwrap();
    |     ^^^^^^^ not found in this scope
```

### Line 220 (Col 18)
**ERROR[E0425]**: cannot find function `shuffle` in this scope

```
error[E0425]: cannot find function `shuffle` in this scope
   --> tests/random_tests.rs:220:18
    |
220 |     let result = shuffle(&mut arr);
    |                  ^^^^^^^ not found in this scope
```
