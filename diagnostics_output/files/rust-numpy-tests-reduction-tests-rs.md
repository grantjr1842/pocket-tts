# Diagnostics: rust-numpy/tests/reduction_tests.rs

**Errors:** 6 | **Warnings:** 0

---

## Errors

### Line 84 (Col 47)
**ERROR[E0689]**: can't call method `abs` on ambiguous numeric type `{float}`

```
error[E0689]: can't call method `abs` on ambiguous numeric type `{float}`
  --> tests/reduction_tests.rs:84:47
   |
84 |         assert!((*mean.get(0).unwrap() - 3.0).abs() < 1e-10);
   |                                               ^^^
```

### Line 92 (Col 53)
**ERROR[E0689]**: can't call method `abs` on ambiguous numeric type `{float}`

```
error[E0689]: can't call method `abs` on ambiguous numeric type `{float}`
  --> tests/reduction_tests.rs:92:53
   |
92 |         assert!((*mean_axis0.get(0).unwrap() - 2.5).abs() < 1e-10);
   |                                                     ^^^
```

### Line 93 (Col 53)
**ERROR[E0689]**: can't call method `abs` on ambiguous numeric type `{float}`

```
error[E0689]: can't call method `abs` on ambiguous numeric type `{float}`
  --> tests/reduction_tests.rs:93:53
   |
93 |         assert!((*mean_axis0.get(1).unwrap() - 3.5).abs() < 1e-10);
   |                                                     ^^^
```

### Line 94 (Col 53)
**ERROR[E0689]**: can't call method `abs` on ambiguous numeric type `{float}`

```
error[E0689]: can't call method `abs` on ambiguous numeric type `{float}`
  --> tests/reduction_tests.rs:94:53
   |
94 |         assert!((*mean_axis0.get(2).unwrap() - 4.5).abs() < 1e-10);
   |                                                     ^^^
```

### Line 102 (Col 51)
**ERROR[E0689]**: can't call method `abs` on ambiguous numeric type `{float}`

```
error[E0689]: can't call method `abs` on ambiguous numeric type `{float}`
   --> tests/reduction_tests.rs:102:51
    |
102 |         assert!((*var.get(0).unwrap() - expected).abs() < 1e-10);
    |                                                   ^^^
```

### Line 265 (Col 47)
**ERROR[E0689]**: can't call method `abs` on ambiguous numeric type `{float}`

```
error[E0689]: can't call method `abs` on ambiguous numeric type `{float}`
   --> tests/reduction_tests.rs:265:47
    |
265 |         assert!((*mean.get(0).unwrap() - 2.5).abs() < 1e-10);
    |                                               ^^^
```
