# Diagnostics: rust-numpy/tests/comprehensive_tests.rs

**Errors:** 1 | **Warnings:** 0

---

## Errors

### Line 129 (Col 30)
**ERROR[E0277]**: the trait bound `{integer}: std::convert::From<f64>` is not satisfied

```
error[E0277]: the trait bound `{integer}: std::convert::From<f64>` is not satisfied
    --> tests/comprehensive_tests.rs:129:30
     |
 129 |         let _mean_result = a.mean(None, false);
     |                              ^^^^ the trait `std::convert::From<f64>` is not implemented for `{integer}`
     |
     = help: the following other types implement trait `std::convert::From<T>`:
               `f128` implements `std::convert::From<bool>`
               `f128` implements `std::convert::From<f16>`
               `f128` implements `std::convert::From<f32>`
               `f128` implements `std::convert::From<f64>`
               `f128` implements `std::convert::From<i16>`
               `f128` implements `std::convert::From<i32>`
               `f128` implements `std::convert::From<i8>`
               `f128` implements `std::convert::From<u16>`
             and 108 others
```
