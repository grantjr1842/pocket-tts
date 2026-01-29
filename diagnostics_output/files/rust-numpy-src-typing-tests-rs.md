# Diagnostics: rust-numpy/src/typing/tests.rs

**Errors:** 16 | **Warnings:** 1

---

## Errors

### Line 10 (Col 18)
**ERROR[E0412]**: cannot find type `NDArray` in this scope

```
error[E0412]: cannot find type `NDArray` in this scope
  --> src/typing/tests.rs:10:18
   |
10 |         let arr: NDArray<f64> = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
   |                  ^^^^^^^
   |
  ::: src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- similarly named struct `Array` defined here
   |
```

### Line 18 (Col 22)
**ERROR[E0412]**: cannot find type `Int32Array` in this scope

```
error[E0412]: cannot find type `Int32Array` in this scope
  --> src/typing/tests.rs:18:22
   |
18 |         let int_arr: Int32Array = Array::from_data(vec![1, 2, 3], vec![3]);
   |                      ^^^^^^^^^^ not found in this scope
   |
```

### Line 21 (Col 24)
**ERROR[E0412]**: cannot find type `Float64Array` in this scope

```
error[E0412]: cannot find type `Float64Array` in this scope
  --> src/typing/tests.rs:21:24
   |
21 |         let float_arr: Float64Array = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
   |                        ^^^^^^^^^^^^ not found in this scope
   |
```

### Line 24 (Col 23)
**ERROR[E0412]**: cannot find type `BoolArray` in this scope

```
error[E0412]: cannot find type `BoolArray` in this scope
  --> src/typing/tests.rs:24:23
   |
24 |         let bool_arr: BoolArray = Array::from_data(vec![true, false, true], vec![3]);
   |                       ^^^^^^^^^ not found in this scope
   |
```

### Line 124 (Col 28)
**ERROR[E0412]**: cannot find type `Complex64Array` in this scope

```
error[E0412]: cannot find type `Complex64Array` in this scope
   --> src/typing/tests.rs:124:28
    |
124 |         let complex32_arr: Complex64Array = Array::from_data(
    |                            ^^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 134 (Col 20)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:134:20
    |
134 |         assert_eq!(DtypeLike::to_dtype(&0i8), Dtype::Int8 { byteorder: None });
    |                    ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 135 (Col 20)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:135:20
    |
135 |         assert_eq!(DtypeLike::to_dtype(&0i16), Dtype::Int16 { byteorder: None });
    |                    ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 136 (Col 20)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:136:20
    |
136 |         assert_eq!(DtypeLike::to_dtype(&0i32), Dtype::Int32 { byteorder: None });
    |                    ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 137 (Col 20)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:137:20
    |
137 |         assert_eq!(DtypeLike::to_dtype(&0i64), Dtype::Int64 { byteorder: None });
    |                    ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 138 (Col 20)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:138:20
    |
138 |         assert_eq!(DtypeLike::to_dtype(&0u8), Dtype::UInt8 { byteorder: None });
    |                    ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 140 (Col 13)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:140:13
    |
140 |             DtypeLike::to_dtype(&0u16),
    |             ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 144 (Col 13)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:144:13
    |
144 |             DtypeLike::to_dtype(&0u32),
    |             ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 148 (Col 13)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:148:13
    |
148 |             DtypeLike::to_dtype(&0u64),
    |             ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 152 (Col 13)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:152:13
    |
152 |             DtypeLike::to_dtype(&0.0f32),
    |             ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 156 (Col 13)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:156:13
    |
156 |             DtypeLike::to_dtype(&0.0f64),
    |             ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

### Line 159 (Col 20)
**ERROR[E0433]**: failed to resolve: use of undeclared type `DtypeLike`

```
error[E0433]: failed to resolve: use of undeclared type `DtypeLike`
   --> src/typing/tests.rs:159:20
    |
159 |         assert_eq!(DtypeLike::to_dtype(&true), Dtype::Bool);
    |                    ^^^^^^^^^ use of undeclared type `DtypeLike`
    |
```

## Warnings

### Line 3 (Col 9)
**WARNING[unused_imports]**: warning: unused import: `super::*`

```
warning: unused import: `super::*`
  |
3 |     use super::*;
  |         ^^^^^^^^
```
