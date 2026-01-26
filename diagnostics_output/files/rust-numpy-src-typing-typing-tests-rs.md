# Diagnostics: rust-numpy/src/typing/typing_tests.rs

**Errors:** 32 | **Warnings:** 3

---

## Errors

### Line 15 (Col 18)
**ERROR[E0412]**: cannot find type `NDArray` in this scope

```
error[E0412]: cannot find type `NDArray` in this scope
  --> src/typing/typing_tests.rs:15:18
   |
15 |         let arr: NDArray<f64> = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
   |                  ^^^^^^^
   |
  ::: src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- similarly named struct `Array` defined here
   |
```

### Line 23 (Col 23)
**ERROR[E0412]**: cannot find type `Int8Array` in this scope

```
error[E0412]: cannot find type `Int8Array` in this scope
  --> src/typing/typing_tests.rs:23:23
   |
23 |         let int8_arr: Int8Array = Array::from_data(vec![1i8, 2i8, 3i8], vec![3]);
   |                       ^^^^^^^^^ not found in this scope
   |
```

### Line 24 (Col 24)
**ERROR[E0412]**: cannot find type `Int16Array` in this scope

```
error[E0412]: cannot find type `Int16Array` in this scope
  --> src/typing/typing_tests.rs:24:24
   |
24 |         let int16_arr: Int16Array = Array::from_data(vec![1i16, 2i16, 3i16], vec![3]);
   |                        ^^^^^^^^^^ not found in this scope
   |
```

### Line 25 (Col 24)
**ERROR[E0412]**: cannot find type `Int32Array` in this scope

```
error[E0412]: cannot find type `Int32Array` in this scope
  --> src/typing/typing_tests.rs:25:24
   |
25 |         let int32_arr: Int32Array = Array::from_data(vec![1i32, 2i32, 3i32], vec![3]);
   |                        ^^^^^^^^^^ not found in this scope
   |
```

### Line 26 (Col 24)
**ERROR[E0412]**: cannot find type `Int64Array` in this scope

```
error[E0412]: cannot find type `Int64Array` in this scope
  --> src/typing/typing_tests.rs:26:24
   |
26 |         let int64_arr: Int64Array = Array::from_data(vec![1i64, 2i64, 3i64], vec![3]);
   |                        ^^^^^^^^^^ not found in this scope
   |
```

### Line 28 (Col 24)
**ERROR[E0412]**: cannot find type `UInt8Array` in this scope

```
error[E0412]: cannot find type `UInt8Array` in this scope
  --> src/typing/typing_tests.rs:28:24
   |
28 |         let uint8_arr: UInt8Array = Array::from_data(vec![1u8, 2u8, 3u8], vec![3]);
   |                        ^^^^^^^^^^ not found in this scope
   |
```

### Line 29 (Col 25)
**ERROR[E0412]**: cannot find type `UInt16Array` in this scope

```
error[E0412]: cannot find type `UInt16Array` in this scope
  --> src/typing/typing_tests.rs:29:25
   |
29 |         let uint16_arr: UInt16Array = Array::from_data(vec![1u16, 2u16, 3u16], vec![3]);
   |                         ^^^^^^^^^^^ not found in this scope
   |
```

### Line 30 (Col 25)
**ERROR[E0412]**: cannot find type `UInt32Array` in this scope

```
error[E0412]: cannot find type `UInt32Array` in this scope
  --> src/typing/typing_tests.rs:30:25
   |
30 |         let uint32_arr: UInt32Array = Array::from_data(vec![1u32, 2u32, 3u32], vec![3]);
   |                         ^^^^^^^^^^^ not found in this scope
   |
```

### Line 31 (Col 25)
**ERROR[E0412]**: cannot find type `UInt64Array` in this scope

```
error[E0412]: cannot find type `UInt64Array` in this scope
  --> src/typing/typing_tests.rs:31:25
   |
31 |         let uint64_arr: UInt64Array = Array::from_data(vec![1u64, 2u64, 3u64], vec![3]);
   |                         ^^^^^^^^^^^ not found in this scope
   |
```

### Line 33 (Col 26)
**ERROR[E0412]**: cannot find type `Float32Array` in this scope

```
error[E0412]: cannot find type `Float32Array` in this scope
  --> src/typing/typing_tests.rs:33:26
   |
33 |         let float32_arr: Float32Array = Array::from_data(vec![1.0f32, 2.0f32, 3.0f32], vec![3]);
   |                          ^^^^^^^^^^^^ not found in this scope
   |
```

### Line 34 (Col 26)
**ERROR[E0412]**: cannot find type `Float64Array` in this scope

```
error[E0412]: cannot find type `Float64Array` in this scope
  --> src/typing/typing_tests.rs:34:26
   |
34 |         let float64_arr: Float64Array = Array::from_data(vec![1.0f64, 2.0f64, 3.0f64], vec![3]);
   |                          ^^^^^^^^^^^^ not found in this scope
   |
```

### Line 36 (Col 23)
**ERROR[E0412]**: cannot find type `BoolArray` in this scope

```
error[E0412]: cannot find type `BoolArray` in this scope
  --> src/typing/typing_tests.rs:36:23
   |
36 |         let bool_arr: BoolArray = Array::from_data(vec![true, false, true], vec![3]);
   |                       ^^^^^^^^^ not found in this scope
   |
```

### Line 53 (Col 72)
**ERROR[E0405]**: cannot find trait `ArrayLike` in this scope

```
error[E0405]: cannot find trait `ArrayLike` in this scope
  --> src/typing/typing_tests.rs:53:72
   |
53 |         fn process_array_like<T: Clone + Default + 'static>(data: &dyn ArrayLike<T>) -> Array<T> {
   |                                                                        ^^^^^^^^^ not found in this scope
   |
```

### Line 74 (Col 34)
**ERROR[E0405]**: cannot find trait `DtypeLike` in this scope

```
error[E0405]: cannot find trait `DtypeLike` in this scope
  --> src/typing/typing_tests.rs:74:34
   |
74 |         fn process_dtype_like<T: DtypeLike>(data: &T) -> Dtype {
   |                                  ^^^^^^^^^ not found in this scope
   |
```

### Line 89 (Col 20)
**ERROR[E0412]**: cannot find type `ShapeLike` in this scope

```
error[E0412]: cannot find type `ShapeLike` in this scope
  --> src/typing/typing_tests.rs:89:20
   |
89 |         let shape: ShapeLike = vec![10, 20, 30];
   |                    ^^^^^^^^^ not found in this scope
   |
```

### Line 97 (Col 20)
**ERROR[E0412]**: cannot find type `SupportsIndex` in this scope

```
error[E0412]: cannot find type `SupportsIndex` in this scope
  --> src/typing/typing_tests.rs:97:20
   |
97 |         let index: SupportsIndex = 42isize;
   |                    ^^^^^^^^^^^^^ not found in this scope
   |
```

### Line 158 (Col 40)
**ERROR[E0412]**: cannot find type `Int32Bit` in this scope

```
error[E0412]: cannot find type `Int32Bit` in this scope
   --> src/typing/typing_tests.rs:158:40
    |
158 |         let dtype = DtypeGetter::get::<Int32Bit>();
    |                                        ^^^^^^^^ not found in this scope
    |
```

### Line 164 (Col 40)
**ERROR[E0412]**: cannot find type `Float64Bit` in this scope

```
error[E0412]: cannot find type `Float64Bit` in this scope
   --> src/typing/typing_tests.rs:164:40
    |
164 |         let dtype = DtypeGetter::get::<Float64Bit>();
    |                                        ^^^^^^^^^^ not found in this scope
    |
```

### Line 170 (Col 40)
**ERROR[E0412]**: cannot find type `Complex128Bit` in this scope

```
error[E0412]: cannot find type `Complex128Bit` in this scope
   --> src/typing/typing_tests.rs:170:40
    |
170 |         let dtype = DtypeGetter::get::<Complex128Bit>();
    |                                        ^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 182 (Col 21)
**ERROR[E0433]**: failed to resolve: use of undeclared type `Int32Bit`

```
error[E0433]: failed to resolve: use of undeclared type `Int32Bit`
   --> src/typing/typing_tests.rs:182:21
    |
182 |         let dtype = Int32Bit::to_dtype();
    |                     ^^^^^^^^ use of undeclared type `Int32Bit`
    |
```

### Line 188 (Col 21)
**ERROR[E0433]**: failed to resolve: use of undeclared type `Float64Bit`

```
error[E0433]: failed to resolve: use of undeclared type `Float64Bit`
   --> src/typing/typing_tests.rs:188:21
    |
188 |         let dtype = Float64Bit::to_dtype();
    |                     ^^^^^^^^^^ use of undeclared type `Float64Bit`
    |
```

### Line 200 (Col 29)
**ERROR[E0412]**: cannot find type `Int32Bit` in this scope

```
error[E0412]: cannot find type `Int32Bit` in this scope
   --> src/typing/typing_tests.rs:200:29
    |
200 |         let dtype = dtype::<Int32Bit>();
    |                             ^^^^^^^^ not found in this scope
    |
```

### Line 206 (Col 29)
**ERROR[E0412]**: cannot find type `Float64Bit` in this scope

```
error[E0412]: cannot find type `Float64Bit` in this scope
   --> src/typing/typing_tests.rs:206:29
    |
206 |         let dtype = dtype::<Float64Bit>();
    |                             ^^^^^^^^^^ not found in this scope
    |
```

### Line 216 (Col 36)
**ERROR[E0412]**: cannot find type `NDArray` in this scope

```
error[E0412]: cannot find type `NDArray` in this scope
   --> src/typing/typing_tests.rs:216:36
    |
216 |         fn process_ndarray<T>(arr: NDArray<T>) -> usize {
    |                                    ^^^^^^^
    |
   ::: src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- similarly named struct `Array` defined here
    |
```

### Line 225 (Col 35)
**ERROR[E0412]**: cannot find type `Int32Array` in this scope

```
error[E0412]: cannot find type `Int32Array` in this scope
   --> src/typing/typing_tests.rs:225:35
    |
225 |         fn process_int_array(arr: Int32Array) -> i32 {
    |                                   ^^^^^^^^^^ not found in this scope
    |
```

### Line 265 (Col 66)
**ERROR[E0412]**: cannot find type `Int8Bit` in this scope

```
error[E0412]: cannot find type `Int8Bit` in this scope
   --> src/typing/typing_tests.rs:265:66
    |
265 |         let _int8_bit: std::any::TypeId = std::any::TypeId::of::<Int8Bit>();
    |                                                                  ^^^^^^^ not found in this scope
    |
```

### Line 266 (Col 69)
**ERROR[E0412]**: cannot find type `Float64Bit` in this scope

```
error[E0412]: cannot find type `Float64Bit` in this scope
   --> src/typing/typing_tests.rs:266:69
    |
266 |         let _float64_bit: std::any::TypeId = std::any::TypeId::of::<Float64Bit>();
    |                                                                     ^^^^^^^^^^ not found in this scope
    |
```

### Line 267 (Col 72)
**ERROR[E0412]**: cannot find type `Complex128Bit` in this scope

```
error[E0412]: cannot find type `Complex128Bit` in this scope
   --> src/typing/typing_tests.rs:267:72
    |
267 |         let _complex128_bit: std::any::TypeId = std::any::TypeId::of::<Complex128Bit>();
    |                                                                        ^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 270 (Col 70)
**ERROR[E0412]**: cannot find type `DtypeGetter` in this scope

```
error[E0412]: cannot find type `DtypeGetter` in this scope
   --> src/typing/typing_tests.rs:270:70
    |
270 |         let _dtype_getter: std::any::TypeId = std::any::TypeId::of::<DtypeGetter>();
    |                                                                      ^^^^^^^^^^^ not found in this scope
    |
```

### Line 273 (Col 68)
**ERROR[E0412]**: cannot find type `ShapeLike` in this scope

```
error[E0412]: cannot find type `ShapeLike` in this scope
   --> src/typing/typing_tests.rs:273:68
    |
273 |         let _shape_like: std::any::TypeId = std::any::TypeId::of::<ShapeLike>();
    |                                                                    ^^^^^^^^^
    |
   ::: src/typing/mod.rs:137:1
    |
137 | pub trait DtypeLike {
    | ------------------- similarly named trait `DtypeLike` defined here
    |
```

### Line 274 (Col 72)
**ERROR[E0412]**: cannot find type `SupportsIndex` in this scope

```
error[E0412]: cannot find type `SupportsIndex` in this scope
   --> src/typing/typing_tests.rs:274:72
    |
274 |         let _supports_index: std::any::TypeId = std::any::TypeId::of::<SupportsIndex>();
    |                                                                        ^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 308 (Col 22)
**ERROR[E0412]**: cannot find type `NDArray` in this scope

```
error[E0412]: cannot find type `NDArray` in this scope
   --> src/typing/typing_tests.rs:308:22
    |
308 |             let arr: NDArray<f64> = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
    |                      ^^^^^^^
    |
   ::: src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- similarly named struct `Array` defined here
    |
```

## Warnings

### Line 8 (Col 9)
**WARNING[unused_imports]**: warning: unused import: `super::*`

```
warning: unused import: `super::*`
  |
8 |     use super::*;
  |         ^^^^^^^^
```

### Line 179 (Col 13)
**WARNING[unused_imports]**: warning: unused import: `super::super::dtype_getter::*`

```
warning: unused import: `super::super::dtype_getter::*`
   --> src/typing/typing_tests.rs:179:13
    |
179 |         use super::super::dtype_getter::*;
    |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

### Line 237 (Col 15)
**WARNING[unexpected_cfgs]**: warning: unexpected `cfg` condition value: `num-complex`

```
warning: unexpected `cfg` condition value: `num-complex`
   --> src/typing/typing_tests.rs:237:15
    |
237 |         #[cfg(feature = "num-complex")]
    |               ^^^^^^^^^^^^^^^^^^^^^^^
    |
    = note: expected values for `feature` are: `chrono`, `cuda`, `cudarc`, `datetime`, `default`, `dyn-stack`, `faer`, `metal`, `pure-linalg`, `rayon`, `reborrow`, `serde`, `simd`, `std`, and `wasm`
    = help: consider adding `num-complex` as a feature in `Cargo.toml`
    = note: see <https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html> for more information about checking conditional configuration
    = note: `#[warn(unexpected_cfgs)]` on by default
```
