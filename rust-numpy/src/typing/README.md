# Typing Module Documentation

This module provides NumPy-compatible type annotations and typing utilities for the rust-numpy library. It enables runtime type annotations similar to NumPy's `typing` module in Python.

## Overview

The typing module includes:

- **NDArray**: Type alias for runtime type annotations with dtype-specific variants
- **ArrayLike**: Trait for array-convertible objects
- **DtypeLike**: Trait for dtype-convertible objects
- **Predefined type aliases**: Common array types for specific dtypes

## NDArray Type Alias

The `NDArray<T>` type alias provides NumPy-compatible type annotations for arrays with specified dtypes. It enables annotating arrays with given dtype and unspecified shape, similar to NumPy's `typing.NDArray` in Python.

### Basic Usage

```rust
use rust_numpy::typing::*;

// Type annotation for float64 array
let arr: NDArray<f64> = array![1.0, 2.0, 3.0];

// Type annotation for integer array
let int_arr: NDArray<i32> = array![1, 2, 3];
```

### Specific Dtype Array Types

The module provides predefined type aliases for common dtypes:

```rust
use rust_numpy::typing::prelude::*;

// Signed integer arrays
let int8_arr: Int8Array = array![1i8, 2i8, 3i8];
let int16_arr: Int16Array = array![1i16, 2i16, 3i16];
let int32_arr: Int32Array = array![1, 2, 3];
let int64_arr: Int64Array = array![1i64, 2i64, 3i64];

// Unsigned integer arrays
let uint8_arr: UInt8Array = array![1u8, 2u8, 3u8];
let uint16_arr: UInt16Array = array![1u16, 2u16, 3u16];
let uint32_arr: UInt32Array = array![1u32, 2u32, 3u32];
let uint64_arr: UInt64Array = array![1u64, 2u64, 3u64];

// Float arrays
let float32_arr: Float32Array = array![1.0f32, 2.0f32, 3.0f32];
let float64_arr: Float64Array = array![1.0, 2.0, 3.0];

// Complex arrays
let complex64_arr: Complex64Array = array![
    num_complex::Complex::new(1.0f32, 2.0f32),
    num_complex::Complex::new(3.0f32, 4.0f32)
];
let complex128_arr: Complex128Array = array![
    num_complex::Complex::new(1.0, 2.0),
    num_complex::Complex::new(3.0, 4.0)
];

// Boolean array
let bool_arr: BoolArray = array![true, false, true];
```

## ArrayLike Trait

The `ArrayLike<T>` trait represents objects that can be converted to arrays, similar to NumPy's `typing.ArrayLike` in Python. It includes various types that can be used as input for array creation functions.

### Supported Types

The `ArrayLike` trait is implemented for:

- `Array<T>` - Existing arrays
- `Vec<T>` - Rust vectors
- `[T; N]` - Fixed-size arrays
- `&[T]` - Slices

### Usage Example

```rust
use rust_numpy::typing::*;

fn process_data<T: Clone + Default + 'static>(data: ArrayLike<T>) -> Array<T> {
    data.to_array().unwrap()
}

// Works with various input types
let vec_data = vec![1, 2, 3];
let array_data = [1, 2, 3];
let slice_data = &[1, 2, 3];

let result1 = process_data(vec_data);
let result2 = process_data(array_data);
let result3 = process_data(slice_data);
```

## DtypeLike Trait

The `DtypeLike` trait represents objects that can be converted to dtypes, similar to NumPy's `typing.DtypeLike` in Python. It includes various ways to specify data types in NumPy operations.

### Supported Types

The `DtypeLike` trait is implemented for:

- `Dtype` - Existing dtype objects
- `&str` - String references (e.g., "int32", "float64")
- `String` - Owned strings
- Primitive types (`i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`)

### Usage Example

```rust
use rust_numpy::typing::*;
use rust_numpy::Dtype;

fn create_array_with_dtype<T: DtypeLike>(dtype_like: T) -> Dtype {
    dtype_like.to_dtype()
}

// Works with various input types
let dtype1 = create_array_with_dtype(Dtype::Int32 { byteorder: None });
let dtype2 = create_array_with_dtype("int32");
let dtype3 = create_array_with_dtype(String::from("float64"));
let dtype4 = create_array_with_dtype(42i32);
let dtype5 = create_array_with_dtype(3.14f64);
let dtype6 = create_array_with_dtype(true);
```

## String Dtype Parsing

The `DtypeLike` implementation for strings supports NumPy-compatible dtype strings:

```rust
// NumPy-style dtype strings
assert_eq!("int32".to_dtype(), Dtype::Int32 { byteorder: None });
assert_eq!("float64".to_dtype(), Dtype::Float64 { byteorder: None });
assert_eq!("bool".to_dtype(), Dtype::Bool);

// With np. prefix
assert_eq!("np.int32".to_dtype(), Dtype::Int32 { byteorder: None });
assert_eq!("np.float64".to_dtype(), Dtype::Float64 { byteorder: None });

// Short aliases
assert_eq!("i4".to_dtype(), Dtype::Int32 { byteorder: None });
assert_eq!("f8".to_dtype(), Dtype::Float64 { byteorder: None });
```

## Prelude Module

The `prelude` module provides convenient exports for commonly used typing aliases:

```rust
use rust_numpy::typing::prelude::*;

// All common types are available without qualification
fn example_function() -> (Int32Array, Float64Array, BoolArray) {
    (
        array![1, 2, 3],
        array![1.0, 2.0, 3.0],
        array![true, false, true]
    )
}
```

## Integration with IDE and Type Checking

These type aliases provide better IDE support and type checking:

```rust
// IDE can provide better autocomplete and type information
fn process_arrays(
    int_data: Int32Array,
    float_data: Float64Array,
    mask: BoolArray
) -> Float64Array {
    // Function implementation with clear type annotations
    // IDE can provide better error messages and suggestions
}
```

## Compatibility with NumPy

This typing system is designed to be compatible with NumPy's typing module:

| NumPy Type                   | Rust Type      | Description                 |
|------------------------------|----------------|-----------------------------|
| `typing.NDArray[np.int32]`   | `Int32Array`   | 32-bit signed integer array |
| `typing.NDArray[np.float64]` | `Float64Array` | 64-bit float array          |
| `typing.NDArray[np.bool_]`   | `BoolArray`    | Boolean array               |
| `typing.ArrayLike`           | `ArrayLike<T>` | Array-convertible objects   |
| `typing.DtypeLike`           | `DtypeLike`    | Dtype-convertible objects   |

## Advanced Usage

### Generic Functions with Type Constraints

```rust
use rust_numpy::typing::*;

fn process_generic_array<T, A>(data: A) -> Array<T>
where
    T: Clone + Default + 'static,
    A: ArrayLike<T>,
{
    data.to_array().unwrap()
}

fn create_with_dtype<D: DtypeLike>(dtype: D) -> Dtype {
    dtype.to_dtype()
}
```

### Type Annotations in Function Signatures

```rust
use rust_numpy::typing::prelude::*;

pub fn matrix_multiply(
    a: Float64Array,
    b: Float64Array
) -> Result<Float64Array, NumPyError> {
    // Implementation with clear type annotations
}

pub fn apply_mask(
    data: Float64Array,
    mask: BoolArray
) -> Float64Array {
    // Implementation with type safety
}
```

## Performance Considerations

- Type aliases have zero runtime overhead - they're just compile-time annotations
- `ArrayLike` conversions may involve cloning data for some input types
- `DtypeLike` conversions are generally cheap (mostly string parsing for string inputs)

## Error Handling

The `ArrayLike::to_array()` method returns a `Result<Array<T>, NumPyError>` to handle potential conversion errors. The `DtypeLike::to_dtype()` method returns a `Dtype` directly, with invalid strings falling back to `Float64`.

## Testing

The module includes comprehensive tests covering:

- All type aliases and their behavior
- `ArrayLike` implementations for all supported types
- `DtypeLike` implementations for all supported types
- String dtype parsing with valid and invalid inputs
- Integration with the prelude module

Run tests with:
```bash
cargo test typing
```
