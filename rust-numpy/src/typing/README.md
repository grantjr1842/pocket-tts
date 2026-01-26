# NumPy Typing Module

This module provides NumPy-compatible type annotations and aliases, offering parity with NumPy's `numpy.typing` module. It enables static type checking and better IDE support for NumPy operations in Rust.

## Overview

The typing module includes:

- **NDArray**: Runtime type annotations for arrays with specified dtypes
- **ArrayLike**: Objects that can be converted to arrays
- **DtypeLike**: Objects that can be converted to dtypes
- **NBitBase**: Bit-width type hierarchy for precision-based typing
- **Type Aliases**: Comprehensive type aliases matching NumPy's structure

## Core Components

### NDArray

The `NDArray<T>` type alias provides NumPy-compatible type annotations for arrays:

```rust
use rust_numpy::typing::*;

// Type annotation for float64 array
let arr: NDArray<f64> = array![1.0, 2.0, 3.0];

// Type annotation for integer array
let int_arr: NDArray<i32> = array![1, 2, 3];
```

### ArrayLike

The `ArrayLike<T>` trait represents objects that can be converted to arrays:

```rust
use rust_numpy::typing::*;

fn process_data<T: Clone + Default + 'static>(data: ArrayLike<T>) -> Array<T> {
    asarray(data)
}

// Works with various input types
process_data(vec![1, 2, 3]);           // Vec<T>
process_data([1, 2, 3]);              // [T; N]
process_data(&[1, 2, 3]);             // &[T]
process_data(array![1, 2, 3]);       // Array<T>
```

### DtypeLike

The `DtypeLike` trait represents objects that can be converted to dtypes:

```rust
use rust_numpy::typing::*;

fn create_array_with_dtype<T: DtypeLike>(dtype_like: T) -> Dtype {
    dtype_like.to_dtype()
}

// Works with various input types
create_array_with_dtype(Dtype::Int32 { byteorder: None });  // Dtype
create_array_with_dtype("int32");                              // &str
create_array_with_dtype("float64".to_string());                // String
create_array_with_dtype(42i32);                                // i32
create_array_with_dtype(3.14f64);                               // f64
```

## Bit-Width Types

### NBitBase

The `NBitBase` trait provides a hierarchy for bit-width type annotations:

```rust
use rust_numpy::typing::bitwidth::*;

fn add_floats<T: NBitBase>(a: T, b: T) -> T
where
    T: std::ops::Add<Output = T>,
{
    a + b
}

// Works with any floating-point precision
let result1 = add_floats(1.0f32);  // Float32Bit
let result2 = add_floats(1.0f64);  // Float64Bit
```

### Available Bit-Width Types

#### Integer Types
- `Int8Bit`, `Int16Bit`, `Int32Bit`, `Int64Bit` - Signed integers
- `UInt8Bit`, `UInt16Bit`, `UInt32Bit`, `UInt64Bit` - Unsigned integers

#### Float Types
- `Float16Bit`, `Float32Bit`, `Float64Bit` - Floating-point numbers

#### Complex Types
- `Complex32Bit`, `Complex64Bit`, `Complex128Bit`, `Complex256Bit` - Complex numbers

### Dtype Getter

The `DtypeGetter` provides NumPy-like dtype access:

```rust
use rust_numpy::typing::dtype_getter::*;

// Get dtype from bit-width type
let dtype = DtypeGetter::get::<Float64Bit>();

// Or use the convenience function
let dtype = dtype::<Int32Bit>();
```

## Type Aliases

### Array Type Aliases

Specific array types for common dtypes:

```rust
use rust_numpy::typing::*;

let int_arr: Int32Array = array![1, 2, 3];
let float_arr: Float64Array = array![1.0, 2.0, 3.0];
let bool_arr: BoolArray = array![true, false, true];
let complex_arr: Complex128Array = array![
    Complex::new(1.0, 2.0),
    Complex::new(3.0, 4.0)
];
```

Available array types:
- `Int8Array`, `Int16Array`, `Int32Array`, `Int64Array`
- `UInt8Array`, `UInt16Array`, `UInt32Array`, `UInt64Array`
- `Float32Array`, `Float64Array`
- `Complex64Array`, `Complex128Array`
- `BoolArray`

### Additional Type Aliases

```rust
use rust_numpy::typing::*;

// Shape and indexing
let shape: ShapeLike = vec![10, 20, 30];
let index: SupportsIndex = 42isize;

// Scalar types (when available)
// let scalar: Scalar = ...;
// let number: Number = ...;
```

## Module Organization

### Structure

The typing module is organized to match NumPy's structure:

```
typing/
├── mod.rs              # Main module with core types
├── bitwidth.rs         # NBitBase and bit-width types
├── dtype_getter.rs     # Dtype getter functionality
├── ndarray_types.rs    # Specific array type aliases
├── aliases.rs          # Additional type aliases
└── prelude.rs          # Commonly used types
```

### Exports

The module re-exports all necessary types for convenience:

```rust
use rust_numpy::typing::*;

// Core types
NDArray<T>
ArrayLike<T>
DtypeLike

// Bit-width types
Int8Bit, Float64Bit, Complex128Bit, etc.

// Array type aliases
Int32Array, Float64Array, BoolArray, etc.

// Convenience functions
dtype::<T>()
```

## Usage Examples

### Function Signatures

```rust
use rust_numpy::typing::*;

// Generic function with type annotations
fn process_array<T: Clone + Default + 'static>(
    data: ArrayLike<T>,
    dtype: DtypeLike
) -> NDArray<T> {
    let array = asarray(data);
    // Process array with specified dtype
    array
}

// Usage
let result = process_array(vec![1, 2, 3], "int32");
```

### Type-Safe Operations

```rust
use rust_numpy::typing::*;

// Type-safe matrix multiplication
fn matrix_multiply<T>(a: NDArray<T>, b: NDArray<T>) -> NDArray<T>
where
    T: std::ops::Mul<Output = T> + Clone + Default + 'static,
{
    // Implementation
    a
}

// Works with any numeric type
let result1 = matrix_multiply(int_matrix, int_matrix);
let result2 = matrix_multiply(float_matrix, float_matrix);
```

### Precision-Based Functions

```rust
use rust_numpy::typing::bitwidth::*;

// Function that preserves precision
fn precision_preserving_add<T: NBitBase>(a: T, b: T) -> T
where
    T: std::ops::Add<Output = T>,
{
    a + b
}

// Maintains input precision
let result_f32 = precision_preserving_add(1.0f32, 2.0f32);
let result_f64 = precision_preserving_add(1.0f64, 2.0f64);
```

## Compatibility with NumPy

This typing module aims for 100% compatibility with NumPy's typing system:

- **ArrayLike**: Matches NumPy's `numpy.typing.ArrayLike`
- **DtypeLike**: Matches NumPy's `numpy.typing.DTypeLike`
- **NDArray**: Matches NumPy's `numpy.typing.NDArray`
- **NBitBase**: Matches NumPy's `numpy.typing.NBitBase` (deprecated in NumPy 2.3+)

## Differences from Runtime NumPy

The typing system is stricter than runtime NumPy for better type safety:

1. **ArrayLike** avoids creating object arrays from generators
2. **DtypeLike** discourages dictionary-based dtype specifications
3. **NBitBase** provides static precision relationships
4. **0D arrays** are treated as arrays rather than scalars in type annotations

## Best Practices

### Use Type Annotations

```rust
// Good: Explicit type annotations
fn process_floats(data: ArrayLike<f64>) -> NDArray<f64> {
    asarray(data)
}

// Avoid: Generic without bounds
fn process_data<T>(data: T) -> T {
    // Less type-safe
}
```

### Prefer Specific Types

```rust
// Good: Specific array type
fn process_integers(data: Int32Array) -> Int32Array {
    data
}

// Acceptable: Generic with bounds
fn process_numbers<T: DtypeLike>(data: NDArray<T>) -> NDArray<T> {
    data
}
```

### Use Prelude for Common Imports

```rust
// Good: Import from prelude
use rust_numpy::typing::prelude::*;

fn example(arr: Int32Array) -> Float64Array {
    // All common types available
    arr
}
```

## Testing

The typing module includes comprehensive tests:

```bash
# Run all typing tests
cargo test typing

# Run specific test categories
cargo test typing::tests::ndarray_type_alias
cargo test typing::tests::array_like_trait
cargo test typing::tests::dtype_like_trait
```

## Migration Guide

### From NumPy Python

```python
# NumPy Python
import numpy as np
from numpy.typing import NDArray, ArrayLike, DTypeLike

def process_data(data: ArrayLike[np.float64]) -> NDArray[np.float64]:
    return np.array(data, dtype=np.float64)
```

```rust
// Rust NumPy equivalent
use rust_numpy::typing::*;

fn process_data(data: ArrayLike<f64>) -> NDArray<f64> {
    asarray(data)
}
```

### From Untyped Rust

```rust
// Before: Untyped
fn process_data(data: Vec<f64>) -> Array<f64> {
    Array::from_data(data, vec![data.len()])
}

// After: With typing annotations
fn process_data(data: ArrayLike<f64>) -> NDArray<f64> {
    asarray(data)
}
```

## Future Enhancements

Planned improvements to match NumPy's evolving typing system:

1. **Shape typing**: More precise shape annotations when PEP 646 is stable
2. **Generic dtypes**: Enhanced generic dtype support
3. **Protocol traits**: Better support for `__array__` protocol
4. **Structured dtypes**: Type-safe structured array support

## References

- [NumPy Typing Documentation](https://numpy.org/doc/stable/reference/typing.html)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 646 - Variadic Generics](https://peps.python.org/pep-0646/)
- [Rust Type System](https://doc.rust-lang.org/book/ch10-00-types.html)
