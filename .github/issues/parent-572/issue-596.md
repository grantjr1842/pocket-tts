# [Sub-Issue] Create Enhanced NDArray with Shape/Dtype Params (#572)

**Issue #596**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/596  
**Created:** 2026-01-29T03:10:37Z  
**Updated:** 2026-01-29T03:24:04Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:typing, ralph/numpy:impl  

---

## Parent Issue
Part of #572 - typing module incomplete vs NumPy.typing

## Scope
Create Comprehensive NDArray Type with Shape and Dtype Parameters

## Background
NumPy's NDArray type includes shape and dtype information. rust-numpy's current NDArray is just an alias for Array.

## Tasks

### 1. Create Shape Types
```rust
// src/typing/shape.rs

/// Shape parameter for type-level shape checking
pub trait Shape: 'static {
    const NDIM: usize;
    const SIZE: Option<usize>;  // None for dynamic
    fn as_slice(&self) -> &[usize];
}

/// Dynamic shape (unknown at compile time)
pub struct Dynamic;
impl Shape for Dynamic {
    const NDIM: usize = 0;  // Unknown
    const SIZE: Option<usize> = None;
    fn as_slice(&self) -> &[usize] { &[] }
}

/// Fixed shape with const generics
pub struct Shape1<const N: usize>;
impl<const N: usize> Shape for Shape1<N> {
    const NDIM: usize = 1;
    const SIZE: Option<usize> = Some(N);
    fn as_slice(&self) -> &[usize] { &[N] }
}

pub struct Shape2<const M: usize, const N: usize>;
impl<const M: usize, const N: usize> Shape for Shape2<M, N> {
    const NDIM: usize = 2;
    const SIZE: Option<usize> = Some(M * N);
    fn as_slice(&self) -> &[usize] { &[M, N] }
}

// ... etc for Shape3, Shape4, etc.

/// Runtime shape
#[derive(Clone, Debug)]
pub struct RuntimeShape(Vec<usize>);
impl Shape for RuntimeShape {
    const NDIM: usize = 0;  // Dynamic
    const SIZE: Option<usize> = None;
    fn as_slice(&self) -> &[usize] { &self.0 }
}
```

### 2. Create Enhanced NDArray Type
```rust
// src/typing/mod.rs

use crate::types::{Generic, int8, int16, int32, int64, float32, float64, complex64, complex128};

/// Typed array with shape and dtype information
pub struct NDArray<T: Generic, S: Shape = RuntimeShape> {
    array: Array<T>,
    shape: S,
}

impl<T: Generic, S: Shape> NDArray<T, S> {
    pub fn new(array: Array<T>, shape: S) -> Self {
        Self { array, shape }
    }
    
    pub fn shape(&self) -> &S { &self.shape }
    pub fn into_array(self) -> Array<T> { self.array }
    pub fn as_array(&self) -> &Array<T> { &self.array }
    
    /// Reshape to different type
    pub fn reshape<S2: Shape>(self, new_shape: S2) -> NDArray<T, S2> {
        NDArray {
            array: self.array,
            shape: new_shape,
        }
    }
}

// Type aliases for common cases
pub type Array1<T> = NDArray<T, Shape1<{ usize::MAX }>>;  // 1-D dynamic
pub type Array2<T> = NDArray<T, Shape2<{ usize::MAX }, { usize::MAX }>>;  // 2-D dynamic
pub type Vector<T> = NDArray<T, Dynamic>;
pub type Matrix<T> = NDArray<T, Dynamic>;

// Dtype-specific aliases
pub type Float64Array<S = RuntimeShape> = NDArray<float64, S>;
pub type Float32Array<S = RuntimeShape> = NDArray<float32, S>;
pub type Int32Array<S = RuntimeShape> = NDArray<int32, S>;
pub type Int64Array<S = RuntimeShape> = NDArray<int64, S>;
pub type Complex64Array<S = RuntimeShape> = NDArray<complex64, S>;
pub type Complex128Array<S = RuntimeShape> = NDArray<complex128, S>;
pub type BoolArray<S = RuntimeShape> = NDArray<bool_, S>;
```

### 3. Create DType Parameter
```rust
/// Dtype marker trait
pub trait DTypeMarker: 'static {
    fn dtype() -> Dtype;
}

// Implement for all types
impl DTypeMarker for int8 {
    fn dtype() -> Dtype { Dtype::Int8 { byteorder: None } }
}
// ... etc

/// Array with compile-time dtype
pub struct TypedArray<D: DTypeMarker, S: Shape = RuntimeShape> {
    data: Vec<u8>,  // Raw bytes, interpreted as D
    shape: S,
    phantom: PhantomData<D>,
}
```

### 4. Export Types
```rust
// src/typing/mod.rs
pub use shape::{Shape, Dynamic, Shape1, Shape2, Shape3, Shape4, RuntimeShape};
pub use types::{
    NDArray, Array1, Array2, Vector, Matrix,
    Float64Array, Float32Array, Int32Array, Int64Array,
    Complex64Array, Complex128Array, BoolArray,
};
```

## Files to Create/Modify
- NEW: `src/typing/shape.rs` - Shape types
- `src/typing/mod.rs` - NDArray type
- Update existing NDArray alias

## Acceptance Criteria
- [ ] Shape trait with const generics
- [ ] NDArray with shape and dtype parameters
- [ ] Type aliases for common cases
- [ ] Backward compatible with existing Array

## Conflict Avoidance
⚠️ **COORDINATION:**
- May change NDArray definition - coordinate
- Backward compatibility important

