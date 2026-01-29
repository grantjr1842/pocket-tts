# rust-numpy: typing module incomplete vs NumPy.typing

**Issue #572**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/572  
**Created:** 2026-01-29T02:47:09Z  
**Updated:** 2026-01-29T03:24:02Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:typing, ralph/numpy:audit  

---

## Summary
NumPy's `numpy.typing` module provides comprehensive type annotations for static type checking. rust-numpy's `typing` module has a partial implementation that doesn't match NumPy's structure.

## NumPy typing Module Reference

### NumPy 2.0 typing Structure
```python
# numpy/typing/__init__.py
from ._array_like import _ArrayLike
from ._char_codes import _CharCode
from ._dtype_like import _DTypeLike
from ._generic_alias import _GenericAlias
from ._nested_sequence import _NestedSequence
from ._scalars import _Scalar
from ._shape import _Shape

# Public exports
ArrayLike      # Type for array-like objects
DTypeLike      # Type for dtype-like objects
NDArray        # Generic array type with dtype
NBitBase       # Base for bit-width types
NumpyGeneric   # Alias for numpy.generic

# Type aliases module
from . import aliases
aliases.ArrayLike
aliases.DTypeLike
aliases.NDArray

# Shape types
_Shape = tuple[int, ...]
_ShapeLike = int | tuple[int, ...]

# Character codes for dtypes
_SCT = TypeVar("_SCT", bound=np.generic)

# Array protocols
SupportsArray  # Protocol for objects with __array__
SupportsDType  # Protocol for objects with __dtype__

# New in NumPy 2.0
StringDType    # New string dtype
```

### NumPy typing.mypy_plugin
```python
# Plugin for mypy to understand NumPy's type system
# Handles shape typing, dtype specificity, etc.
```

## Current rust-numpy typing State

### In `src/typing/mod.rs`
```rust
pub type NDArray<T> = Array<T>;

pub trait ArrayLike<T: Clone + Default + 'static> {
    fn to_array(&self) -> Result<Array<T>, crate::error::NumPyError>;
}

pub trait DtypeLike {
    fn to_dtype(&self) -> Dtype;
}

pub mod aliases {
    pub type ShapeLike = Vec<usize>;
    pub type SupportsIndex = isize;
    pub type Boolean = bool;
    // ... etc
}

pub mod prelude {
    pub use super::{ArrayLike, BoolArray, ..., NDArray, ...};
}

// Bit-width types
pub struct NBitBase;
pub struct nbit_8;
pub struct nbit_16;
pub struct nbit_32;
pub struct nbit_64;
pub struct nbit_128;
pub struct nbit_256;
```

### In `src/typing/bitwidth.rs`
```rust
// Bit-width type system for NumPy typing
pub trait BitWidth: 'static {}
pub struct NBitBase;
// ... sized types
```

### In `src/typing/dtype_getter.rs`
```rust
// Dtype getter functionality
pub trait DtypeGetter {
    fn dtype(&self) -> Dtype;
}
```

## Problems Identified

### 1. Missing Protocol Types
NumPy typing defines protocols for duck typing:
```python
class _SupportsArray(Protocol):
    def __array__(self) -> np.ndarray: ...

class _SupportsDType(Protocol[_DType]):
    @property
    def dtype(self) -> _DType: ...
    
class _NestedSequence(Protocol[_T_co]):
    def __getitem__(self, index: int) -> _T_co | _NestedSequence[_T_co]: ...
    def __len__(self) -> int: ...
```

rust-numpy missing:
- `SupportsArray` protocol
- `SupportsDType` protocol
- `NestedSequence` protocol
- Protocol-based `ArrayLike` (current uses trait, which is different)

### 2. Missing Concrete Types
```python
# These are in NumPy typing but missing in rust-numpy:
- _Scalar        # Generic scalar type
- _Shape         # Shape tuple
- _CharCode      # Dtype character codes
- _GenericAlias  # For generic type aliases
```

### 3. Incomplete ArrayLike
NumPy's `ArrayLike` includes:
- `_SupportsArray`
- `_NestedSequence[_SupportsArray]`
- `_Scalar` (for 0-d arrays)
- `bool`, `int`, `float`, `complex`, `str`, `bytes`

Current rust-numpy only implements for `Array`, `Vec`, `[T; N]`, `&[T]`.

### 4. Missing StringDType (NumPy 2.0)
New in NumPy 2.0, StringDType for variable-length strings.

### 5. Incomplete NDArray Type
NumPy's `NDArray` is more sophisticated:
```python
NDArray = np.ndarray[Any, np.dtype[_Scalar]]
```

Current is just `type NDArray<T> = Array<T>`.

### 6. Missing mypy_plugin Equivalent
No support for Rust type system to understand NumPy's complex typing.

## Required Changes

### 1. Add Protocol Traits
```rust
// src/typing/protocols.rs
pub trait SupportsArray {
    type Element;
    fn __array__(&self) -> Array<Self::Element>;
}

pub trait SupportsDType {
    fn dtype(&self) -> Dtype;
}

pub trait NestedSequence<T> {
    fn get(&self, index: usize) -> Option<Either<T, &dyn NestedSequence<T>>>;
    fn len(&self) -> usize;
}
```

### 2. Expand ArrayLike
```rust
// ArrayLike should accept:
// - Array<T>
// - Vec<T>
// - [T; N]
// - &[T]
// - Nested sequences of the above
// - Scalar values (for 0-d)
// - Any type implementing SupportsArray

pub enum ArrayLike<'a, T: Clone + Default + 'static> {
    Array(Array<T>),
    Vec(Vec<T>),
    Slice(&'a [T]),
    Scalar(T),
    Nested(Box<dyn NestedSequence<T>>),
    SupportsArray(Box<dyn SupportsArray<Element = T>>),
}
```

### 3. Add Missing Types
```rust
// src/typing/mod.rs
pub type _Shape = Vec<usize>;  // Or tuple if we want exact shape typing
pub type _ShapeLike = Either<usize, Vec<usize>>;

pub trait Scalar: Clone + 'static {}
impl Scalar for i8 {}
impl Scalar for i16 {}
// ... etc

pub trait CharCode {
    fn char_code(&self) -> char;
}
```

### 4. NDArray with dtype parameter
```rust
// More sophisticated NDArray type
pub struct NDArray<T: Scalar, D: Dtype = Dtype> {
    // phantom data for type system
    _phantom: PhantomData<(T, D)>,
}

// Type aliases for common cases
pub type Float64Array = NDArray<f64>;
pub type Int32Array = NDArray<i32>;
// etc
```

### 5. Module Structure
```rust
// src/typing/mod.rs structure
pub mod protocols;     // SupportsArray, SupportsDType, NestedSequence
pub mod aliases;       // ArrayLike, DTypeLike, NDArray, etc.
pub mod bitwidth;      // NBitBase, nbit_8, etc. (existing)
pub mod dtype_getter;  // DtypeGetter trait (existing)
pub mod scalars;       // Scalar types and protocols
pub mod shape;         // Shape types

// Re-exports
pub use aliases::*;
pub use protocols::*;
pub use scalars::*;
pub use shape::*;
```

### 6. Update Prelude
```rust
// src/typing/prelude.rs
pub use super::{
    // Array types
    NDArray, ArrayLike, BoolArray, Complex128Array, Complex64Array,
    Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    
    // Traits
    ArrayLike as ArrayLikeTrait,  // if different from type
    DtypeLike,
    SupportsArray,
    SupportsDType,
    
    // DType
    DtypeLike as DTypeLikeType,
    
    // Bit-width
    NBitBase, nbit_128, nbit_16, nbit_256, nbit_32, nbit_64, nbit_8,
};
```

## Acceptance Criteria
- [ ] `SupportsArray` protocol trait implemented
- [ ] `SupportsDType` protocol trait implemented  
- [ ] `NestedSequence` protocol trait implemented
- [ ] `ArrayLike` accepts all NumPy-compatible types
- [ ] `DTypeLike` accepts all dtype-like objects
- [ ] `NDArray` has dtype parameter
- [ ] Missing scalar, shape types added
- [ ] All types exported at `typing` module level
- [ ] Prelude includes commonly used types

## Affected Files
- `rust-numpy/src/typing/mod.rs` - Restructure
- `rust-numpy/src/typing/protocols.rs` - NEW
- `rust-numpy/src/typing/aliases.rs` - NEW or modify existing
- `rust-numpy/src/typing/scalars.rs` - NEW
- `rust-numpy/src/typing/shape.rs` - NEW
- `rust-numpy/src/typing/prelude.rs` - Update
- `rust-numpy/src/lib.rs` - Update re-exports


---

## Sub-Issues (for parallel development)

| Issue | Title | Scope |
|-------|-------|-------|
| #597 | Create Protocol Traits | SupportsArray, SupportsDType, NestedSequence |
| #596 | Create Enhanced NDArray | Shape and dtype parameters |

**Coordination Notes:**
- #597 creates protocol traits for duck typing
- #596 creates enhanced NDArray type
- May have some overlap in ArrayLike definition

