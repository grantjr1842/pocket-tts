# [Sub-Issue] Create Abstract Base Type Traits (#571)

**Issue #586**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/586  
**Created:** 2026-01-29T03:05:16Z  
**Updated:** 2026-01-29T03:24:01Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:dtype, ralph/numpy:impl  

---

## Parent Issue
Part of #571 - dtype type system and exports don't match NumPy

## Scope
Create Abstract Base Type Traits (generic, number, integer, floating, etc.)

## Background
NumPy has a type hierarchy with abstract base types:
```
generic
├── number
│   ├── integer
│   │   ├── signedinteger
│   │   └── unsignedinteger
│   ├── floating
│   └── complexfloating
├── flexible
│   └── character
└── bool_
```

Rust needs equivalent traits for type classification.

## Tasks

### 1. Create Abstract Base Traits in `src/dtype.rs` or `src/types.rs`

```rust
/// Base trait for all NumPy scalar types
pub trait Generic: 'static + Clone + Debug {}

/// Base trait for numeric types
pub trait Number: Generic {}

/// Base trait for integer types
pub trait Integer: Number {}

/// Base trait for signed integers
pub trait SignedInteger: Integer {}

/// Base trait for unsigned integers  
pub trait UnsignedInteger: Integer {}

/// Base trait for inexact types (float/complex)
pub trait Inexact: Number {}

/// Base trait for floating point types
pub trait Floating: Inexact {}

/// Base trait for complex floating types
pub trait ComplexFloating: Inexact {}

/// Base trait for flexible types (strings, void)
pub trait Flexible: Generic {}

/// Base trait for character types
pub trait Character: Flexible {}

/// Boolean type
pub trait Bool_: Generic {}
```

### 2. Implement Traits for Primitive Types

```rust
// Integers
impl Generic for i8 {}
impl Generic for i16 {}
impl Generic for i32 {}
impl Generic for i64 {}
// ... etc

impl Number for i8 {}
impl Number for i16 {}
// ... etc

impl Integer for i8 {}
impl Integer for i16 {}
// ... etc

impl SignedInteger for i8 {}
impl SignedInteger for i16 {}
// ... etc

impl UnsignedInteger for u8 {}
impl UnsignedInteger for u16 {}
// ... etc

// Floats
impl Generic for f32 {}
impl Generic for f64 {}
impl Number for f32 {}
impl Number for f64 {}
impl Inexact for f32 {}
impl Inexact for f64 {}
impl Floating for f32 {}
impl Floating for f64 {}

// Complex (using num_complex)
impl Generic for Complex<f32> {}
impl Generic for Complex<f64> {}
impl Number for Complex<f32> {}
impl Number for Complex<f64> {}
impl Inexact for Complex<f32> {}
impl Inexact for Complex<f64> {}
impl ComplexFloating for Complex<f32> {}
impl ComplexFloating for Complex<f64> {}

// Bool
impl Generic for bool {}
impl Bool_ for bool {}

// Flexible types
impl Generic for String {}
impl Generic for Vec<u8> {}
impl Flexible for String {}
impl Flexible for Vec<u8> {}
impl Character for String {}
impl Character for Vec<u8> {}
```

### 3. Add Type Checking Functions

```rust
/// Check if type is a subdtype of another
pub fn issubdtype<T: Generic, U: Generic>() -> bool {
    // Use trait bounds to determine hierarchy
}

/// Check if type is a number
pub fn isnumber<T: Number>() -> bool { true }

/// Check if type is integer
pub fn isinteger<T: Integer>() -> bool { true }

// ... etc
```

### 4. Export at Root

```rust
// In src/lib.rs
pub use dtype::{
    Generic, Number, Integer, SignedInteger, UnsignedInteger,
    Inexact, Floating, ComplexFloating, Flexible, Character, Bool_,
    issubdtype, isnumber, isinteger, issignedinteger, isunsignedinteger,
    isfloating, iscomplexfloating,
};
```

## Files to Modify
- `src/dtype.rs` - Add traits and implementations
- `src/lib.rs` - Export traits and functions
- NEW: `src/types.rs` - Alternative: put all type traits here

## Acceptance Criteria
- [ ] All abstract base traits defined
- [ ] Traits implemented for all primitive types
- [ ] Type hierarchy works correctly
- [ ] Type checking functions work
- [ ] All exported at crate root

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #563 (Add top-level dtype constructors)
- This issue focuses on TRAITS/HIERARCHY
- #563 focuses on CONCRETE TYPE CONSTRUCTORS

