# [Sub-Issue] Create Concrete Type Aliases (#571)

**Issue #585**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/585  
**Created:** 2026-01-29T03:05:16Z  
**Updated:** 2026-01-29T03:23:59Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:dtype, ralph/numpy:impl  

---

## Parent Issue
Part of #571 - dtype type system and exports don't match NumPy

## Scope
Create Concrete Type Aliases (int8, float32, complex64, etc.)

## Background
NumPy exposes concrete types at the root level:
```python
np.int8, np.int16, np.int32, np.int64
np.float32, np.float64
np.complex64, np.complex128
np.bool_, np.str_, np.bytes_
```

Current rust-numpy has placeholder FUNCTIONS instead of types.

## Tasks

### 1. Create Type Aliases

```rust
// In src/types.rs or src/dtype.rs

// Sized integers
pub type int8 = i8;
pub type int16 = i16;
pub type int32 = i32;
pub type int64 = i64;
pub type uint8 = u8;
pub type uint16 = u16;
pub type uint32 = u32;
pub type uint64 = u64;

// Platform-dependent integers
pub type intp = isize;  // For indexing
pub type uintp = usize;  // For indexing
pub type int_ = i64;  // Default integer (like Python int)
pub type uint = u64;
pub type intc = i32;  // C int
pub type uintc = u32;  // C unsigned int
pub type long = i64;  // C long
pub type ulong = u64;  // C unsigned long
pub type longlong = i64;  // C long long
pub type ulonglong = u64;  // C unsigned long long
pub type short = i16;  // C short
pub type ushort = u16;  // C unsigned short
pub type byte = i8;  // C char
pub type ubyte = u8;  // C unsigned char

// Floats
pub type float16 = f16;  // Need half crate or custom implementation
pub type float32 = f32;
pub type float64 = f64;
pub type float128 = f128;  // If platform supports

// Complex
pub type complex64 = Complex<f32>;
pub type complex128 = Complex<f64>;
pub type complex256 = Complex<f128>;  // If platform supports

// Other
pub type bool_ = bool;
pub type str_ = String;
pub type bytes_ = Vec<u8>;
pub type void = ();
```

### 2. Remove Placeholder Functions

In `src/utils.rs`, REMOVE or DEPRECATE:
```rust
// REMOVE these placeholder functions:
pub fn int8() -> i8 { 0 }
pub fn int16() -> i16 { 0 }
// ... etc

// REPLACE with proper type aliases exported from types module
```

### 3. Export at Root

```rust
// In src/lib.rs
pub use types::{
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    intp, uintp, int_, uint, intc, uintc, long, ulong, longlong, ulonglong,
    short, ushort, byte, ubyte,
    float16, float32, float64, float128,
    complex64, complex128, complex256,
    bool_, str_, bytes_, void,
};
```

### 4. Update Dtype Enum

Ensure `Dtype` enum has variants matching these types:
```rust
pub enum Dtype {
    Int8 { ... },
    Int16 { ... },
    // ... etc
}
```

## Files to Modify
- `src/types.rs` - NEW file with type aliases
- `src/utils.rs` - REMOVE placeholder functions
- `src/lib.rs` - Export types
- `src/dtype.rs` - Ensure Dtype enum matches

## Acceptance Criteria
- [ ] All type aliases created
- [ ] Placeholder functions removed
- [ ] Types exported at crate root
- [ ] Can use as `numpy::int32` not `numpy::int32()`

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #563 (Add top-level dtype constructors)
- This issue: TYPE ALIASES
- #563: May have overlap - coordinate to avoid duplicate work

