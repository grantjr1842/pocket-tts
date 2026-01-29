# rust-numpy: dtype type system and exports don't match NumPy

**Issue #571**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/571  
**Created:** 2026-01-29T02:47:06Z  
**Updated:** 2026-01-29T03:23:57Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:dtype, ralph/numpy:audit  

---

## Summary
NumPy's dtype system has a very specific export structure that rust-numpy does not match. Type constructors, dtype classes, and type-checking functions must be available at both the root level and in the `numpy.core.numerictypes` module.

## NumPy dtype Reference

### NumPy Type Hierarchy (from `numpy.core.numerictypes`)
```python
# Abstract base types (at numpy root and numpy.core.numerictypes)
numpy.generic          # Base class for all numpy scalar types
numpy.number           # Base for numeric types
numpy.integer          # Base for integer types
numpy.signedinteger    # Base for signed integers
numpy.unsignedinteger  # Base for unsigned integers
numpy.inexact          # Base for floating/complex
numpy.floating         # Base for float types
numpy.complexfloating  # Base for complex types
numpy.flexible         # Base for flexible types (strings, void)
numpy.character        # Base for character types

# Concrete type constructors (at numpy root)
numpy.bool_            # Boolean type
numpy.byte             # Platform char
numpy.ubyte            # Platform unsigned char
numpy.short            # Platform short
numpy.ushort           # Platform unsigned short
numpy.intc             # Platform int
numpy.uintc            # Platform unsigned int
numpy.int_             # Platform long (Python int)
numpy.uint             # Platform unsigned long
numpy.longlong         # Platform long long
numpy.ulonglong        # Platform unsigned long long
numpy.single           # Platform float
numpy.double           # Platform double (Python float)
numpy.longdouble       # Platform long double
numpy.csingle          # Complex64
numpy.cdouble          # Complex128
numpy.clongdouble      # Complex long double

# Sized types (at numpy root)
numpy.int8, numpy.int16, numpy.int32, numpy.int64
numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64
numpy.float16, numpy.float32, numpy.float64, numpy.float128
numpy.complex64, numpy.complex128, numpy.complex256

# Character code types (at numpy root)
numpy.character        # String type base
numpy.bytes_           # Byte string (S)
numpy.str_             # Unicode string (U)
numpy.void             # Void type (V)
numpy.object_          # Python object (O)
numpy.datetime64       # Datetime (M)
numpy.timedelta64      # Timedelta (m)
```

### Type-checking functions (at numpy root)
```python
numpy.issubdtype(arg1, arg2)   # Check if arg1 is a subdtype of arg2
numpy.issubsctype(arg1, arg2)  # Check if arg1 is a subdtype of arg2 (same as issubdtype)
numpy.issubclass_(arg1, arg2)  # issubclass that works with dtypes
numpy.find_common_type(array_types, scalar_types)  # Find common type
numpy.mintypecode(typechars, typeset='?'), default='d'  # Minimum type code
numpy.sctype2char(sctype)      # Convert type to character code
numpy.chararray.sctype2char    # Same as above
```

## Current rust-numpy State

### In `src/utils.rs` (PLACEHOLDERS - NOT REAL TYPES)
```rust
pub fn bool() -> bool { true }
pub fn bool_() -> bool { true }
pub fn byte() -> u8 { 0 }
pub fn bytes_() -> Vec<u8> { vec![] }
pub fn double() -> f64 { 0.0 }
pub fn single() -> f32 { 0.0 }
pub fn half() -> f16 { f16::from_f32(0.0) }
// ... etc, all return values, not type constructors
```

### In `src/dtype.rs`
```rust
pub enum Dtype {
    Int8 { byteorder: Option<ByteOrder> },
    Int16 { byteorder: Option<ByteOrder> },
    // ... etc
}
```

### In `src/lib.rs`
```rust
pub use dtype::{Casting, Dtype, DtypeKind};  // Limited
pub use utils::{  // These are WRONG - they're functions returning values
    bool, bool_, byte, bytes_, ..., int8, int16, int32, int64,
    uint8, uint16, uint32, uint64, float32, float64, complex64, complex128,
    // ... etc
};
```

## Problems Identified

### 1. Type Constructors Should Be Types, Not Functions
**Current:**
```rust
pub fn int32() -> i32 { 0 }  // WRONG: Returns a value, not a type
```

**Expected:**
```rust
pub type int32 = i32;         // CORRECT: Type alias
// OR
pub struct int32(i32);        // CORRECT: Newtype wrapper for numpy compat
```

### 2. Abstract Base Types Are Missing
NumPy allows:
```python
np.issubdtype(np.int32, np.integer)  # True
np.issubdtype(np.float64, np.number) # True
```

Current rust-numpy has no equivalents for:
- `generic`, `number`, `integer`, `signedinteger`, `unsignedinteger`
- `inexact`, `floating`, `complexfloating`
- `flexible`, `character`

### 3. Character Type Codes
NumPy uses single-character codes:
- `'?'` = bool, `'b'` = int8, `'B'` = uint8, `'i'` = int32, `'u'` = uint32
- `'f'` = float32, `'d'` = float64, `'c'` = complex64, `'D'` = complex128
- `'S'` = bytes, `'U'` = str_, `'V'` = void, `'O'` = object, `'M'` = datetime64, `'m'` = timedelta64

These are exposed in NumPy as `np.dtype('i')`, etc.

### 4. dtype Function Missing
NumPy's `np.dtype()` function creates dtype objects:
```python
np.dtype(np.int32)
np.dtype('i4')
np.dtype([('x', 'f4'), ('y', 'f4')])
```

## Required Changes

### 1. Create Type Aliases/Structs in `src/dtype.rs` or `src/types.rs`
```rust
// Abstract base types (marker traits)
pub trait Generic: 'static {}
pub trait Number: Generic {}
pub trait Integer: Number {}
pub trait SignedInteger: Integer {}
pub trait UnsignedInteger: Integer {}
pub trait Inexact: Number {}
pub trait Floating: Inexact {}
pub trait ComplexFloating: Inexact {}

// Implement for primitive types
impl Generic for i8 {}
impl Generic for i16 {}
// ... etc
impl Integer for i8 {}
impl Integer for i16 {}
// ... etc
impl SignedInteger for i8 {}
// ... etc

// Concrete type aliases
pub type bool_ = bool;
pub type byte = i8;      // or platform-dependent
pub type ubyte = u8;
pub type short = i16;    // or platform-dependent
pub type ushort = u16;
pub type intc = i32;     // or platform-dependent
pub type uintc = u32;
pub type int_ = i64;     // Python int = platform long
pub type uint = u64;
pub type longlong = i64;
pub type ulonglong = u64;
pub type single = f32;
pub type double = f64;
pub type longdouble = f64; // or f128 if available
pub type csingle = Complex<f32>;
pub type cdouble = Complex<f64>;
pub type clongdouble = Complex<f64>; // or Complex<f128>

// Sized types
pub type int8 = i8;
pub type int16 = i16;
pub type int32 = i32;
pub type int64 = i64;
pub type uint8 = u8;
pub type uint16 = u16;
pub type uint32 = u32;
pub type uint64 = u64;
pub type float16 = f16;  // from half crate
pub type float32 = f32;
pub type float64 = f64;
pub type float128 = f128; // if available
pub type complex64 = Complex<f32>;
pub type complex128 = Complex<f64>;
pub type complex256 = Complex<f128>; // if available

// Character types
pub type character = u8;  // Base for char types
pub type bytes_ = Vec<u8>;
pub type str_ = String;
pub type void = ();
pub type object_ = PyObject; // or std::any::Any equivalent
```

### 2. Create dtype Constructor Function
```rust
pub fn dtype<T: Into<Dtype>>(obj: T) -> Dtype {
    obj.into()
}

// Implement From for various types
impl From<&str> for Dtype { /* parse string codes */ }
impl From<type> for Dtype { /* for each numpy type */ }
```

### 3. Create Type-checking Functions
```rust
pub fn issubdtype(arg1: &Dtype, arg2: &Dtype) -> bool;
pub fn issubsctype(arg1: &Dtype, arg2: &Dtype) -> bool;
pub fn issubclass_<T: Generic, U: Generic>() -> bool;
pub fn find_common_type(types: &[Dtype]) -> Dtype;
pub fn mintypecode(typechars: &[char]) -> char;
```

### 4. Export at Root Level
```rust
// In src/lib.rs
pub use dtype::{
    bool_, byte, ubyte, short, ushort, intc, uintc, int_, uint,
    longlong, ulonglong, single, double, longdouble,
    csingle, cdouble, clongdouble,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64, float128,
    complex64, complex128, complex256,
    bytes_, str_, void, object_,
    character, generic, number, integer, signedinteger, unsignedinteger,
    inexact, floating, complexfloating, flexible,
    dtype, issubdtype, issubsctype, find_common_type, mintypecode,
};
```

## Acceptance Criteria
- [ ] All NumPy abstract base types available as traits or markers
- [ ] All NumPy concrete types available as type aliases
- [ ] `dtype()` constructor function works like NumPy's
- [ ] Type-checking functions `issubdtype`, `issubsctype` work correctly
- [ ] All types exported at crate root
- [ ] Types have correct hierarchy (e.g., `int32` is `Integer`, `float64` is `Floating`)
- [ ] Character codes work: `dtype('i4')`, `dtype('f8')`, etc.

## Affected Files
- `rust-numpy/src/dtype.rs` - Add types and traits
- `rust-numpy/src/lib.rs` - Add exports
- `rust-numpy/src/utils.rs` - Remove placeholder functions
- May need: `rust-numpy/src/types.rs` - New file for type system


---

## Sub-Issues (for parallel development)

| Issue | Title | Scope |
|-------|-------|-------|
| #586 | Create Abstract Base Type Traits | generic, number, integer, floating traits |
| #585 | Create Concrete Type Aliases | int8, float32, complex64, etc. |

**Coordination Notes:**
- #586 focuses on TRAIT HIERARCHY (abstract base types)
- #585 focuses on TYPE ALIASES (concrete types)
- Coordinate to ensure consistency

