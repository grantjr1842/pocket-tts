//! Dtype constructors matching NumPy's type system
//!
//! This module provides constructor functions for all NumPy dtype types.
//! These functions return `Dtype` enum variants that can be used for
//! type checking, array creation, and dtype operations.

use crate::dtype::{ByteOrder, DatetimeUnit, Dtype, DtypeKind, TimedeltaUnit};

// ============================================================================
// Boolean Type Constructors
// ============================================================================

/// Boolean type constructor
///
/// Returns the boolean dtype. Equivalent to `numpy.bool_` or `numpy.bool`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::bool_;
/// let dt = bool_();
/// assert_eq!(dt.to_string(), "bool");
/// ```
pub fn bool_() -> Dtype {
    Dtype::Bool
}

// ============================================================================
// Signed Integer Type Constructors
// ============================================================================

/// 8-bit signed integer type constructor
///
/// Returns the int8 dtype. Equivalent to `numpy.int8`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::int8;
/// let dt = int8();
/// assert_eq!(dt.to_string(), "int8");
/// ```
pub fn int8() -> Dtype {
    Dtype::Int8 { byteorder: None }
}

/// 16-bit signed integer type constructor
///
/// Returns the int16 dtype. Equivalent to `numpy.int16`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::int16;
/// let dt = int16();
/// assert_eq!(dt.to_string(), "int16");
/// ```
pub fn int16() -> Dtype {
    Dtype::Int16 { byteorder: None }
}

/// 32-bit signed integer type constructor
///
/// Returns the int32 dtype. Equivalent to `numpy.int32`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::int32;
/// let dt = int32();
/// assert_eq!(dt.to_string(), "int32");
/// ```
pub fn int32() -> Dtype {
    Dtype::Int32 { byteorder: None }
}

/// 64-bit signed integer type constructor
///
/// Returns the int64 dtype. Equivalent to `numpy.int64`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::int64;
/// let dt = int64();
/// assert_eq!(dt.to_string(), "int64");
/// ```
pub fn int64() -> Dtype {
    Dtype::Int64 { byteorder: None }
}

/// Platform-dependent signed integer type constructor
///
/// Returns the intp dtype (i32 on 32-bit, i64 on 64-bit).
/// Equivalent to `numpy.intp`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::intp;
/// let dt = intp();
/// assert_eq!(dt.to_string(), "intp");
/// ```
pub fn intp() -> Dtype {
    Dtype::Intp { byteorder: None }
}

/// Platform-dependent signed integer type constructor (alias)
///
/// Returns the intp dtype. Alias for `intp()`.
/// Equivalent to `numpy.int_`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::int_;
/// let dt = int_();
/// assert_eq!(dt.to_string(), "intp");
/// ```
pub fn int_() -> Dtype {
    intp()
}

/// C int type constructor (alias)
///
/// Returns the int32 dtype. Alias for `int32()`.
/// Equivalent to `numpy.intc`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::intc;
/// let dt = intc();
/// assert_eq!(dt.to_string(), "int32");
/// ```
pub fn intc() -> Dtype {
    int32()
}

/// C short type constructor
///
/// Returns the int16 dtype on most platforms.
/// Equivalent to `numpy.short`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::short;
/// let dt = short();
/// assert_eq!(dt.to_string(), "int16");
/// ```
pub fn short() -> Dtype {
    Dtype::Int16 { byteorder: None }
}

/// C long type constructor
///
/// Returns the int64 dtype on 64-bit platforms, int32 on 32-bit.
/// Equivalent to `numpy.long`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::long;
/// let dt = long();
/// // Returns int64 on 64-bit, int32 on 32-bit
/// ```
pub fn long() -> Dtype {
    Dtype::Int64 { byteorder: None }
}

/// C long long type constructor
///
/// Returns the int64 dtype.
/// Equivalent to `numpy.longlong`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::longlong;
/// let dt = longlong();
/// assert_eq!(dt.to_string(), "int64");
/// ```
pub fn longlong() -> Dtype {
    Dtype::Int64 { byteorder: None }
}

// ============================================================================
// Unsigned Integer Type Constructors
// ============================================================================

/// 8-bit unsigned integer type constructor
///
/// Returns the uint8 dtype. Equivalent to `numpy.uint8`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::uint8;
/// let dt = uint8();
/// assert_eq!(dt.to_string(), "uint8");
/// ```
pub fn uint8() -> Dtype {
    Dtype::UInt8 { byteorder: None }
}

/// 16-bit unsigned integer type constructor
///
/// Returns the uint16 dtype. Equivalent to `numpy.uint16`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::uint16;
/// let dt = uint16();
/// assert_eq!(dt.to_string(), "uint16");
/// ```
pub fn uint16() -> Dtype {
    Dtype::UInt16 { byteorder: None }
}

/// 32-bit unsigned integer type constructor
///
/// Returns the uint32 dtype. Equivalent to `numpy.uint32`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::uint32;
/// let dt = uint32();
/// assert_eq!(dt.to_string(), "uint32");
/// ```
pub fn uint32() -> Dtype {
    Dtype::UInt32 { byteorder: None }
}

/// 64-bit unsigned integer type constructor
///
/// Returns the uint64 dtype. Equivalent to `numpy.uint64`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::uint64;
/// let dt = uint64();
/// assert_eq!(dt.to_string(), "uint64");
/// ```
pub fn uint64() -> Dtype {
    Dtype::UInt64 { byteorder: None }
}

/// Platform-dependent unsigned integer type constructor
///
/// Returns the uintp dtype (u32 on 32-bit, u64 on 64-bit).
/// Equivalent to `numpy.uintp`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::uintp;
/// let dt = uintp();
/// assert_eq!(dt.to_string(), "uintp");
/// ```
pub fn uintp() -> Dtype {
    Dtype::Uintp { byteorder: None }
}

/// Platform-dependent unsigned integer type constructor (alias)
///
/// Returns the uintp dtype. Alias for `uintp()`.
/// Equivalent to `numpy.uint`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::uint;
/// let dt = uint();
/// assert_eq!(dt.to_string(), "uintp");
/// ```
pub fn uint() -> Dtype {
    uintp()
}

/// C unsigned int type constructor (alias)
///
/// Returns the uint32 dtype. Alias for `uint32()`.
/// Equivalent to `numpy.uintc`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::uintc;
/// let dt = uintc();
/// assert_eq!(dt.to_string(), "uint32");
/// ```
pub fn uintc() -> Dtype {
    uint32()
}

/// C unsigned short type constructor
///
/// Returns the uint16 dtype on most platforms.
/// Equivalent to `numpy.ushort`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::ushort;
/// let dt = ushort();
/// assert_eq!(dt.to_string(), "uint16");
/// ```
pub fn ushort() -> Dtype {
    Dtype::UInt16 { byteorder: None }
}

/// C unsigned long type constructor
///
/// Returns the uint64 dtype on 64-bit platforms, uint32 on 32-bit.
/// Equivalent to `numpy.ulong`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::ulong;
/// let dt = ulong();
/// // Returns uint64 on 64-bit, uint32 on 32-bit
/// ```
pub fn ulong() -> Dtype {
    Dtype::UInt64 { byteorder: None }
}

/// C unsigned long long type constructor
///
/// Returns the uint64 dtype.
/// Equivalent to `numpy.ulonglong`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::ulonglong;
/// let dt = ulonglong();
/// assert_eq!(dt.to_string(), "uint64");
/// ```
pub fn ulonglong() -> Dtype {
    Dtype::UInt64 { byteorder: None }
}

// ============================================================================
// Floating Point Type Constructors
// ============================================================================

/// Half-precision floating point type constructor
///
/// Returns the float16 dtype. Equivalent to `numpy.float16`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::float16;
/// let dt = float16();
/// assert_eq!(dt.to_string(), "float16");
/// ```
pub fn float16() -> Dtype {
    Dtype::Float16 { byteorder: None }
}

/// Single-precision floating point type constructor
///
/// Returns the float32 dtype. Equivalent to `numpy.float32`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::float32;
/// let dt = float32();
/// assert_eq!(dt.to_string(), "float32");
/// ```
pub fn float32() -> Dtype {
    Dtype::Float32 { byteorder: None }
}

/// Double-precision floating point type constructor
///
/// Returns the float64 dtype. Equivalent to `numpy.float64`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::float64;
/// let dt = float64();
/// assert_eq!(dt.to_string(), "float64");
/// ```
pub fn float64() -> Dtype {
    Dtype::Float64 { byteorder: None }
}

/// Extended-precision floating point type constructor
///
/// Returns the float128 dtype (platform-dependent).
/// Equivalent to `numpy.float128`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::float128;
/// let dt = float128();
/// assert_eq!(dt.to_string(), "float128");
/// ```
pub fn float128() -> Dtype {
    Dtype::Float128 { byteorder: None }
}

// ============================================================================
// Complex Type Constructors
// ============================================================================

/// 32-bit complex number type constructor (2x float16)
///
/// Returns the complex64 dtype. Equivalent to `numpy.complex64`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::complex64;
/// let dt = complex64();
/// assert_eq!(dt.to_string(), "complex64");
/// ```
pub fn complex64() -> Dtype {
    Dtype::Complex32 { byteorder: None }
}

/// 64-bit complex number type constructor (2x float32)
///
/// Returns the complex128 dtype. Equivalent to `numpy.complex128`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::complex128;
/// let dt = complex128();
/// assert_eq!(dt.to_string(), "complex128");
/// ```
pub fn complex128() -> Dtype {
    Dtype::Complex64 { byteorder: None }
}

/// 128-bit complex number type constructor (2x float64)
///
/// Returns the complex256 dtype (platform-dependent).
/// Equivalent to `numpy.complex256`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::complex256;
/// let dt = complex256();
/// assert_eq!(dt.to_string(), "complex128");
/// ```
pub fn complex256() -> Dtype {
    Dtype::Complex128 { byteorder: None }
}

/// C single-precision complex type constructor
///
/// Returns the complex64 dtype. Equivalent to `numpy.csingle`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::csingle;
/// let dt = csingle();
/// assert_eq!(dt.to_string(), "complex64");
/// ```
pub fn csingle() -> Dtype {
    Dtype::Complex32 { byteorder: None }
}

/// C double-precision complex type constructor
///
/// Returns the complex128 dtype. Equivalent to `numpy.cdouble`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::cdouble;
/// let dt = cdouble();
/// assert_eq!(dt.to_string(), "complex128");
/// ```
pub fn cdouble() -> Dtype {
    Dtype::Complex64 { byteorder: None }
}

/// C extended-precision complex type constructor
///
/// Returns the complex256 dtype (platform-dependent).
/// Equivalent to `numpy.clongdouble`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::clongdouble;
/// let dt = clongdouble();
/// assert_eq!(dt.to_string(), "complex128");
/// ```
pub fn clongdouble() -> Dtype {
    Dtype::Complex128 { byteorder: None }
}

// ============================================================================
// Datetime and Timedelta Type Constructors
// ============================================================================

/// Datetime64 type constructor
///
/// Returns the datetime64 dtype with the specified unit.
/// Equivalent to `numpy.datetime64`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::datetime64;
/// use crate::dtype::DatetimeUnit;
/// let dt = datetime64(DatetimeUnit::us);
/// ```
pub fn datetime64(unit: DatetimeUnit) -> Dtype {
    Dtype::Datetime64(unit)
}

/// Timedelta64 type constructor
///
/// Returns the timedelta64 dtype with the specified unit.
/// Equivalent to `numpy.timedelta64`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::timedelta64;
/// use crate::dtype::TimedeltaUnit;
/// let dt = timedelta64(TimedeltaUnit::s);
/// ```
pub fn timedelta64(unit: TimedeltaUnit) -> Dtype {
    Dtype::Timedelta64(unit)
}

// ============================================================================
// String and Bytes Type Constructors
// ============================================================================

/// Byte string type constructor
///
/// Returns the bytes dtype with optional length.
/// Equivalent to `numpy.bytes_`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::bytes_;
/// let dt = bytes_(None);
/// ```
pub fn bytes_(length: Option<usize>) -> Dtype {
    Dtype::Bytes { length: length.unwrap_or(1) }
}

// ============================================================================
// Generic Type Constructors (Base Classes)
// ============================================================================

/// Base class for generic types
///
/// Returns a placeholder for the generic base class.
/// Equivalent to `numpy.generic`.
///
/// # Note
/// This is a type marker for isinstance checks. In practice,
/// you would check if a dtype's kind matches expected kinds.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::generic;
/// let dt = generic();
/// // Use for type checking
/// ```
pub fn generic() -> Dtype {
    // Generic is a base class marker, return a representative dtype
    // In practice, use kind() for type checking
    Dtype::Object
}

/// Base class for numeric types
///
/// Returns a placeholder for the number base class.
/// Equivalent to `numpy.number`.
///
/// # Note
/// This is a type marker for isinstance checks. Use `DtypeKind` for
/// type checking instead.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::number;
/// let dt = number();
/// // Use for type checking
/// ```
pub fn number() -> Dtype {
    // Number is a base class marker, return a representative dtype
    // In practice, use kind() for type checking
    Dtype::Float64 { byteorder: None }
}

/// Base class for integer types
///
/// Returns a placeholder for the integer base class.
/// Equivalent to `numpy.integer`.
///
/// # Note
/// This is a type marker for isinstance checks. Use `DtypeKind::Integer`
/// for type checking instead.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::integer;
/// let dt = integer();
/// // Use for type checking
/// ```
pub fn integer() -> Dtype {
    // Integer is a base class marker, return a representative dtype
    // In practice, use kind() for type checking
    Dtype::Int64 { byteorder: None }
}

/// Base class for signed integer types
///
/// Returns a placeholder for the signed integer base class.
/// Equivalent to `numpy.signedinteger`.
///
/// # Note
/// This is a type marker for isinstance checks. Use `DtypeKind::Integer`
/// for type checking instead.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::signedinteger;
/// let dt = signedinteger();
/// // Use for type checking
/// ```
pub fn signedinteger() -> Dtype {
    // SignedInteger is a base class marker, return a representative dtype
    // In practice, use kind() for type checking
    Dtype::Int64 { byteorder: None }
}

/// Base class for unsigned integer types
///
/// Returns a placeholder for the unsigned integer base class.
/// Equivalent to `numpy.unsignedinteger`.
///
/// # Note
/// This is a type marker for isinstance checks. Use `DtypeKind::Unsigned`
/// for type checking instead.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::unsignedinteger;
/// let dt = unsignedinteger();
/// // Use for type checking
/// ```
pub fn unsignedinteger() -> Dtype {
    // UnsignedInteger is a base class marker, return a representative dtype
    // In practice, use kind() for type checking
    Dtype::UInt64 { byteorder: None }
}

/// Base class for floating point types
///
/// Returns a placeholder for the floating base class.
/// Equivalent to `numpy.floating`.
///
/// # Note
/// This is a type marker for isinstance checks. Use `DtypeKind::Float`
/// for type checking instead.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::floating;
/// let dt = floating();
/// // Use for type checking
/// ```
pub fn floating() -> Dtype {
    // Floating is a base class marker, return a representative dtype
    // In practice, use kind() for type checking
    Dtype::Float64 { byteorder: None }
}

/// Base class for complex types
///
/// Returns a placeholder for the complex floating base class.
/// Equivalent to `numpy.complexfloating`.
///
/// # Note
/// This is a type marker for isinstance checks. Use `DtypeKind::Complex`
/// for type checking instead.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::complexfloating;
/// let dt = complexfloating();
/// // Use for type checking
/// ```
pub fn complexfloating() -> Dtype {
    // ComplexFloating is a base class marker, return a representative dtype
    // In practice, use kind() for type checking
    Dtype::Complex128 { byteorder: None }
}

// ============================================================================
// Other Type Constructors
// ============================================================================

/// Unicode string type constructor
///
/// Returns the unicode string dtype. Equivalent to `numpy.str_`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::str_;
/// let dt = str_();
/// assert_eq!(dt.to_string(), "unicode");
/// ```
pub fn str_() -> Dtype {
    Dtype::Unicode { length: None }
}

/// Object type constructor
///
/// Returns the object dtype. Equivalent to `numpy.object_`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::object_;
/// let dt = object_();
/// assert_eq!(dt.to_string(), "object");
/// ```
pub fn object_() -> Dtype {
    Dtype::Object
}

/// Void type constructor
///
/// Returns the void dtype. Equivalent to `numpy.void`.
///
/// # Examples
/// ```
/// use rust_numpy::dtype_constructors::void;
/// let dt = void(1);
/// assert_eq!(dt.itemsize(), 1);
/// ```
pub fn void(size: usize) -> Dtype {
    Dtype::Void { size }
}

// ============================================================================
// Helper Functions for Type Checking
// ============================================================================

/// Check if a dtype is a generic type
///
/// Returns true if the dtype is any valid dtype.
/// Equivalent to `isinstance(dtype, numpy.generic)`.
pub fn is_generic(dtype: &Dtype) -> bool {
    true // All dtypes are generic
}

/// Check if a dtype is a numeric type
///
/// Returns true if the dtype is integer, unsigned, float, or complex.
/// Equivalent to `isinstance(dtype, numpy.number)`.
pub fn is_number(dtype: &Dtype) -> bool {
    matches!(
        dtype.kind(),
        DtypeKind::Integer | DtypeKind::Unsigned | DtypeKind::Float | DtypeKind::Complex
    )
}

/// Check if a dtype is an integer type
///
/// Returns true if the dtype is a signed integer.
/// Equivalent to `isinstance(dtype, numpy.integer)`.
pub fn is_integer(dtype: &Dtype) -> bool {
    dtype.kind() == DtypeKind::Integer
}

/// Check if a dtype is a signed integer type
///
/// Returns true if the dtype is a signed integer.
/// Equivalent to `isinstance(dtype, numpy.signedinteger)`.
pub fn is_signedinteger(dtype: &Dtype) -> bool {
    dtype.kind() == DtypeKind::Integer
}

/// Check if a dtype is an unsigned integer type
///
/// Returns true if the dtype is an unsigned integer.
/// Equivalent to `isinstance(dtype, numpy.unsignedinteger)`.
pub fn is_unsignedinteger(dtype: &Dtype) -> bool {
    dtype.kind() == DtypeKind::Unsigned
}

/// Check if a dtype is a floating point type
///
/// Returns true if the dtype is a float.
/// Equivalent to `isinstance(dtype, numpy.floating)`.
pub fn is_floating(dtype: &Dtype) -> bool {
    dtype.kind() == DtypeKind::Float
}

/// Check if a dtype is a complex type
///
/// Returns true if the dtype is complex.
/// Equivalent to `isinstance(dtype, numpy.complexfloating)`.
pub fn is_complexfloating(dtype: &Dtype) -> bool {
    dtype.kind() == DtypeKind::Complex
}

/// Check if a dtype is a boolean type
///
/// Returns true if the dtype is boolean.
pub fn is_bool(dtype: &Dtype) -> bool {
    dtype.kind() == DtypeKind::Bool
}

/// Check if a dtype is a string type
///
/// Returns true if the dtype is a string or unicode.
pub fn is_string(dtype: &Dtype) -> bool {
    dtype.kind() == DtypeKind::String
}

/// Check if a dtype is an object type
///
/// Returns true if the dtype is object.
pub fn is_object(dtype: &Dtype) -> bool {
    dtype.kind() == DtypeKind::Object
}

/// Check if a dtype is a void type
///
/// Returns true if the dtype is void.
pub fn is_void(dtype: &Dtype) -> bool {
    dtype.kind() == DtypeKind::Void
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_constructor() {
        let dt = bool_();
        assert_eq!(dt.to_string(), "bool");
        assert!(is_bool(&dt));
    }

    #[test]
    fn test_int_constructors() {
        assert_eq!(int8().to_string(), "int8");
        assert_eq!(int16().to_string(), "int16");
        assert_eq!(int32().to_string(), "int32");
        assert_eq!(int64().to_string(), "int64");
        assert_eq!(intp().to_string(), "intp");
        assert_eq!(int_().to_string(), "intp");
        assert_eq!(intc().to_string(), "int32");
        assert_eq!(short().to_string(), "int16");
        assert_eq!(long().to_string(), "int64");
        assert_eq!(longlong().to_string(), "int64");
    }

    #[test]
    fn test_uint_constructors() {
        assert_eq!(uint8().to_string(), "uint8");
        assert_eq!(uint16().to_string(), "uint16");
        assert_eq!(uint32().to_string(), "uint32");
        assert_eq!(uint64().to_string(), "uint64");
        assert_eq!(uintp().to_string(), "uintp");
        assert_eq!(uint().to_string(), "uintp");
        assert_eq!(uintc().to_string(), "uint32");
        assert_eq!(ushort().to_string(), "uint16");
        assert_eq!(ulong().to_string(), "uint64");
        assert_eq!(ulonglong().to_string(), "uint64");
    }

    #[test]
    fn test_float_constructors() {
        assert_eq!(float16().to_string(), "float16");
        assert_eq!(float32().to_string(), "float32");
        assert_eq!(float64().to_string(), "float64");
        assert_eq!(float128().to_string(), "float128");
    }

    #[test]
    fn test_complex_constructors() {
        assert_eq!(complex64().to_string(), "complex64");
        assert_eq!(complex128().to_string(), "complex128");
        assert_eq!(complex256().to_string(), "complex128");
        assert_eq!(csingle().to_string(), "complex64");
        assert_eq!(cdouble().to_string(), "complex128");
        assert_eq!(clongdouble().to_string(), "complex128");
    }

    #[test]
    fn test_datetime_constructors() {
        let dt = datetime64(DatetimeUnit::us);
        assert_eq!(dt.to_string(), "datetime64[us]");
        let td = timedelta64(TimedeltaUnit::s);
        assert_eq!(td.to_string(), "timedelta64[s]");
    }

    #[test]
    fn test_other_constructors() {
        assert_eq!(str_().to_string(), "unicode");
        assert_eq!(object_().to_string(), "object");
        assert_eq!(void(8).itemsize(), 8);
        assert_eq!(bytes_(None).itemsize(), 1);
        assert_eq!(bytes_(Some(10)).itemsize(), 10);
    }

    #[test]
    fn test_type_checking() {
        assert!(is_integer(&int32()));
        assert!(is_signedinteger(&int32()));
        assert!(is_unsignedinteger(&uint32()));
        assert!(is_floating(&float64()));
        assert!(is_complexfloating(&complex128()));
        assert!(is_bool(&bool_()));
        assert!(is_string(&str_()));
        assert!(is_object(&object_()));
        assert!(is_void(&void(4)));
        assert!(is_number(&int32()));
        assert!(is_number(&float64()));
        assert!(is_number(&complex128()));
    }
}
