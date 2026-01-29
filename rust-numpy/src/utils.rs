//! NumPy utility functions

use crate::error::{NumPyError, Result};

/// Convert an integer to a string in a given base
pub fn base_repr(number: i64, base: i64, padding: Option<usize>) -> Result<String> {
    if base < 2 || base > 36 {
        return Err(NumPyError::invalid_value("base must be between 2 and 36"));
    }

    let mut num = number;
    if num == 0 {
        let pad_len = padding.unwrap_or(1);
        return Ok("0".repeat(pad_len));
    }

    let mut result = String::new();
    let is_negative = num < 0;
    if is_negative {
        num = -num;
    }

    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

    while num > 0 {
        let digit = (num % base) as usize;
        result.insert(0, DIGITS[digit] as char);
        num /= base;
    }

    if is_negative {
        result.insert(0, '-');
    }

    if let Some(pad_len) = padding {
        if result.len() < pad_len {
            let pad_len = pad_len - result.len();
            if is_negative {
                result.insert_str(1, &"0".repeat(pad_len));
            } else {
                result.insert_str(0, &"0".repeat(pad_len));
            }
        }
    }

    Ok(result)
}

/// Convert an integer to a binary string representation
pub fn binary_repr(number: i64, width: Option<usize>) -> Result<String> {
    base_repr(number, 2, width)
}

/// Count the number of set bits (population count)
pub fn bitwise_count<T>(number: T) -> Result<T>
where
    T: Clone
        + Default
        + std::ops::BitAnd<Output = T>
        + std::ops::Shr<Output = T>
        + From<u8>
        + PartialEq
        + std::ops::Add<Output = T>
        + 'static,
{
    // Simple implementation - this would need to be specialized for each integer type
    let mut n = number.clone();
    let mut count = T::from(0u8);
    let one = T::from(1u8);

    while n != T::default() {
        count = count + (n.clone() & one.clone());
        n = n >> one.clone();
    }

    Ok(count)
}

/// Bitwise NOT (invert)
pub fn bitwise_invert<T>(number: T) -> Result<T>
where
    T: std::ops::Not<Output = T>,
{
    Ok(!number)
}

/// Left shift operation
pub fn bitwise_left_shift<T>(number: T, shift: i32) -> Result<T>
where
    T: std::ops::Shl<i32, Output = T>,
{
    if shift < 0 {
        return Err(NumPyError::invalid_value("negative shift count"));
    }
    Ok(number << shift)
}

/// Right shift operation
pub fn bitwise_right_shift<T>(number: T, shift: i32) -> Result<T>
where
    T: std::ops::Shr<i32, Output = T>,
{
    if shift < 0 {
        return Err(NumPyError::invalid_value("negative shift count"));
    }
    Ok(number >> shift)
}

/// Create matrix from nested arrays
pub fn bmat(obj: &str) -> Result<String> {
    // Simplified implementation - just return the input as a matrix description
    Ok(format!("bmat({})", obj))
}

/// Boolean type placeholder
pub fn bool() -> bool {
    true
}

/// Boolean type placeholder (underscore version)
pub fn bool_() -> bool {
    true
}

/// Byte type placeholder
pub fn byte() -> u8 {
    0
}

/// Bytes type placeholder
pub fn bytes_() -> Vec<u8> {
    vec![]
}

/// Double precision float placeholder
pub fn double() -> f64 {
    0.0
}

/// Single precision float placeholder
pub fn single() -> f32 {
    0.0
}

/// Half precision float placeholder
pub fn half() -> f16 {
    f16::from_f32(0.0)
}

/// Complex number placeholders
pub fn complex64() -> num_complex::Complex<f32> {
    num_complex::Complex::new(0.0, 0.0)
}

pub fn complex128() -> num_complex::Complex<f64> {
    num_complex::Complex::new(0.0, 0.0)
}

/// Integer type placeholders
pub fn int8() -> i8 {
    0
}

pub fn int16() -> i16 {
    0
}

pub fn int32() -> i32 {
    0
}

pub fn int64() -> i64 {
    0
}

pub fn uint8() -> u8 {
    0
}

pub fn uint16() -> u16 {
    0
}

pub fn uint32() -> u32 {
    0
}

pub fn uint64() -> u64 {
    0
}

/// Generic type placeholders
pub fn floating() -> &'static str {
    "floating"
}

pub fn integer() -> &'static str {
    "integer"
}

pub fn generic() -> &'static str {
    "generic"
}

pub fn flexible() -> &'static str {
    "flexible"
}

pub fn inexact() -> &'static str {
    "inexact"
}

pub fn signedinteger() -> &'static str {
    "signedinteger"
}

pub fn unsignedinteger() -> &'static str {
    "unsignedinteger"
}

pub fn character() -> &'static str {
    "character"
}

pub fn complexfloating() -> &'static str {
    "complexfloating"
}

/// String type placeholder
pub fn str_() -> &'static str {
    "str"
}

/// Void type placeholder
pub fn void() -> &'static str {
    "void"
}

/// Object type placeholder
pub fn object_() -> &'static str {
    "object"
}

/// NumPy version info
pub fn version() -> &'static str {
    "2.4.0"
}

/// Show NumPy configuration
pub fn show_config() -> &'static str {
    "NumPy configuration"
}

/// Show runtime info
pub fn show_runtime() -> &'static str {
    "NumPy runtime info"
}

/// Get include directory
pub fn get_include() -> &'static str {
    "/usr/include/numpy"
}

/// Test function placeholder
pub fn test() -> &'static str {
    "NumPy test suite"
}

/// Info function
pub fn info() -> &'static str {
    "NumPy info"
}

/// Type name function
pub fn typename(dtype: &str) -> Result<String> {
    Ok(format!("numpy.{}", dtype))
}

/// Print options placeholder
pub fn get_printoptions() -> &'static str {
    "print options"
}

pub fn set_printoptions(_precision: Option<usize>) -> Result<()> {
    Ok(())
}

/// Buffer size functions
pub fn getbufsize() -> usize {
    8192
}

pub fn setbufsize(_size: usize) -> Result<()> {
    Ok(())
}

/// Error handling functions
pub fn geterr() -> &'static str {
    "error handling state"
}

pub fn seterr(
    _all: Option<&str>,
    _divide: Option<&str>,
    _over: Option<&str>,
    _under: Option<&str>,
    _invalid: Option<&str>,
) -> Result<()> {
    Ok(())
}

pub fn geterrcall() -> &'static str {
    "error callback"
}

pub fn seterrcall(_func: Option<&str>) -> Result<()> {
    Ok(())
}

/// Error state context manager placeholder
pub fn errstate() -> &'static str {
    "errstate context manager"
}

/// Floating point info
pub fn finfo(dtype: &str) -> Result<String> {
    Ok(format!("Machine limits for float {}", dtype))
}

/// Integer info
pub fn iinfo(dtype: &str) -> Result<String> {
    Ok(format!("Machine limits for integer {}", dtype))
}

/// Check if scalar
pub fn isscalar(obj: &str) -> bool {
    // Simple heuristic
    obj.chars()
        .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
}

/// Check if iterable
pub fn iterable(obj: &str) -> bool {
    obj.contains('[') || obj.contains('(') || obj.contains('{')
}

/// Check subtype relationship
pub fn issubdtype(arg1: &str, arg2: &str) -> bool {
    arg1 == arg2 || arg2 == "generic"
}

/// Check if dtype
pub fn isdtype(dtype: &str) -> bool {
    dtype.contains("int") || dtype.contains("float") || dtype.contains("complex")
}

/// Check if complex
pub fn iscomplex(obj: &str) -> bool {
    obj.contains('j') || obj.contains('i')
}

pub fn isreal(obj: &str) -> bool {
    !iscomplex(obj)
}

pub fn iscomplexobj(obj: &str) -> bool {
    iscomplex(obj)
}

pub fn isrealobj(obj: &str) -> bool {
    isreal(obj)
}

/// Memory layout checks
pub fn isfortran(arr: &str) -> bool {
    arr.contains("F order")
}

/// Check if arrays share memory
pub fn may_share_memory(a: &str, b: &str) -> bool {
    a == b // Simplified
}

pub fn shares_memory(a: &str, b: &str) -> bool {
    may_share_memory(a, b)
}

/// NaN/Inf checks
pub fn isnat(val: &str) -> bool {
    val == "NaT"
}

/// Minimum scalar type
pub fn min_scalar_type(_arr: &str) -> &'static str {
    "uint8"
}

/// Minimum type code
pub fn mintypecode(_arr: &str) -> &'static str {
    "B"
}

/// Common type
pub fn common_type(_arrays: &[&str]) -> &'static str {
    "float64"
}

/// Result type
pub fn result_type(_arrays: &[&str]) -> &'static str {
    "float64"
}

/// Promote types
pub fn promote_types(a: &str, b: &str) -> &'static str {
    if a == "float64" || b == "float64" {
        "float64"
    } else if a == "float32" || b == "float32" {
        "float32"
    } else {
        "int64"
    }
}

/// Can cast check
pub fn can_cast(from_: &str, to_: &str) -> bool {
    // Simplified casting rules
    match (from_, to_) {
        ("int8", "int16") => true,
        ("int8", "int32") => true,
        ("int8", "int64") => true,
        ("int8", "float32") => true,
        ("int8", "float64") => true,
        ("int16", "int32") => true,
        ("int16", "int64") => true,
        ("int16", "float32") => true,
        ("int16", "float64") => true,
        ("int32", "int64") => true,
        ("int32", "float64") => true,
        ("float32", "float64") => true,
        _ => from_ == to_,
    }
}

// f16 type placeholder (would need the half crate)
#[allow(non_camel_case_types)]
pub struct f16(f32);

impl f16 {
    pub fn from_f32(val: f32) -> Self {
        f16(val)
    }
}
