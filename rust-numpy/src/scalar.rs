// Scalar type definitions for NumPy compatibility

use num_traits::Float;

/// Generic scalar type (placeholder for Python object compatibility)
#[derive(Debug, Clone)]
pub struct Generic(pub Box<dyn std::any::Any>);

/// Complex floating point type
pub type ComplexFloating = num_complex::Complex64;

/// Void type for structured arrays
#[derive(Debug, Clone)]
pub struct Void(pub Vec<u8>);

/// Scalar type alias for primitive scalars
pub type Scalar = f64;

/// Number type alias for numeric types
pub type Number = f64;

/// Integer type alias for integer types
pub type Int = i64;

/// Floating point type alias
pub type Floating = f64;
