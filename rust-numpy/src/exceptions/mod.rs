//! NumPy exception types for error handling
//!
//! Equivalent to numpy.exceptions (NumPy 2.0+)

use std::fmt;

/// Base class for all NumPy exceptions
#[derive(Debug, Clone)]
pub struct NumpyError {
    pub message: String,
}

impl fmt::Display for NumpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for NumpyError {}

/// Exception raised for complex number warnings
#[derive(Debug, Clone)]
pub struct ComplexWarning {
    pub message: String,
}

/// Exception raised for overflow warnings
#[derive(Debug, Clone)]
pub struct OverflowWarning {
    pub message: String,
}

/// Exception raised for invalid value warnings
#[derive(Debug, Clone)]
pub struct InvalidValueWarning {
    pub message: String,
}

/// Exception raised for divide by zero warnings
#[derive(Debug, Clone)]
pub struct DivideByZeroWarning {
    pub message: String,
}

/// Exception raised for casting warnings
#[derive(Debug, Clone)]
pub struct CastingWarning {
    pub message: String,
}
