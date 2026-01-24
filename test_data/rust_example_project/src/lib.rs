//! Rust NumPy Validator
//!
//! This library demonstrates how to validate a Rust NumPy implementation
//! against the comprehensive test examples generated from the original NumPy source.

pub mod array;
pub mod functions;
pub mod test_utils;

pub use array::{array, Array};
pub use functions::*;
pub use test_utils::*;

/// Your NumPy port's prelude
pub mod prelude {
    pub use crate::{array, Array};
    pub use num_complex::Complex64;
    pub use num_traits::{Float, Num};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_validation() {
        // This is a simple example of how to validate your implementation
        let validator = test_utils::NumPyTestValidator::new();

        // Test array creation
        let results = validator.validate_array_creation().unwrap();

        for result in results {
            println!("{}", result);
            assert!(result.contains("✅") || result.contains("Skipped"));
        }
    }
}
