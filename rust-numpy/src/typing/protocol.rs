//! Protocol traits for NumPy-compatible duck typing
//!
//! This module provides protocol traits similar to NumPy's array function protocol
//! system, enabling duck-typing for types that behave like arrays without requiring
//! explicit inheritance.
//!
//! # Example
//!
//! ```rust
//! use rust_numpy::typing::protocol::{SupportsArray, SupportsDType, SupportsShape};
//! use rust_numpy::{Array, Dtype};
//!
//! fn process_array_like<T: SupportsArray<f64>>(item: &T) -> rust_numpy::Result<Array<f64>> {
//!     item.as_array()
//! }
//! ```

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::Result;

/// Protocol trait for types that can be converted to arrays
///
/// This trait enables duck-typing for array-convertible types, similar to
/// NumPy's `__array__` protocol in Python. Types implementing this trait
/// can be converted to arrays either by reference (cheap) or by value (owned).
///
/// # Examples
///
/// ```rust
/// use rust_numpy::typing::protocol::SupportsArray;
/// use rust_numpy::Array;
///
/// fn process<T: SupportsArray<f64>>(data: &T) -> rust_numpy::Result<Array<f64>> {
///     data.as_array()
/// }
/// ```
pub trait SupportsArray<T: Clone + Default + 'static> {
    /// Convert to an array by reference (cheap, may share data)
    ///
    /// This method attempts to create an array view or lightweight conversion
    /// from the type. For types that already are arrays, this may return a clone.
    ///
    /// # Returns
    ///
    /// * `Ok(Array<T>)` - The converted array
    /// * `Err(NumPyError)` - If conversion fails
    fn as_array(&self) -> Result<Array<T>>;

    /// Convert to an array by value (owned conversion)
    ///
    /// This method consumes the value to create an array. For types that
    /// already are arrays, this may simply return the array directly.
    ///
    /// # Returns
    ///
    /// * `Ok(Array<T>)` - The converted array
    /// * `Err(NumPyError)` - If conversion fails
    fn to_array(self) -> Result<Array<T>>
    where
        Self: Sized;
}

/// Protocol trait for types that have a dtype
///
/// This trait enables duck-typing for types with an associated data type,
/// similar to NumPy's `.dtype` attribute. Types implementing this trait
/// expose their element type information.
///
/// # Examples
///
/// ```rust
/// use rust_numpy::typing::protocol::SupportsDType;
/// use rust_numpy::Dtype;
///
/// fn check_dtype<T: SupportsDType>(item: &T) -> Dtype {
///     item.dtype()
/// }
/// ```
pub trait SupportsDType {
    /// Get the dtype of this type
    ///
    /// Returns the data type descriptor that describes the element type
    /// of this array-like or scalar value.
    ///
    /// # Returns
    ///
    /// The `Dtype` describing the element type
    fn dtype(&self) -> Dtype;
}

/// Protocol trait for types that have a shape
///
/// This trait enables duck-typing for types with shape information,
/// similar to NumPy's `.shape` and `.ndim` attributes. Types implementing
/// this trait expose their dimensional structure.
///
/// # Examples
///
/// ```rust
/// use rust_numpy::typing::protocol::SupportsShape;
///
/// fn get_dimensions<T: SupportsShape>(item: &T) -> usize {
///     item.ndim()
/// }
/// ```
pub trait SupportsShape {
    /// Get the shape of this type
    ///
    /// Returns the dimensions of the array as a slice of sizes.
    /// For a 2D array with 3 rows and 4 columns, this returns `&[3, 4]`.
    ///
    /// # Returns
    ///
    /// A slice containing the size of each dimension
    fn shape(&self) -> &[usize];

    /// Get the number of dimensions
    ///
    /// Returns the number of axes (dimensions) in the array.
    /// For a scalar, this returns 0. For a vector, 1. For a matrix, 2, etc.
    ///
    /// # Returns
    ///
    /// The number of dimensions
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements
    ///
    /// Returns the product of all dimension sizes.
    ///
    /// # Returns
    ///
    /// The total number of elements in the array
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if the array is 0-dimensional (scalar-like)
    ///
    /// # Returns
    ///
    /// `true` if the array has no dimensions (shape is empty)
    fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }

    /// Check if the array is empty (has zero elements)
    ///
    /// # Returns
    ///
    /// `true` if any dimension is 0
    fn is_empty(&self) -> bool {
        self.shape().iter().any(|&d| d == 0)
    }
}

// Implementation for Array<T>
impl<T: Clone + Default + 'static> SupportsArray<T> for Array<T> {
    fn as_array(&self) -> Result<Array<T>> {
        Ok(self.clone())
    }

    fn to_array(self) -> Result<Array<T>>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

impl<T: Clone + Default + 'static> SupportsDType for Array<T> {
    fn dtype(&self) -> Dtype {
        self.dtype.clone()
    }
}

impl<T: Clone + Default + 'static> SupportsShape for Array<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// Implementation for Vec<T> - acts as 1D array
impl<T: Clone + Default + 'static> SupportsArray<T> for Vec<T> {
    fn as_array(&self) -> Result<Array<T>> {
        Ok(Array::from_data(self.clone(), vec![self.len()]))
    }

    fn to_array(self) -> Result<Array<T>>
    where
        Self: Sized,
    {
        let len = self.len();
        Ok(Array::from_data(self, vec![len]))
    }
}

// Vec doesn't have a dtype on its own
impl<T: Clone + Default + 'static> SupportsShape for Vec<T> {
    fn shape(&self) -> &[usize] {
        // We can't return a reference to a temporary, so we use a static
        // This is a limitation - Vec doesn't naturally have a shape
        // Users should convert to Array first for full protocol support
        &[]
    }

    fn ndim(&self) -> usize {
        1 // Vec acts as 1D
    }

    fn size(&self) -> usize {
        self.len()
    }
}

// Implementation for scalar types (0D)
macro_rules! impl_supports_dtype_for_scalar {
    ($type:ty, $dtype_variant:ident) => {
        impl SupportsDType for $type {
            fn dtype(&self) -> Dtype {
                Dtype::$dtype_variant { byteorder: None }
            }
        }

        impl SupportsShape for $type {
            fn shape(&self) -> &[usize] {
                &[]
            }

            fn ndim(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                1
            }

            fn is_scalar(&self) -> bool {
                true
            }
        }
    };
}

impl_supports_dtype_for_scalar!(i8, Int8);
impl_supports_dtype_for_scalar!(i16, Int16);
impl_supports_dtype_for_scalar!(i32, Int32);
impl_supports_dtype_for_scalar!(i64, Int64);
impl_supports_dtype_for_scalar!(u8, UInt8);
impl_supports_dtype_for_scalar!(u16, UInt16);
impl_supports_dtype_for_scalar!(u32, UInt32);
impl_supports_dtype_for_scalar!(u64, UInt64);
impl_supports_dtype_for_scalar!(f32, Float32);
impl_supports_dtype_for_scalar!(f64, Float64);
impl_supports_dtype_for_scalar!(bool, Bool);

// Complex numbers
impl<T: Clone + Default + 'static> SupportsDType for num_complex::Complex<T>
where
    T: SupportsDType,
{
    fn dtype(&self) -> Dtype {
        // Complex dtype is based on the inner type
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            Dtype::Complex64 { byteorder: None }
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            Dtype::Complex128 { byteorder: None }
        } else {
            Dtype::Object
        }
    }
}

impl<T: Clone + Default + 'static> SupportsShape for num_complex::Complex<T> {
    fn shape(&self) -> &[usize] {
        &[]
    }

    fn ndim(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        1
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_supports_array() {
        let arr = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let arr2 = arr.as_array().unwrap();
        assert_eq!(arr2.shape(), &[3]);

        let arr3 = arr.clone().to_array().unwrap();
        assert_eq!(arr3.shape(), &[3]);
    }

    #[test]
    fn test_array_supports_dtype() {
        let arr = Array::from_data(vec![1.0f64, 2.0, 3.0], vec![3]);
        let dtype = arr.dtype();
        assert_eq!(dtype, Dtype::Float64 { byteorder: None });
    }

    #[test]
    fn test_array_supports_shape() {
        let arr = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.size(), 4);
        assert!(!arr.is_scalar());
        assert!(!arr.is_empty());
    }

    #[test]
    fn test_vec_supports_array() {
        let data = vec![1.0, 2.0, 3.0];
        let arr = data.as_array().unwrap();
        assert_eq!(arr.shape(), &[3]);
    }

    #[test]
    fn test_scalar_supports_dtype() {
        let val: f64 = 3.14;
        assert_eq!(val.dtype(), Dtype::Float64 { byteorder: None });

        let val: i32 = 42;
        assert_eq!(val.dtype(), Dtype::Int32 { byteorder: None });

        let val: bool = true;
        assert_eq!(val.dtype(), Dtype::Bool);
    }

    #[test]
    fn test_scalar_supports_shape() {
        let val: f64 = 3.14;
        assert_eq!(val.shape(), &[]);
        assert_eq!(val.ndim(), 0);
        assert_eq!(val.size(), 1);
        assert!(val.is_scalar());
    }

    #[test]
    fn test_empty_array_shape() {
        let arr = Array::<f64>::from_data(vec![], vec![0, 5]);
        assert!(arr.is_empty());
    }
}
