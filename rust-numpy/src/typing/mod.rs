pub mod bitwidth;
#[cfg(test)]
mod bitwidth_tests;
pub mod dtype_getter;
pub use bitwidth::*;
pub use dtype_getter::*;

use crate::array::Array;
use crate::dtype::Dtype;

/// NDArray type alias for runtime type annotations
///
/// This provides NumPy-compatible type annotations for arrays with specified dtypes.
/// It enables annotating arrays with given dtype and unspecified shape, similar to
/// NumPy's typing.NDArray in Python.
///
/// # Examples
///
/// ```rust
/// use rust_numpy::typing::*;
///
/// // Type annotation for float64 array
/// let arr: NDArray<f64> = array![1.0, 2.0, 3.0];
///
/// // Type annotation for integer array
/// let int_arr: NDArray<i32> = array![1, 2, 3];
/// ```
pub type NDArray<T> = Array<T>;

/// Common NDArray type aliases for specific dtypes
pub mod ndarray_types {
    use super::*;

    /// 8-bit signed integer array
    pub type Int8Array = NDArray<i8>;

    /// 16-bit signed integer array
    pub type Int16Array = NDArray<i16>;

    /// 32-bit signed integer array
    pub type Int32Array = NDArray<i32>;

    /// 64-bit signed integer array
    pub type Int64Array = NDArray<i64>;

    /// 8-bit unsigned integer array
    pub type UInt8Array = NDArray<u8>;

    /// 16-bit unsigned integer array
    pub type UInt16Array = NDArray<u16>;

    /// 32-bit unsigned integer array
    pub type UInt32Array = NDArray<u32>;

    /// 64-bit unsigned integer array
    pub type UInt64Array = NDArray<u64>;

    /// 32-bit float array
    pub type Float32Array = NDArray<f32>;

    /// 64-bit float array
    pub type Float64Array = NDArray<f64>;

    /// 64-bit complex array (two 32-bit floats)
    pub type Complex64Array = NDArray<num_complex::Complex<f32>>;

    /// 128-bit complex array (two 64-bit floats)
    pub type Complex128Array = NDArray<num_complex::Complex<f64>>;

    /// Boolean array
    pub type BoolArray = NDArray<bool>;
}

/// Export common NDArray types
pub use ndarray_types::*;

/// ArrayLike type alias for array-convertible objects
///
/// This represents objects that can be converted to arrays, similar to
/// NumPy's typing.ArrayLike in Python. It includes various types that
/// can be used as input for array creation functions.
///
/// # Examples
///
/// ```rust
/// use rust_numpy::typing::*;
///
/// fn process_data<T: Clone + Default + 'static>(data: ArrayLike<T>) -> Array<T> {
///     asarray(data)
/// }
/// ```
pub trait ArrayLike<T: Clone + Default + 'static> {
    fn to_array(&self) -> Result<Array<T>, crate::error::NumPyError>;
}

/// Implement ArrayLike for common types
impl<T: Clone + Default + 'static> ArrayLike<T> for Array<T> {
    fn to_array(&self) -> Result<Array<T>, crate::error::NumPyError> {
        Ok(self.clone())
    }
}

impl<T: Clone + Default + 'static> ArrayLike<T> for Vec<T> {
    fn to_array(&self) -> Result<Array<T>, crate::error::NumPyError> {
        Ok(Array::from_data(self.clone(), vec![self.len()]))
    }
}

impl<T: Clone + Default + 'static, const N: usize> ArrayLike<T> for [T; N] {
    fn to_array(&self) -> Result<Array<T>, crate::error::NumPyError> {
        Ok(Array::from_data(self.to_vec(), vec![N]))
    }
}

impl<T: Clone + Default + 'static> ArrayLike<T> for &[T] {
    fn to_array(&self) -> Result<Array<T>, crate::error::NumPyError> {
        Ok(Array::from_data(self.to_vec(), vec![self.len()]))
    }
}

/// DtypeLike type alias for dtype-convertible objects
///
/// This represents objects that can be converted to dtypes, similar to
/// NumPy's typing.DtypeLike in Python. It includes various ways to specify
/// data types in NumPy operations.
///
/// # Examples
///
/// ```rust
/// use rust_numpy::typing::*;
/// use rust_numpy::Dtype;
///
/// fn create_array_with_dtype<T: DtypeLike>(dtype_like: T) -> Dtype {
///     dtype_like.to_dtype()
/// }
/// ```
pub trait DtypeLike {
    fn to_dtype(&self) -> Dtype;
}

/// Implement DtypeLike for common types
impl DtypeLike for Dtype {
    fn to_dtype(&self) -> Dtype {
        self.clone()
    }
}

impl DtypeLike for &str {
    fn to_dtype(&self) -> Dtype {
        Dtype::from_str(self).unwrap_or(Dtype::Float64 { byteorder: None })
    }
}

impl DtypeLike for String {
    fn to_dtype(&self) -> Dtype {
        Dtype::from_str(self).unwrap_or(Dtype::Float64 { byteorder: None })
    }
}

// Implement for primitive types
impl DtypeLike for i8 {
    fn to_dtype(&self) -> Dtype {
        Dtype::Int8 { byteorder: None }
    }
}
impl DtypeLike for i16 {
    fn to_dtype(&self) -> Dtype {
        Dtype::Int16 { byteorder: None }
    }
}
impl DtypeLike for i32 {
    fn to_dtype(&self) -> Dtype {
        Dtype::Int32 { byteorder: None }
    }
}
impl DtypeLike for i64 {
    fn to_dtype(&self) -> Dtype {
        Dtype::Int64 { byteorder: None }
    }
}
impl DtypeLike for u8 {
    fn to_dtype(&self) -> Dtype {
        Dtype::UInt8 { byteorder: None }
    }
}
impl DtypeLike for u16 {
    fn to_dtype(&self) -> Dtype {
        Dtype::UInt16 { byteorder: None }
    }
}
impl DtypeLike for u32 {
    fn to_dtype(&self) -> Dtype {
        Dtype::UInt32 { byteorder: None }
    }
}
impl DtypeLike for u64 {
    fn to_dtype(&self) -> Dtype {
        Dtype::UInt64 { byteorder: None }
    }
}
impl DtypeLike for f32 {
    fn to_dtype(&self) -> Dtype {
        Dtype::Float32 { byteorder: None }
    }
}
impl DtypeLike for f64 {
    fn to_dtype(&self) -> Dtype {
        Dtype::Float64 { byteorder: None }
    }
}
impl DtypeLike for bool {
    fn to_dtype(&self) -> Dtype {
        Dtype::Bool
    }
}

/// Re-export commonly used typing aliases for convenience
pub mod prelude {
    pub use super::{
        ArrayLike, BoolArray, Complex128Array, Complex64Array, DtypeLike, Float32Array,
        Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, NDArray, UInt16Array,
        UInt32Array, UInt64Array, UInt8Array,
    };
}

/// Additional type aliases matching NumPy's typing module
/// These provide compatibility with NumPy's type annotations
pub mod aliases {
    /// Shape-like type for array shapes
    /// Represents the shape parameter in NumPy arrays
    pub type ShapeLike = Vec<usize>;

    /// Index-like type for array indexing
    /// Represents valid index types for NumPy arrays
    pub type SupportsIndex = isize;

    // TODO: Implement these types when the corresponding modules are created
    // /// Generic scalar type
    // /// Represents any scalar value compatible with NumPy
    // pub type Scalar = crate::Scalar;
    //
    // /// Number type alias
    // /// Represents NumPy number types
    // pub type Number = crate::Number;
    //
    // /// Integer type alias
    // /// Represents NumPy integer types
    // pub type Integer = crate::Integer;
    //
    // /// Floating type alias
    // /// Represents NumPy floating types
    // pub type Floating = crate::Floating;
    //
    // /// Complex floating type alias
    // /// Represents NumPy complex floating types
    // pub type ComplexFloating = crate::ComplexFloating;
    //
    // /// Generic type alias
    // /// Represents generic NumPy types
    // pub type Generic = crate::Generic;

    /// Boolean type alias
    /// Represents NumPy boolean type
    pub type Boolean = bool;

    // TODO: Implement these types when the corresponding modules are created
    // /// Object type alias
    // /// Represents NumPy object type
    // pub type Object = crate::Object;
    //
    // /// String type alias
    // /// Represents NumPy string type
    // pub type String_ = crate::String_;
    //
    // /// Unicode string type alias
    // /// Represents NumPy unicode string type
    // pub type Unicode = crate::Unicode;
    //
    // /// Bytes type alias
    // /// Represents NumPy bytes type
    // pub type Bytes = crate::Bytes;

    // TODO: Implement Void type when the corresponding module is created
    // /// Void type alias
    // /// Represents NumPy void type
    // pub type Void = crate::Void;
}

/// Re-export additional aliases
pub use aliases::*;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod typing_tests;
