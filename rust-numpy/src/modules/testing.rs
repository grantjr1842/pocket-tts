use crate::array::Array;
use crate::error::{NumPyError, Result};
use num_traits::Float;
use std::fmt::Debug;

/// Assert that two arrays are equal.
///
/// Raises an error if shapes or elements differ.
pub fn assert_array_equal<T>(actual: &Array<T>, desired: &Array<T>) -> Result<()>
where
    T: Clone + PartialEq + Debug + 'static,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    let actual_data = actual.data();
    let desired_data = desired.data();

    for (_i, (a, b)) in actual_data.iter().zip(desired_data.iter()).enumerate() {
        if a != b {
            return Err(NumPyError::value_error(
                format!("{:?}", a),
                format!("{:?}", b),
            ));
        }
    }

    Ok(())
}

/// Assert that two arrays are equal within a certain tolerance.
///
/// Specifically for floating point types.
pub fn assert_array_almost_equal<T>(
    actual: &Array<T>,
    desired: &Array<T>,
    decimal: usize,
) -> Result<()>
where
    T: Clone + Float + Debug + 'static,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    let threshold = T::from(10.0).unwrap().powi(-(decimal as i32));
    let actual_data = actual.data();
    let desired_data = desired.data();

    for (_i, (a, b)) in actual_data.iter().zip(desired_data.iter()).enumerate() {
        if (*a - *b).abs() > threshold {
            return Err(NumPyError::value_error(
                format!("{:?}", a),
                format!("{:?}", b),
            ));
        }
    }

    Ok(())
}

/// Assert that two arrays have the same shape.
pub fn assert_array_shape_equal<T, U>(a: &Array<T>, b: &Array<U>) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(NumPyError::shape_mismatch(
            b.shape().to_vec(),
            a.shape().to_vec(),
        ));
    }
    Ok(())
}

pub mod exports {
    pub use super::{assert_array_almost_equal, assert_array_equal, assert_array_shape_equal};
}
