// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Standalone wrapper functions for Array methods
//!
//! This module provides NumPy-compatible standalone functions that wrap Array methods,
//! allowing them to be called as `function(array)` instead of `array.function()`.

use crate::array::Array;
use crate::error::NumPyError;

/// Permute dimensions of an array
///
/// # Arguments
/// * `arr` - Input array
///
/// # Returns
/// A view of the array with axes transposed
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let arr = array![[1, 2], [3, 4], [5, 6]];
/// let transposed = transpose(&arr);
/// assert_eq!(transposed.shape(), &[2, 2]);
/// ```
pub fn transpose<T>(arr: &Array<T>) -> Array<T>
where
    T: Clone + Default + 'static,
{
    arr.transpose()
}

/// Return an array formed from the elements of `arr` at the given indices
///
/// # Arguments
/// * `arr` - Input array
/// * `indices` - The indices of the values to extract
/// * `axis` - The axis over which to select values
///
/// # Returns
/// The output array
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let arr = array![[1, 2], [3, 4], [5, 6]];
/// let indices = array![0, 2];
/// let result = take(&arr, &indices, None)?;
/// assert_eq!(result.shape(), &[2, 2]);
/// ```
pub fn take<T>(
    arr: &Array<T>,
    indices: &Array<usize>,
    axis: Option<usize>,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static,
{
    arr.take(indices, axis)
}

/// Return a new array with the specified shape
///
/// # Arguments
/// * `arr` - Input array
/// * `newshape` - New shape for the array
///
/// # Returns
/// The resized array
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let arr = array![[1, 2, 3], [4, 5, 6]];
/// let resized = resize(&arr, &[4])?;
/// assert_eq!(resized.shape(), &[4]);
/// ```
pub fn resize<T>(arr: &Array<T>, newshape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static,
{
    arr.resize(newshape)
}

/// Return the difference of two arrays
///
/// # Arguments
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
/// The difference of `x1` and `x2`
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![4.0, 5.0, 6.0];
/// let result = subtract(&a, &b)?;
/// assert_eq!(result.data(), &[-3.0, -3.0, -3.0]);
/// ```
pub fn subtract<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static,
{
    x1.subtract(x2, None, crate::dtype::Casting::Safe)
}

/// Return the quotient of two arrays
///
/// # Arguments
/// * `x1` - First input array
/// * `x2` - Second input array
///
/// # Returns
/// The quotient of `x1` and `x2`
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let a = array![10.0, 20.0, 30.0];
/// let b = array![2.0, 4.0, 6.0];
/// let result = divide(&a, &b)?;
/// assert_eq!(result.data(), &[5.0, 5.0, 6.0]);
/// ```
pub fn divide<T>(x1: &Array<T>, x2: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static,
{
    x1.divide(x2, None, crate::dtype::Casting::Safe)
}

/// Return the negative of an array (element-wise)
///
/// # Arguments
/// * `arr` - Input array
///
/// # Returns
/// The negated array
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let arr = array![1.0, -2.0, 3.0];
/// let result = negative(&arr)?;
/// assert_eq!(result.data(), &[-1.0, 2.0, 3.0]);
/// ```
pub fn negative<T>(arr: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static,
{
    arr.negative(None, crate::dtype::Casting::Safe)
}

/// Return the element-wise minimum of two arrays
///
/// # Arguments
/// * `arr1` - First input array
/// * `arr2` - Second input array
///
/// # Returns
/// The element-wise minimum
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let a = array![1.0, 5.0, 3.0];
/// let b = array![2.0, 3.0, 4.0];
/// let result = minimum(&a, &b)?;
/// assert_eq!(result.data(), &[1.0, 3.0, 3.0]);
/// ```
pub fn minimum<T>(arr1: &Array<T>, arr2: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + std::cmp::PartialOrd + 'static,
{
    arr1.minimum(arr2)
}

/// Return the cumulative sum of elements along a given axis
///
/// # Arguments
/// * `arr` - Input array
/// * `axis` - Axis along which cumulative sum is computed
///
/// # Returns
/// The cumulative sum array
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let arr = array![1, 2, 3, 4];
/// let result = cumsum(&arr, None)?;
/// assert_eq!(result.data(), &[1, 3, 6, 10]);
/// ```
pub fn cumsum<T>(arr: &Array<T>, axis: Option<isize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Add<Output = T> + 'static,
{
    arr.cumsum(axis)
}

/// Return the cumulative product of elements along a given axis
///
/// # Arguments
/// * `arr` - Input array
/// * `axis` - Axis along which cumulative product is computed
///
/// # Returns
/// The cumulative product array
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let arr = array![1, 2, 3, 4];
/// let result = cumprod(&arr, None)?;
/// assert_eq!(result.data(), &[1, 2, 6, 24]);
/// ```
pub fn cumprod<T>(arr: &Array<T>, axis: Option<isize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Mul<Output = T> + 'static,
{
    arr.cumprod(axis)
}

/// Return the cumulative sum of elements, treating NaNs as zero
///
/// # Arguments
/// * `arr` - Input array
/// * `axis` - Axis along which cumulative sum is computed
///
/// # Returns
/// The cumulative sum array with NaNs treated as zero
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let arr = array![1.0, 2.0, f64::NAN, 4.0];
/// let result = nancumsum(&arr, None)?;
/// assert_eq!(result.data()[2], 3.0); // NaN is treated as zero
/// ```
pub fn nancumsum<T>(arr: &Array<T>, axis: Option<isize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Add<Output = T> + num_traits::Float + 'static,
{
    arr.nancumsum(axis)
}

/// Return the cumulative product of elements, treating NaNs as one
///
/// # Arguments
/// * `arr` - Input array
/// * `axis` - Axis along which cumulative product is computed
///
/// # Returns
/// The cumulative product array with NaNs treated as one
///
/// # Examples
/// ```rust,ignore
/// use numpy::*;
///
/// let arr = array![1.0, 2.0, f64::NAN, 4.0];
/// let result = nancumprod(&arr, None)?;
/// assert_eq!(result.data()[2], 2.0); // NaN is treated as one
/// ```
pub fn nancumprod<T>(arr: &Array<T>, axis: Option<isize>) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Mul<Output = T> + num_traits::Float + 'static,
{
    arr.nancumprod(axis)
}

