// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! NumPy-compatible array content operations
//!
//! This module provides comprehensive array content operations that match the NumPy API,
//! including copying, cumulative operations, filling, and other content manipulation methods.

use num_traits::{Float, One, Zero};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::array::Array;
use crate::error::{NumPyError, Result};

/// Create and return a deep copy of the array
///
/// # Arguments
/// * `a` - Input array to copy
///
/// # Returns
/// New array with copied data and same shape/strides
///
/// # Examples
/// ```
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = copy(&a);
/// assert_eq!(a.to_vec(), b.to_vec());
/// assert_ne!(a.as_ptr(), b.as_ptr());
/// ```
pub fn copy<T>(a: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    Ok(Array::from_data(a.to_vec(), a.shape().to_vec()))
}

/// Cumulative product of elements along a given axis
///
/// # Arguments
/// * `a` - Input array
/// * `axis` - Axis along which to compute the cumulative product
///
/// # Returns
/// Array of same shape containing cumulative product
///
/// # Examples
/// ```
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2])?;
/// let result = cumprod(&a, 0)?;
/// ```
pub fn cumprod<T>(a: &Array<T>, axis: isize) -> Result<Array<T>>
where
    T: Clone + Default + One + 'static,
{
    let axis = if axis < 0 {
        a.ndim() as isize + axis
    } else {
        axis
    };
    if axis < 0 || axis >= a.ndim() as isize {
        return Err(NumPyError::invalid_value("axis out of bounds"));
    }

    let axis = axis as usize;
    let shape = a.shape();
    let mut result_data = Vec::with_capacity(a.size());

    // Compute cumulative product along the specified axis
    for i in 0..a.size() {
        let mut indices = crate::strides::compute_multi_indices(i, shape);

        // Compute cumulative product up to this position along the axis
        let mut cumprod_val = T::one();
        for k in 0..=indices[axis] {
            indices[axis] = k;
            let linear_idx = crate::strides::compute_linear_index(&indices, a.strides());
            if let Some(val) = a.get(linear_idx as usize) {
                cumprod_val = cumprod_val * val.clone();
            }
        }

        result_data.push(cumprod_val);
    }

    Ok(Array::from_data(result_data, shape.to_vec()))
}

/// Cumulative sum of elements along a given axis
///
/// # Arguments
/// * `a` - Input array
/// * `axis` - Axis along which to compute the cumulative sum
///
/// # Returns
/// Array of same shape containing cumulative sum
///
/// # Examples
/// ```
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2])?;
/// let result = cumsum(&a, 0)?;
/// ```
pub fn cumsum<T>(a: &Array<T>, axis: isize) -> Result<Array<T>>
where
    T: Clone + Default + Zero + 'static,
{
    let axis = if axis < 0 {
        a.ndim() as isize + axis
    } else {
        axis
    };
    if axis < 0 || axis >= a.ndim() as isize {
        return Err(NumPyError::invalid_value("axis out of bounds"));
    }

    let axis = axis as usize;
    let shape = a.shape();
    let mut result_data = Vec::with_capacity(a.size());

    // Compute cumulative sum along the specified axis
    for i in 0..a.size() {
        let mut indices = crate::strides::compute_multi_indices(i, shape);

        // Compute cumulative sum up to this position along the axis
        let mut cumsum_val = T::zero();
        for k in 0..=indices[axis] {
            indices[axis] = k;
            let linear_idx = crate::strides::compute_linear_index(&indices, a.strides());
            if let Some(val) = a.get(linear_idx as usize) {
                cumsum_val = cumsum_val + val.clone();
            }
        }

        result_data.push(cumsum_val);
    }

    Ok(Array::from_data(result_data, shape.to_vec()))
}

/// Fill array with scalar value in-place
///
/// # Arguments
/// * `a` - Array to fill (modified in-place)
/// * `value` - Scalar value to fill with
///
/// # Examples
/// ```
/// let mut a = Array::zeros(&[3, 3]);
/// fill(&mut a, 5);
/// assert_eq!(a.to_vec(), vec![5; 9]);
/// ```
pub fn fill<T>(a: &mut Array<T>, value: T)
where
    T: Clone + Default + 'static,
{
    for i in 0..a.size() {
        a.set_linear(i, value.clone());
    }
}

/// Extract single element from array as scalar
///
/// # Arguments
/// * `a` - Input array
/// * `args` - Optional indices (if None, returns first element)
///
/// # Returns
/// Single element value
///
/// # Examples
/// ```
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let val = item(&a, None)?;
/// assert_eq!(val, 1);
/// ```
pub fn item<T>(a: &Array<T>, args: Option<&[usize]>) -> Result<T>
where
    T: Clone + Default + 'static,
{
    if a.size() == 0 {
        return Err(NumPyError::invalid_value(
            "cannot extract item from empty array",
        ));
    }

    let index = if let Some(indices) = args {
        if indices.len() != a.ndim() {
            return Err(NumPyError::invalid_value(
                "indices must match array dimensions",
            ));
        }
        crate::strides::compute_linear_index(indices, a.strides()) as usize
    } else {
        0 // First element
    };

    if let Some(val) = a.get(index) {
        Ok(val.clone())
    } else {
        Err(NumPyError::index_error(index, a.size()))
    }
}

/// Set values at flattened indices
///
/// # Arguments
/// * `a` - Array to modify (modified in-place)
/// * `indices` - Flattened indices to set
/// * `values` - Values to set at the specified indices
///
/// # Examples
/// ```
/// let mut a = Array::from_vec(vec![1, 2, 3, 4]);
/// put(&mut a, &[1, 3], &[10, 20])?;
/// assert_eq!(a.to_vec(), vec![1, 10, 3, 20]);
/// ```
pub fn put<T>(a: &mut Array<T>, indices: &[usize], values: &[T]) -> Result<()>
where
    T: Clone + Default + 'static,
{
    if indices.is_empty() {
        return Ok(());
    }

    for (i, &idx) in indices.iter().enumerate() {
        if idx >= a.size() {
            return Err(NumPyError::index_error(idx, a.size()));
        }

        let value = if i < values.len() {
            &values[i]
        } else {
            // If values shorter than indices, repeat the last value
            &values[values.len() - 1]
        };

        a.set_linear(idx, value.clone());
    }

    Ok(())
}

/// Round array elements to given number of decimals
///
/// # Arguments
/// * `a` - Input array
/// * `decimals` - Number of decimal places to round to
///
/// # Returns
/// New array with rounded values
///
/// # Examples
/// ```
/// let a = Array::from_vec(vec![1.234, 2.567, 3.891]);
/// let result = round(&a, 2)?;
/// assert_eq!(result.to_vec(), vec![1.23, 2.57, 3.89]);
/// ```
pub fn round<T>(a: &Array<T>, decimals: i32) -> Result<Array<T>>
where
    T: Clone + Default + Float + 'static,
{
    let factor = T::from(10.0f64).unwrap().powi(decimals);
    let mut result_data = Vec::with_capacity(a.size());

    for i in 0..a.size() {
        if let Some(val) = a.get_linear(i) {
            let rounded = (val.clone() * factor.clone()).round() / factor.clone();
            result_data.push(rounded);
        } else {
            result_data.push(T::default());
        }
    }

    Ok(Array::from_data(result_data, a.shape().to_vec()))
}

/// Compute sum along diagonal of array
///
/// # Arguments
/// * `a` - Input array
/// * `offset` - Diagonal offset (0 for main diagonal, positive for upper, negative for lower)
/// * `axis1` - First axis of diagonal (default: 0)
/// * `axis2` - Second axis of diagonal (default: 1)
///
/// # Returns
/// Scalar sum along the specified diagonal
///
/// # Examples
/// ```
/// let a = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2])?;
/// let trace_val = trace(&a, 0, 0, 1)?;
/// assert_eq!(trace_val, 5); // 1 + 4
/// ```
pub fn trace<T>(a: &Array<T>, offset: isize, axis1: usize, axis2: usize) -> Result<T>
where
    T: Clone + Default + Zero + 'static,
{
    if a.ndim() < 2 {
        return Err(NumPyError::invalid_value(
            "trace requires at least 2D array",
        ));
    }

    if axis1 >= a.ndim() || axis2 >= a.ndim() {
        return Err(NumPyError::invalid_value("axis out of bounds"));
    }

    let shape = a.shape();
    let dim1 = shape[axis1];
    let dim2 = shape[axis2];

    let mut sum = T::zero();
    let mut count = 0;

    // Iterate over diagonal elements
    for i in 0..dim1.min(dim2) {
        let mut indices = vec![0; a.ndim()];

        // Set axis1 index
        indices[axis1] = i;

        // Set axis2 index with offset
        let j = (i as isize + offset) as usize;
        if j < dim2 {
            indices[axis2] = j;

            let linear_idx = crate::strides::compute_linear_index(&indices, a.strides());
            if let Some(val) = a.get(linear_idx as usize) {
                sum = sum + val.clone();
                count += 1;
            }
        }
    }

    if count == 0 {
        Err(NumPyError::invalid_value("no diagonal elements found"))
    } else {
        Ok(sum)
    }
}

/// Dump array to pickle file (placeholder implementation)
///
/// # Arguments
/// * `a` - Array to serialize
/// * `file` - File path to write to
///
/// # Returns
/// Ok(()) on success
///
/// # Note
/// This is a placeholder implementation. Full pickle support would require
/// serde integration and proper pickle protocol implementation.
pub fn dump<T>(a: &Array<T>, file: &Path) -> Result<()>
where
    T: Clone + Default + std::fmt::Debug + 'static,
{
    let mut f = File::create(file).map_err(|e| NumPyError::io_error(e.to_string()))?;

    // Simple text-based serialization as placeholder
    let content = format!("Array{{shape: {:?}, data: {:?}}}", a.shape(), a.to_vec());

    f.write_all(content.as_bytes())
        .map_err(|e| NumPyError::io_error(e.to_string()))?;

    Ok(())
}

/// Dump array to pickle string (placeholder implementation)
///
/// # Arguments
/// * `a` - Array to serialize
///
/// # Returns
/// Serialized string representation
///
/// # Note
/// This is a placeholder implementation. Full pickle support would require
/// serde integration and proper pickle protocol implementation.
pub fn dumps<T>(a: &Array<T>) -> Result<String>
where
    T: Clone + Default + std::fmt::Debug + 'static,
{
    // Simple text-based serialization as placeholder
    Ok(format!(
        "Array{{shape: {:?}, data: {:?}}}",
        a.shape(),
        a.to_vec()
    ))
}

/// Set array flags (placeholder implementation)
///
/// # Arguments
/// * `a` - Array to modify
/// * `write` - Whether array is writable
/// * `align` - Whether array is aligned
/// * `uic` - Whether array update-in-place is allowed
///
/// # Note
/// This is a placeholder implementation. Full flag support would require
/// maintaining flag state in the Array struct.
pub fn setflags<T>(_a: &mut Array<T>, write: bool, align: bool, uic: bool) -> Result<()>
where
    T: Clone + Default + 'static,
{
    // Placeholder: just validate parameters
    if write && align && !uic {
        return Err(NumPyError::invalid_value("incompatible flag combination"));
    }

    // TODO: Implement actual flag setting when Array struct supports flags
    Ok(())
}

/// Set values into field (placeholder implementation)
///
/// # Arguments
/// * `a` - Array to modify
/// * `field` - Field name
/// * `val` - Value to set
/// * `dtype` - Optional data type
///
/// # Note
/// This is a placeholder implementation. Full field support would require
/// structured array support.
pub fn setfield<T>(_a: &mut Array<T>, _field: &str, _val: &T, _dtype: Option<&str>) -> Result<()>
where
    T: Clone + Default + 'static,
{
    // Placeholder: structured arrays not yet implemented
    Err(NumPyError::NotImplemented {
        operation: "setfield".to_string(),
    })
}
