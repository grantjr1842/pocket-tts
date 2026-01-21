//! Additional array manipulation functions
//!
//! This module provides functions that are missing from the main
//! array_manipulation module but needed for NumPy compatibility.

use crate::array::Array;
use crate::error::{NumPyError, Result};

/// Concatenate arrays along an existing axis (similar to np.concatenate).
///
/// # Arguments
/// - `arrays`: Slice of arrays to concatenate
/// - `axis`: Axis along which to concatenate (default 0)
///
/// # Returns
/// Concatenated array
pub fn concatenate<T>(arrays: &[&Array<T>], axis: isize) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to concatenate",
        ));
    }

    let first = &arrays[0];
    let ndim = first.ndim();

    if ndim == 0 {
        // For 0D arrays, just flatten them all
        let mut all_data = Vec::new();
        for arr in arrays {
            all_data.extend_from_slice(&arr.to_vec());
        }
        return Ok(Array::from_vec(all_data));
    }

    let axis = if axis < 0 { ndim as isize + axis } else { axis } as usize;

    if axis >= ndim {
        return Err(NumPyError::invalid_operation(format!(
            "axis {} is out of bounds for {}-dimensional array",
            axis, ndim
        )));
    }

    // Verify all arrays have compatible shapes
    for arr in arrays {
        if arr.ndim() != ndim {
            return Err(NumPyError::shape_mismatch(vec![ndim], vec![arr.ndim()]));
        }

        let mut arr_shape = arr.shape().to_vec();
        let mut first_shape = first.shape().to_vec();

        arr_shape.remove(axis);
        first_shape.remove(axis);

        if arr_shape != first_shape {
            return Err(NumPyError::shape_mismatch(arr_shape, first_shape));
        }
    }

    // Calculate output shape
    let mut output_shape = first.shape().to_vec();
    let mut total_axis_size = 0;
    for arr in arrays {
        total_axis_size += arr.shape()[axis];
    }
    output_shape[axis] = total_axis_size;

    // Collect all data
    let mut all_data = Vec::new();
    for arr in arrays {
        all_data.extend_from_slice(&arr.to_vec());
    }

    Ok(Array::from_shape_vec(output_shape, all_data))
}

/// Stack arrays vertically (row-wise) (similar to np.vstack).
///
/// # Arguments
/// - `arrays`: Slice of arrays to stack vertically
///
/// # Returns
/// Vertically stacked array
pub fn vstack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to vstack",
        ));
    }

    let first = &arrays[0];
    let ndim = first.ndim();

    // Handle 1D arrays: treat them as row vectors
    if ndim == 1 {
        let mut rows = Vec::new();
        for arr in arrays {
            if arr.ndim() != 1 {
                return Err(NumPyError::shape_mismatch(vec![1], vec![arr.ndim()]));
            }
            rows.push((*arr).clone());
        }

        let mut total_length = 0;
        for arr in arrays {
            total_length += arr.shape()[0];
        }

        let mut all_data = Vec::new();
        for arr in arrays {
            all_data.extend_from_slice(&arr.to_vec());
        }

        return Ok(Array::from_shape_vec(
            vec![arrays.len(), total_length / arrays.len()],
            all_data,
        ));
    }

    // For nD arrays, concatenate along axis 0
    concatenate(arrays, 0)
}

/// Stack arrays horizontally (column-wise) (similar to np.hstack).
///
/// # Arguments
/// - `arrays`: Slice of arrays to stack horizontally
///
/// # Returns
/// Horizontally stacked array
pub fn hstack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to hstack",
        ));
    }

    let first = &arrays[0];
    let ndim = first.ndim();

    // Handle 1D arrays: treat them as row vectors and concatenate
    if ndim == 1 {
        let mut all_data = Vec::new();
        for arr in arrays {
            if arr.ndim() != 1 {
                return Err(NumPyError::shape_mismatch(vec![1], vec![arr.ndim()]));
            }
            all_data.extend_from_slice(&arr.to_vec());
        }
        return Ok(Array::from_vec(all_data));
    }

    // For nD arrays, concatenate along the last axis
    concatenate(arrays, (ndim - 1) as isize)
}

/// Linear interpolation (similar to np.interp).
///
/// # Arguments
/// - `x`: x-coordinates where to evaluate the interpolated values
/// - `xp`: x-coordinates of the data points
/// - `fp`: y-coordinates of the data points, same length as xp
///
/// # Returns
/// Interpolated values at x
pub fn interp<T>(x: &Array<T>, xp: &Array<T>, fp: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    if xp.shape()[0] != fp.shape()[0] {
        return Err(NumPyError::invalid_operation(
            "fp and xp must have the same size",
        ));
    }

    let x_data = x.to_vec();
    let xp_data = xp.to_vec();
    let fp_data = fp.to_vec();

    let mut result = Vec::with_capacity(x_data.len());

    for &x_val in &x_data {
        if x_val <= xp_data[0] {
            result.push(fp_data[0]);
        } else if x_val >= xp_data[xp_data.len() - 1] {
            result.push(fp_data[fp_data.len() - 1]);
        } else {
            // Find the interval
            let mut i = 0;
            while i < xp_data.len() - 1 && xp_data[i + 1] < x_val {
                i += 1;
            }

            let x0 = xp_data[i];
            let x1 = xp_data[i + 1];
            let y0 = &fp_data[i];
            let y1 = &fp_data[i + 1];

            // Linear interpolation
            let t = (x_val - x0) / (x1 - x0);
            let interpolated = *y0 + t * (*y1 - *y0);
            result.push(interpolated);
        }
    }

    Ok(Array::from_vec(result))
}

/// Element-wise power (similar to np.power).
///
/// # Arguments
/// - `array`: Input array
/// - `exponent`: Power to raise each element to
///
/// # Returns
/// Array with each element raised to the given power
pub fn power<T>(array: &Array<T>, exponent: T) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    let data: Vec<T> = array.to_vec().iter().map(|&x| x.powf(exponent)).collect();

    Ok(Array::from_vec(data))
}

/// Clip (limit) the values in an array (similar to np.clip).
///
/// # Arguments
/// - `array`: Input array
/// - `min`: Minimum value (use None for no minimum)
/// - `max`: Maximum value (use None for no maximum)
///
/// # Returns
/// Array with values clipped to the specified range
pub fn clip<T>(array: &Array<T>, min: Option<T>, max: Option<T>) -> Result<Array<T>>
where
    T: Clone + Default + PartialOrd + 'static,
{
    let data = array.to_vec();

    let clipped: Vec<T> = data
        .iter()
        .map(|x| {
            let mut val = x.clone();
            if let Some(ref min_val) = min {
                if val < *min_val {
                    val = min_val.clone();
                }
            }
            if let Some(ref max_val) = max {
                if val > *max_val {
                    val = max_val.clone();
                }
            }
            val
        })
        .collect();

    Ok(Array::from_shape_vec(array.shape().to_vec(), clipped))
}

/// Round elements of the array to the given number of decimals (similar to np.round).
///
/// Uses NumPy-compatible rounding (banker's rounding: round half to even).
///
/// # Arguments
/// - `array`: Input array
/// - `decimals`: Number of decimal places to round to (default 0)
///
/// # Returns
/// Array with values rounded to the specified number of decimals
pub fn round<T>(array: &Array<T>, decimals: i32) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    let factor = T::from(10.0_f64).unwrap().powi(decimals);

    let rounded: Vec<T> = array
        .to_vec()
        .iter()
        .map(|&x| {
            let scaled = x * factor.clone();
            let fract = scaled.fract();
            let whole_scaled = scaled.trunc();

            // Banker's rounding: round half to even
            let result = if fract.abs() == T::from(0.5_f64).unwrap() {
                // Exactly half - round to nearest even integer
                let whole_i64 = whole_scaled.to_i64().unwrap_or(0);
                if whole_i64 % 2 == 0 {
                    whole_scaled
                } else {
                    whole_scaled + T::from(whole_scaled.signum()).unwrap()
                }
            } else {
                scaled.round()
            };

            result / factor.clone()
        })
        .collect();

    Ok(Array::from_shape_vec(array.shape().to_vec(), rounded))
}
