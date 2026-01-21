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

    let output_strides = crate::array::compute_strides(&output_shape);
    let output_size = output_shape.iter().product();
    let mut new_data = vec![T::default(); output_size];

    let mut current_axis_offset = 0;

    for arr in arrays {
        let arr_size = arr.size();
        let arr_shape = arr.shape();

        for i in 0..arr_size {
            if let Some(val) = arr.get_linear(i) {
                let mut coords = crate::strides::compute_multi_indices(i, arr_shape);
                coords[axis] += current_axis_offset;
                let out_linear = crate::strides::compute_linear_index(&coords, &output_strides);
                new_data[out_linear as usize] = val.clone();
            }
        }
        current_axis_offset += arr.shape()[axis];
    }

    Ok(Array::from_shape_vec(output_shape, new_data))
}

/// Stack arrays along a new axis.
pub fn stack<T>(arrays: &[&Array<T>], axis: isize) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to stack",
        ));
    }
    let first = arrays[0];
    let ndim = first.ndim();
    let shape = first.shape();

    for arr in arrays {
        if arr.shape() != shape {
            return Err(NumPyError::shape_mismatch(
                shape.to_vec(),
                arr.shape().to_vec(),
            ));
        }
    }

    // Normalize axis. The output has ndim + 1 dimensions.
    // For 0D arrays, ndim=0, output is 1D. Axis must be 0.
    let axis_limit = ndim + 1;
    let axis = if axis < 0 {
        axis + axis_limit as isize
    } else {
        axis
    };
    if axis < 0 || axis >= axis_limit as isize {
        return Err(NumPyError::invalid_operation(format!(
            "axis {} out of bounds",
            axis
        )));
    }
    let axis = axis as usize;

    let mut new_shape = shape.to_vec();
    new_shape.insert(axis, 1);

    let mut reshaped_arrays = Vec::with_capacity(arrays.len());
    for arr in arrays {
        reshaped_arrays.push(arr.reshape(&new_shape)?);
    }

    let refs: Vec<&Array<T>> = reshaped_arrays.iter().collect();
    concatenate(&refs, axis as isize)
}

/// Vertically stack arrays (row-wise).
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

    concatenate(arrays, 0)
}

/// Horizontally stack arrays (column-wise).
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

    concatenate(arrays, (ndim - 1) as isize)
}

/// Stack arrays in depth wise (along third axis).
pub fn dstack<T>(arrays: &[&Array<T>]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one array to dstack",
        ));
    }

    let mut reshaped_arrays = Vec::with_capacity(arrays.len());

    for arr in arrays {
        let ndim = arr.ndim();
        if ndim <= 2 {
            let shape = arr.shape();
            let mut new_shape = vec![1, 1, 1];
            if ndim == 1 {
                // (N,) -> (1, N, 1)
                new_shape[0] = 1;
                new_shape[1] = shape[0];
                new_shape[2] = 1;
            } else {
                // (M, N) -> (M, N, 1)
                new_shape[0] = shape[0];
                new_shape[1] = shape[1];
                new_shape[2] = 1;
            }
            reshaped_arrays.push(arr.reshape(&new_shape)?);
        } else {
            reshaped_arrays.push((*arr).clone());
        }
    }

    let refs: Vec<&Array<T>> = reshaped_arrays.iter().collect();
    concatenate(&refs, 2)
}

/// Linear interpolation.
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
            let mut i = 0;
            while i < xp_data.len() - 1 && xp_data[i + 1] < x_val {
                i += 1;
            }
            let x0 = xp_data[i];
            let x1 = xp_data[i + 1];
            let y0 = &fp_data[i];
            let y1 = &fp_data[i + 1];
            let t = (x_val - x0) / (x1 - x0);
            let interpolated = *y0 + t * (*y1 - *y0);
            result.push(interpolated);
        }
    }
    Ok(Array::from_vec(result))
}

/// Element-wise power.
pub fn power<T>(array: &Array<T>, exponent: T) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Float + 'static,
{
    let data: Vec<T> = array.to_vec().iter().map(|&x| x.powf(exponent)).collect();
    Ok(Array::from_vec(data))
}
