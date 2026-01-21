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

/// Construct an array from an index array and a list of arrays (similar to np.choose).
///
/// # Arguments
/// - `index`: Array of indices (must be same shape as choice arrays)
/// - `choices`: Slice of arrays to choose from
/// - `mode`: Mode for handling out-of-bounds indices
///   - "raise" (default): raise an error
///   - "wrap": wrap around using modulo
///   - "clip": clip to the valid range
///
/// # Returns
/// Array constructed from choices at index positions
pub fn choose<T>(index: &Array<i32>, choices: &[&Array<T>], mode: &str) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if choices.is_empty() {
        return Err(NumPyError::invalid_value(
            "need at least one choice array",
        ));
    }

    let n_choices = choices.len();

    if !matches!(mode, "raise" | "wrap" | "clip") {
        return Err(NumPyError::invalid_operation(
            "choose() mode must be 'raise', 'wrap', or 'clip'",
        ));
    }

    let index_data = index.to_vec();
    let mut result = Vec::with_capacity(index_data.len());

    for (pos, &idx) in index_data.iter().enumerate() {
        // First, compute the adjusted choice index based on mode
        let choice_idx = if idx < 0 {
            match mode {
                "raise" => {
                    return Err(NumPyError::index_error(
                        (-idx) as usize,
                        n_choices,
                    ))
                }
                "wrap" => {
                    let mut i = idx % n_choices as i32;
                    if i < 0 {
                        i += n_choices as i32;
                    }
                    i as usize
                }
                "clip" => 0,
                _ => unreachable!(),
            }
        } else if (idx as usize) >= n_choices {
            match mode {
                "raise" => {
                    return Err(NumPyError::index_error(
                        idx as usize,
                        n_choices,
                    ))
                }
                "wrap" => idx as usize % n_choices,
                "clip" => n_choices - 1,
                _ => unreachable!(),
            }
        } else {
            idx as usize
        };

        // Now determine the element index
        let choice_array = choices[choice_idx];
        let choice_data = choice_array.to_vec();
        let element_idx = if mode == "wrap" || mode == "clip" {
            // For wrap/clip modes, use the (wrapped) index value for element selection
            let idx_for_element = if mode == "wrap" {
                // Apply same wrap logic to index for element selection
                let mut i = idx % choice_data.len() as i32;
                if i < 0 {
                    i += choice_data.len() as i32;
                }
                i as usize
            } else {
                // Clip mode: clamp to array bounds
                if idx < 0 {
                    0
                } else {
                    (idx as usize).min(choice_data.len().saturating_sub(1))
                }
            };
            idx_for_element
        } else {
            // For raise mode, use the position in the index array
            pos % choice_data.len().max(1)
        };
        result.push(choice_data[element_idx].clone());
    }

    Ok(Array::from_shape_vec(index.shape().to_vec(), result))
}

/// Return selected slices of an array along given axis (similar to np.compress).
///
/// # Arguments
/// - `condition`: Boolean or integer array used to select elements
/// - `array`: Input array
/// - `axis`: Axis along which to select (None for flattened selection)
///
/// # Returns
/// Array with selected elements
pub fn compress<T>(condition: &Array<bool>, array: &Array<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let cond_data = condition.to_vec();
    let arr_data = array.to_vec();

    if let Some(ax) = axis {
        let ndim = array.ndim();
        if ndim == 0 {
            return Ok(array.clone());
        }

        let ax = if ax < 0 { ndim as isize + ax } else { ax } as usize;

        if ax >= ndim {
            return Err(NumPyError::index_error(ax, ndim));
        }

        // For 2D arrays with axis selection
        if ndim == 2 {
            let axis_len = array.shape()[ax];
            if cond_data.len() != axis_len {
                return Err(NumPyError::shape_mismatch(
                    vec![cond_data.len()],
                    vec![axis_len],
                ));
            }

            let rows = array.shape()[0];
            let cols = array.shape()[1];

            if ax == 0 {
                // Select rows
                let mut result = Vec::new();
                for (i, &keep) in cond_data.iter().enumerate() {
                    if keep && i < rows {
                        for j in 0..cols {
                            result.push(arr_data[i * cols + j].clone());
                        }
                    }
                }
                let kept_count = cond_data.iter().filter(|&&x| x).count();
                return Ok(Array::from_shape_vec(vec![kept_count, cols], result));
            } else {
                // Select columns
                let mut result = Vec::new();
                for i in 0..rows {
                    for (j, &keep) in cond_data.iter().enumerate() {
                        if keep && j < cols {
                            result.push(arr_data[i * cols + j].clone());
                        }
                    }
                }
                let kept_count = cond_data.iter().filter(|&&x| x).count();
                return Ok(Array::from_shape_vec(vec![rows, kept_count], result));
            }
        }

        // For higher dimensions, flatten and use condition
        let mut result = Vec::new();
        for (i, val) in arr_data.iter().enumerate() {
            if cond_data.get(i % cond_data.len()).copied().unwrap_or(false) {
                result.push(val.clone());
            }
        }
        return Ok(Array::from_vec(result));
    } else {
        // No axis specified - flatten and select
        if cond_data.len() != arr_data.len() {
            return Err(NumPyError::shape_mismatch(
                vec![cond_data.len()],
                vec![arr_data.len()],
            ));
        }

        let mut result = Vec::new();
        for (&cond, val) in cond_data.iter().zip(arr_data.iter()) {
            if cond {
                result.push(val.clone());
            }
        }

        Ok(Array::from_vec(result))
    }
}
