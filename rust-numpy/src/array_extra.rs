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

/// Extract a diagonal from a 2D array (similar to np.diagonal).
///
/// # Arguments
/// - `array`: Input array (must be at least 2D)
/// - `offset`: Diagonal offset (default 0)
///   - offset > 0: upper diagonals
///   - offset < 0: lower diagonals
///   - offset = 0: main diagonal
/// - `axis1`: First axis of diagonal (default 0)
/// - `axis2`: Second axis of diagonal (default 1)
///
/// # Returns
/// 1D array containing the specified diagonal
pub fn diagonal<T>(
    array: &Array<T>,
    offset: isize,
    axis1: usize,
    axis2: usize,
) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let ndim = array.ndim();

    if ndim < 2 {
        return Err(NumPyError::invalid_operation(
            "diagonal() requires array with at least 2 dimensions",
        ));
    }

    if axis1 >= ndim || axis2 >= ndim {
        return Err(NumPyError::index_error(axis1.max(axis2), ndim));
    }

    if axis1 == axis2 {
        return Err(NumPyError::invalid_operation(
            "diagonal() requires axis1 and axis2 to be different",
        ));
    }

    // For 2D arrays, extract the diagonal directly
    if ndim == 2 {
        let rows = array.shape()[0];
        let cols = array.shape()[1];
        let data = array.to_vec();

        let mut diagonal_elements = Vec::new();

        // Calculate diagonal bounds based on offset
        let (row_start, col_start): (isize, isize) = if offset >= 0 {
            (0, offset)
        } else {
            (-offset, 0)
        };

        let row_start = row_start as usize;
        let col_start = col_start as usize;

        // Iterate along the diagonal
        let mut i = row_start;
        let mut j = col_start;

        while i < rows && j < cols {
            let idx = i * cols + j;
            if idx < data.len() {
                diagonal_elements.push(data[idx].clone());
            }
            i += 1;
            j += 1;
        }

        return Ok(Array::from_vec(diagonal_elements));
    }

    // For nD arrays, use axis1 and axis2
    let mut diagonal_shape = array.shape().to_vec();
    diagonal_shape.remove(axis1.max(axis2));
    diagonal_shape.remove(axis1.min(axis2));

    let dim1_size = array.shape()[axis1];
    let dim2_size = array.shape()[axis2];

    let mut diagonal_elements = Vec::new();

    let (start1, start2): (isize, isize) = if offset >= 0 {
        (0, offset)
    } else {
        (-offset, 0)
    };

    let start1 = start1 as usize;
    let start2 = start2 as usize;

    // Collect diagonal elements
    let mut d1 = start1;
    let mut d2 = start2;

    while d1 < dim1_size && d2 < dim2_size {
        let mut indices = vec![0; ndim];
        indices[axis1] = d1;
        indices[axis2] = d2;

        // Collect all elements for this diagonal position across other dimensions
        collect_diagonal_elements(array, &mut indices, axis1.min(axis2), 0, &mut diagonal_elements);

        d1 += 1;
        d2 += 1;
    }

    let final_shape = if diagonal_shape.is_empty() {
        vec![1]
    } else {
        diagonal_shape
    };

    Ok(Array::from_shape_vec(final_shape, diagonal_elements))
}

/// Helper function to collect elements along diagonal for nD arrays
fn collect_diagonal_elements<T>(
    array: &Array<T>,
    indices: &mut [usize],
    skip_axis: usize,
    current_axis: usize,
    result: &mut Vec<T>,
) where
    T: Clone + Default + 'static,
{
    if current_axis == indices.len() {
        let linear_idx = compute_linear_index_from_indices(array, indices);
        if let Some(elem) = array.get_linear(linear_idx) {
            result.push(elem.clone());
        }
        return;
    }

    if current_axis == skip_axis {
        collect_diagonal_elements(array, indices, skip_axis, current_axis + 1, result);
    } else {
        let dim_size = array.shape()[current_axis];
        for i in 0..dim_size {
            indices[current_axis] = i;
            collect_diagonal_elements(array, indices, skip_axis, current_axis + 1, result);
        }
    }
}

/// Compute linear index from multi-dimensional indices
fn compute_linear_index_from_indices<T>(array: &Array<T>, indices: &[usize]) -> usize
where
    T: Clone + Default + 'static,
{
    let strides = array.strides();
    let offset = array.offset;

    let mut linear_idx: isize = offset as isize;
    for (i, &idx) in indices.iter().enumerate() {
        linear_idx += strides[i] * idx as isize;
    }

    linear_idx as usize
}

/// Extract a diagonal or construct a diagonal array (similar to np.diag).
///
/// # Arguments
/// - `v`: Input array
///   - If 2D: extracts the k-th diagonal
///   - If 1D: constructs a 2D array with v on the k-th diagonal
/// - `k`: Diagonal offset (default 0)
///   - k > 0: upper diagonals
///   - k < 0: lower diagonals
///   - k = 0: main diagonal
///
/// # Returns
/// - If input is 2D: 1D array containing the k-th diagonal
/// - If input is 1D: 2D array with input on the k-th diagonal
pub fn diag<T>(v: &Array<T>, k: isize) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let ndim = v.ndim();

    if ndim == 1 {
        // Construct 2D array with v on the k-th diagonal
        let n = v.shape()[0];
        let data = v.to_vec();

        let size = if k >= 0 {
            n + k as usize
        } else {
            n + (-k) as usize
        };

        let mut matrix = vec![T::default(); size * size];

        for (i, val) in data.iter().enumerate() {
            let (row, col) = if k >= 0 {
                (i, i + k as usize)
            } else {
                (i + (-k) as usize, i)
            };
            if row < size && col < size {
                matrix[row * size + col] = val.clone();
            }
        }

        Ok(Array::from_shape_vec(vec![size, size], matrix))
    } else if ndim == 2 {
        // Extract the k-th diagonal from 2D array
        diagonal(v, k, 0, 1)
    } else {
        Err(NumPyError::invalid_operation(
            "diag() requires 1D or 2D array",
        ))
    }
}
