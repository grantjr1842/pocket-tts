// Advanced broadcasting patterns for NumPy compatibility
//
// This module provides array manipulation functions for repeating,
// tiling, and broadcasting arrays to target shapes.

use crate::array::Array;
use crate::broadcasting::broadcast_to;
use crate::error::{NumPyError, Result};

/// Repeat array along a given axis
///
/// Arguments:
/// - a: Input array
/// - repeats: Number of times to repeat each element
/// - axis: Axis along which to repeat (0 means flatten and repeat, 1 means repeat along first axis, etc.)
///
/// Returns: Array with repeated elements
///
/// Examples:
/// ```rust,ignore
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let result = repeat(&a, 2, Some(0)).unwrap();
/// // result == [1, 1, 2, 2, 2, 3, 3]
/// ```
pub fn repeat<T>(a: &Array<T>, repeats: usize, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot repeat empty array"));
    }

    if repeats == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    if axis.is_none() {
        // Flatten and repeat
        let mut result_data = Vec::with_capacity(a.size() * repeats);
        for elem in a.iter() {
            for _ in 0..repeats {
                result_data.push(elem.clone());
            }
        }
        return Ok(Array::from_data(result_data, vec![a.size() * repeats]));
    }

    let ndim = a.ndim();
    let ax = if ndim == 0 {
        0
    } else {
        let ax = axis.unwrap();
        if ax < 0 {
            (ax + ndim as isize) as usize
        } else {
            ax as usize
        }
    };

    if ax >= ndim && ndim > 0 {
        return Err(NumPyError::invalid_value(format!(
            "axis {} out of bounds for array with {} dimensions",
            ax,
            a.ndim()
        )));
    }

    // Correct NumPy repeat behavior:
    // 1. Reshape to add a new axis of size 1 at ax + 1
    // 2. Broadcast that axis to 'repeats'
    // 3. Reshape back to merge them

    let mut expanded_shape = a.shape().to_vec();
    expanded_shape.insert(ax + 1, 1);

    // We need a proper view-based reshape or just do it manually if reshape usually copies.
    // In our case, Array::reshape copies if it can't just change shape/strides.
    // Let's use the property that we can manually create the expanded view.

    let mut new_strides = a.strides().to_vec();
    new_strides.insert(ax + 1, 0); // 0-stride for the new dimension

    let expanded_view = Array {
        data: a.data.clone(),
        shape: expanded_shape,
        strides: new_strides,
        dtype: a.dtype.clone(),
        offset: a.offset,
    };

    let mut broadcast_shape = expanded_view.shape().to_vec();
    broadcast_shape[ax + 1] = repeats;

    let broadcasted = broadcast_to(&expanded_view, &broadcast_shape)?;

    let mut final_shape = a.shape().to_vec();
    final_shape[ax] *= repeats;

    // Reshape the broadcasted view into the final shape (this will copy)
    broadcasted.reshape(&final_shape)
}

/// Tile array by repeating it
///
/// Arguments:
/// - a: Input array
/// - reps: Number of repetitions per dimension
///
/// Returns: Tiled array
///
/// Examples:
/// ```rust,ignore
/// let a = Array::from_vec(vec![1, 2]);
/// let result = tile(&a, &[3, 2]).unwrap();
/// // result shape is [3, 2, 2] with 6 elements
/// ```
pub fn tile<T>(a: &Array<T>, reps: &[usize]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    if a.is_empty() {
        return Err(NumPyError::invalid_value("Cannot tile empty array"));
    }

    if reps.is_empty() {
        return Ok(a.clone());
    }

    let a_ndim = a.ndim();
    let r_ndim = reps.len();
    let ndim = a_ndim.max(r_ndim);

    // 1. Pad reps with 1s at origin if needed
    let mut full_reps = vec![1; ndim];
    let offset = ndim - r_ndim;
    for i in 0..r_ndim {
        full_reps[offset + i] = reps[i];
    }

    // 2. Pad array shape with 1s at origin if needed
    let mut a_shape_padded = vec![1; ndim];
    let a_offset = ndim - a_ndim;
    for i in 0..a_ndim {
        a_shape_padded[a_offset + i] = a.shape()[i];
    }

    // 3. Reshape A to (1, d1, 1, d2, ...)
    let mut expanded_shape = Vec::with_capacity(ndim * 2);
    let mut broadcast_shape = Vec::with_capacity(ndim * 2);
    let mut final_shape = Vec::with_capacity(ndim);

    for i in 0..ndim {
        expanded_shape.push(1);
        expanded_shape.push(a_shape_padded[i]);

        broadcast_shape.push(full_reps[i]);
        broadcast_shape.push(a_shape_padded[i]);

        final_shape.push(full_reps[i] * a_shape_padded[i]);
    }

    // Create the expanded view manually (to avoid multiple copies)
    // Strides for expanded_shape [1, d1, 1, d2, ...]
    // The '1' dimensions get 0-stride if we were broadcasting, but here we will broadcast later.
    // We can just use add_newaxis iteratively or just compute strides.

    let mut current = a.clone();
    // Pad current to ndim dimensions
    for _ in 0..a_offset {
        current = current.add_newaxis(0)?;
    }

    // Now current has shape (d1, d2, ..., dn) and ndim dimensions.
    // Reshape to (1, d1, 1, d2, ...)
    let mut reshaped_strides = Vec::with_capacity(ndim * 2);
    for i in 0..ndim {
        reshaped_strides.push(0); // For the '1' in (1, di)
        reshaped_strides.push(current.strides()[i]);
    }

    let expanded_view = Array {
        data: current.data.clone(),
        shape: expanded_shape,
        strides: reshaped_strides,
        dtype: current.dtype.clone(),
        offset: current.offset,
    };

    let broadcasted = broadcast_to(&expanded_view, &broadcast_shape)?;
    broadcasted.reshape(&final_shape)
}

/// Broadcast array to specific shape
///
/// This function extends the broadcast_to functionality to support
/// arbitrary target shapes, similar to numpy.broadcast_to.
///
/// Arguments:
/// - array: Input array
/// - shape: Target shape to broadcast to
///
/// Returns: Broadcasted array
///
/// Examples:
/// ```rust,ignore
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let result = broadcast_to(&a, &[3, 4]).unwrap();
/// // result shape is [3, 4] with repeated pattern
/// ```
pub fn broadcast_to_enhanced<T>(array: &Array<T>, shape: &[usize]) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    // Delegate to existing broadcast_to function
    // This is a convenience wrapper with NumPy-compatible name
    broadcast_to(array, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_axis_0() {
        let a = Array::from_vec(vec![1i64, 2, 3]);
        let result = repeat(&a, 2, Some(0)).unwrap();
        assert_eq!(result.shape(), vec![6]);
        assert_eq!(result.to_vec(), vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn test_repeat_axis_none() {
        let a = Array::from_vec(vec![1i64, 2, 3]);
        let result = repeat(&a, 2, None).unwrap();
        assert_eq!(result.shape(), vec![6]);
        assert_eq!(result.to_vec(), vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn test_tile_basic() {
        let a = Array::from_vec(vec![1i64, 2]);
        let result = tile(&a, &[2]).unwrap();
        assert_eq!(result.shape(), vec![4]);
        assert_eq!(result.to_vec(), vec![1, 2, 1, 2]);

        let result2 = tile(&a, &[2, 1]).unwrap();
        assert_eq!(result2.shape(), vec![2, 2]);
        assert_eq!(result2.to_vec(), vec![1, 2, 1, 2]);
    }

    #[test]
    fn test_broadcast_to_enhanced() {
        let a = Array::from_vec(vec![1i64]);
        let result = broadcast_to_enhanced(&a, &[3, 4]).unwrap();
        assert_eq!(result.shape(), vec![3, 4]);
    }
}
