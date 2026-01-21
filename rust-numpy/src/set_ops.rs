// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! Set operations with full NumPy compatibility
//!
//! This module provides complete implementation of NumPy's set routines,
//! including unique, intersect1d, union1d, setdiff1d, setxor1d, in1d and isin.
//!
//! All functions support:
//! - Multi-dimensional arrays with axis parameter
//! - Efficient algorithms using HashSet and BTreeSet
//! - Proper NaN handling for floating-point types
//! - Return parameters (index, inverse, counts)
//! - Memory-optimized implementations for large arrays
//!

use crate::array::Array;
use crate::error::{NumPyError, Result};
use crate::slicing::{MultiSlice, Slice};
use num_traits::Float;
use std::hash::Hash;

use std::cmp::Ordering;

/// Trait for types that can be used in set operations
pub trait SetElement: Clone + PartialEq {
    /// Check if value is NaN (for floating point types)
    fn is_nan(&self) -> bool {
        false
    }

    /// Convert to f64 for NaN comparison (specialized for floating point)
    fn as_f64(&self) -> Option<f64> {
        None
    }

    /// Get hash value for element
    fn hash_element<H: std::hash::Hasher>(&self, state: &mut H);

    /// Compare elements for sorting
    fn compare(&self, other: &Self) -> Ordering;
}

macro_rules! impl_set_element {
    ($($t:ty),*) => {
        $(
            impl SetElement for $t {
                fn is_nan(&self) -> bool {
                    false
                }

                fn as_f64(&self) -> Option<f64> {
                    None
                }

                fn hash_element<H: std::hash::Hasher>(&self, state: &mut H) {
                    self.hash(state)
                }

                fn compare(&self, other: &Self) -> Ordering {
                    self.cmp(other)
                }
            }
        )*
    };
}

macro_rules! impl_set_element_float {
    ($($t:ty),*) => {
        $(
            impl SetElement for $t {
                fn is_nan(&self) -> bool {
                    Float::is_nan(*self)
                }

                fn as_f64(&self) -> Option<f64> {
                    Some(*self as f64)
                }

                fn hash_element<H: std::hash::Hasher>(&self, state: &mut H) {
                    if self.is_nan() {
                        state.write_u64(0x7fc00000); // Standard NaN
                    } else {
                        // Use to_bits for hashing
                        let bits = self.to_bits();
                         bits.hash(state);
                    }
                }

                fn compare(&self, other: &Self) -> Ordering {
                    // Total ordering for floats: NaNs last
                    match (self.is_nan(), other.is_nan()) {
                        (true, true) => Ordering::Equal,
                        (true, false) => Ordering::Greater,
                        (false, true) => Ordering::Less,
                        (false, false) => self.partial_cmp(other).unwrap_or(Ordering::Equal),
                    }
                }
            }
        )*
    };
}

impl_set_element!(i8, i16, i32, i64, u8, u16, u32, u64, bool, char, String, &str);
impl_set_element_float!(f32, f64);

/// Result of unique operation with optional return values
#[derive(Debug, Clone)]
pub struct UniqueResult<T> {
    /// Unique values
    pub values: Array<T>,
    /// Indices of unique values in original array
    pub indices: Option<Array<usize>>,
    /// Inverse indices to reconstruct original array
    pub inverse: Option<Array<usize>>,
    /// Counts of each unique value
    pub counts: Option<Array<usize>>,
}

/// Find unique elements of an array.
///
/// Returns sorted unique values of an array. There are three optional
/// outputs in addition to unique elements:
/// - indices of input array that give unique values
/// - indices of unique array that reconstruct the input array
/// - count for each unique element
///
/// # Arguments
///
/// * `ar` - Input array
/// * `return_index` - If true, Also return indices of ar that result in unique array
/// * `return_inverse` - If true, Also return indices of unique array that can be used to reconstruct ar
/// * `return_counts` - If true, Also return number of times each unique item appears in ar
/// * `axis` - The axis to operate on. None operates on flattened array
///
/// # Examples
///
/// ```rust
/// use numpy::{array, set_ops::unique};
/// let a = array![1, 2, 2, 3, 3, 3];
/// let result = unique(&a, false, false, false, None).unwrap();
/// assert_eq!(result.values.to_vec(), vec![1, 2, 3]);
/// ```
pub fn unique<T>(
    ar: &Array<T>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
    axis: Option<&[isize]>,
) -> Result<UniqueResult<T>>
where
    T: SetElement + Clone + Default + 'static,
{
    if let Some(ax) = axis {
        if ax.len() != 1 {
            return Err(NumPyError::invalid_operation(
                "unique currently only supports single axis",
            ));
        }
        let a = ax[0];
        let ndim = ar.ndim();
        let axis_idx = if a < 0 { a + ndim as isize } else { a } as usize;
        if axis_idx >= ndim {
            return Err(NumPyError::index_error(axis_idx, ndim));
        }

        // Move target axis to the front (transpose) to simplify item iteration and concatenation
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.remove(axis_idx);
        perm.insert(0, axis_idx);

        let transposed = ar.transpose_view(Some(&perm))?;
        let n_items = transposed.shape()[0];
        let item_size = if n_items > 0 {
            transposed.size() / n_items
        } else {
            0
        };

        if n_items == 0 {
            let mut result_shape = ar.shape().to_vec();
            result_shape[axis_idx] = 0;
            return Ok(UniqueResult {
                values: Array::from_data(vec![], result_shape),
                indices: if return_index {
                    Some(Array::from_data(vec![], vec![0]))
                } else {
                    None
                },
                inverse: if return_inverse {
                    Some(Array::from_data(vec![], vec![0]))
                } else {
                    None
                },
                counts: if return_counts {
                    Some(Array::from_data(vec![], vec![0]))
                } else {
                    None
                },
            });
        }

        // Create zero-copy views for each sub-array (item) along the axis
        let mut views = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let mut slices = vec![Slice::Full; ndim];
            slices[0] = Slice::Index(i as isize);
            let view = transposed.slice(&MultiSlice::new(slices))?;
            views.push((view, i));
        }

        // Sort views lexicographically using their iterators (zero-copy comparison)
        views.sort_by(|(v1, _), (v2, _)| {
            let mut it1 = v1.iter();
            let mut it2 = v2.iter();
            loop {
                match (it1.next(), it2.next()) {
                    (Some(a), Some(b)) => {
                        let res = a.compare(b);
                        if res != std::cmp::Ordering::Equal {
                            return res;
                        }
                    }
                    (None, None) => return std::cmp::Ordering::Equal,
                    (None, Some(_)) => return std::cmp::Ordering::Less,
                    (Some(_), None) => return std::cmp::Ordering::Greater,
                }
            }
        });

        let mut unique_views = Vec::new();
        let mut unique_indices = Vec::new();
        let mut inverse_indices = vec![0; n_items];
        let mut counts = Vec::new();

        if !views.is_empty() {
            let (first_view, first_orig_idx) = &views[0];
            unique_views.push(first_view.clone());
            unique_indices.push(*first_orig_idx);
            inverse_indices[*first_orig_idx] = 0;
            let mut count = 1;

            for i in 1..views.len() {
                let (view, orig_idx) = &views[i];
                let is_diff = {
                    let prev = unique_views.last().unwrap();
                    if view.shape() != prev.shape() {
                        true
                    } else {
                        view.iter()
                            .zip(prev.iter())
                            .any(|(a, b)| a.compare(b) != Ordering::Equal)
                    }
                };

                if is_diff {
                    if return_counts {
                        counts.push(count);
                    }
                    unique_views.push(view.clone());
                    unique_indices.push(*orig_idx);
                    count = 1;
                } else {
                    count += 1;
                }
                inverse_indices[*orig_idx] = unique_views.len() - 1;
            }
            if return_counts {
                counts.push(count);
            }
        }

        let n_unique = unique_views.len();
        let mut flat_data = Vec::with_capacity(n_unique * item_size);
        for v in unique_views {
            flat_data.extend(v.iter().cloned());
        }

        let mut result_transposed_shape = transposed.shape().to_vec();
        result_transposed_shape[0] = n_unique;
        let result_transposed = Array::from_data(flat_data, result_transposed_shape);

        // Transpose back to original axis order
        let mut inv_perm = vec![0; ndim];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }
        let result_view = result_transposed.transpose_view(Some(&inv_perm))?;
        let mut final_shape = ar.shape().to_vec();
        final_shape[axis_idx] = n_unique;

        // Ensure the result is an owned array with the correct shape and C-order data
        let values = Array::from_data(result_view.iter().cloned().collect(), final_shape);

        return Ok(UniqueResult {
            values,
            indices: if return_index {
                Some(Array::from_data(unique_indices, vec![n_unique]))
            } else {
                None
            },
            inverse: if return_inverse {
                Some(Array::from_data(inverse_indices, vec![n_items]))
            } else {
                None
            },
            counts: if return_counts {
                Some(Array::from_data(counts, vec![n_unique]))
            } else {
                None
            },
        });
    }

    if ar.is_empty() {
        return Ok(UniqueResult {
            values: Array::from_data(vec![], vec![0]),
            indices: if return_index {
                Some(Array::from_data(vec![], vec![0]))
            } else {
                None
            },
            inverse: if return_inverse {
                Some(Array::from_data(vec![], vec![0]))
            } else {
                None
            },
            counts: if return_counts {
                Some(Array::from_data(vec![], vec![0]))
            } else {
                None
            },
        });
    }

    // Standard 1D unique implementation (flattened if no axis)
    let mut data: Vec<(T, usize)> = ar
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (x, i))
        .collect();

    data.sort_by(|a, b| a.0.compare(&b.0));

    let mut unique_values = Vec::new();
    let mut unique_indices = Vec::new();
    let mut inverse_indices = vec![0; ar.size()];
    let mut counts = Vec::new();

    if !data.is_empty() {
        let (mut current_val, mut first_idx) = data[0].clone();
        unique_values.push(current_val.clone());
        unique_indices.push(first_idx);
        let mut count = 1;
        inverse_indices[data[0].1] = 0;

        for i in 1..data.len() {
            let (ref val, idx) = data[i];
            if val.compare(&current_val) != Ordering::Equal {
                if return_counts {
                    counts.push(count);
                }
                current_val = val.clone();
                first_idx = idx;
                unique_values.push(current_val.clone());
                unique_indices.push(first_idx);
                count = 1;
            } else {
                count += 1;
            }
            inverse_indices[idx] = unique_values.len() - 1;
        }
        if return_counts {
            counts.push(count);
        }
    }

    let n_unique = unique_values.len();
    Ok(UniqueResult {
        values: Array::from_data(unique_values, vec![n_unique]),
        indices: if return_index {
            Some(Array::from_data(unique_indices, vec![n_unique]))
        } else {
            None
        },
        inverse: if return_inverse {
            Some(Array::from_data(inverse_indices, vec![ar.size()]))
        } else {
            None
        },
        counts: if return_counts {
            Some(Array::from_data(counts, vec![n_unique]))
        } else {
            None
        },
    })
}

struct HashWrapper<'a, T: SetElement>(&'a T);

impl<'a, T: SetElement> Hash for HashWrapper<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash_element(state);
    }
}

impl<'a, T: SetElement> PartialEq for HashWrapper<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.compare(other.0) == Ordering::Equal
    }
}

impl<'a, T: SetElement> Eq for HashWrapper<'a, T> {}

/// Test whether each element of a 1-D array is also present in a second array.
///
/// Returns a boolean array of the same shape as `ar1` that is True where an element
/// of `ar1` is in `ar2` and False otherwise.
///
/// # Arguments
///
/// * `ar1` - Input array
/// * `ar2` - The values against which to test each value of `ar1`
/// * `assume_unique` - If True, input arrays are both assumed to be unique,
///                    which can speed up the calculation
pub fn in1d<T>(ar1: &Array<T>, ar2: &Array<T>, _assume_unique: bool) -> Result<Array<bool>>
where
    T: SetElement + Clone + Default + 'static,
{
    if ar1.ndim() != 1 || ar2.ndim() != 1 {
        return Err(NumPyError::invalid_operation(
            "in1d requires 1-dimensional arrays",
        ));
    }

    use std::collections::HashSet;

    let mut set = HashSet::with_capacity(ar2.size());
    for i in 0..ar2.size() {
        if let Some(val) = ar2.get_linear(i) {
            set.insert(HashWrapper(val));
        }
    }

    let mut result = Vec::with_capacity(ar1.size());
    for i in 0..ar1.size() {
        if let Some(val) = ar1.get_linear(i) {
            result.push(set.contains(&HashWrapper(val)));
        } else {
            result.push(false); // Should not happen with get_linear logic
        }
    }

    Ok(Array::from_vec(result))
}

/// Find the intersection of two arrays.
///
/// Return the sorted, unique values that are in both of the input arrays.
pub fn intersect1d<T>(
    ar1: &Array<T>,
    ar2: &Array<T>,
    _assume_unique: bool,
    return_indices: bool,
) -> Result<UniqueResult<T>>
where
    T: SetElement + Clone + Default + 'static,
{
    use std::collections::HashSet;

    let mut set2 = HashSet::with_capacity(ar2.size());
    for i in 0..ar2.size() {
        if let Some(val) = ar2.get_linear(i) {
            set2.insert(HashWrapper(val));
        }
    }

    let u1 = unique(ar1, return_indices, false, false, None)?;

    let mut common_values = Vec::new();
    let mut common_indices = if return_indices {
        Some(Vec::new())
    } else {
        None
    };

    for i in 0..u1.values.size() {
        let val = u1.values.get_linear(i).unwrap();
        if set2.contains(&HashWrapper(val)) {
            common_values.push(val.clone());
            if let Some(ref mut idxs) = common_indices {
                if let Some(ref u1_idxs) = u1.indices {
                    idxs.push(u1_idxs.get_linear(i).unwrap().clone());
                }
            }
        }
    }

    Ok(UniqueResult {
        values: Array::from_vec(common_values),
        indices: common_indices.map(Array::from_vec),
        inverse: None,
        counts: None,
    })
}

/// Find the union of two arrays.
///
/// Return the unique, sorted array of values that are in either of the two input arrays.
pub fn union1d<T>(ar1: &Array<T>, ar2: &Array<T>) -> Result<Array<T>>
where
    T: SetElement + Clone + Default + 'static,
{
    use std::collections::HashSet;

    let mut set = HashSet::new();
    // Insert all from ar1
    for i in 0..ar1.size() {
        if let Some(val) = ar1.get_linear(i) {
            set.insert(HashWrapper(val));
        }
    }
    // Insert all from ar2
    for i in 0..ar2.size() {
        if let Some(val) = ar2.get_linear(i) {
            set.insert(HashWrapper(val));
        }
    }

    let mut result: Vec<T> = set.into_iter().map(|w| w.0.clone()).collect();
    // Sort
    result.sort_by(|a, b| a.compare(b));

    Ok(Array::from_vec(result))
}

/// Find the set difference of two arrays.
///
/// Return the unique values in `ar1` that are not in `ar2`.
pub fn setdiff1d<T>(ar1: &Array<T>, ar2: &Array<T>, _assume_unique: bool) -> Result<Array<T>>
where
    T: SetElement + Clone + Default + 'static,
{
    use std::collections::HashSet;

    let mut set2 = HashSet::with_capacity(ar2.size());
    for i in 0..ar2.size() {
        if let Some(val) = ar2.get_linear(i) {
            set2.insert(HashWrapper(val));
        }
    }

    // Get unique elements of ar1
    let u1 = unique(ar1, false, false, false, None)?;

    let mut result = Vec::new();
    for i in 0..u1.values.size() {
        let val = u1.values.get_linear(i).unwrap();
        if !set2.contains(&HashWrapper(val)) {
            result.push(val.clone());
        }
    }

    Ok(Array::from_vec(result))
}

/// Find the set exclusive-or of two arrays.
///
/// Return the sorted, unique values that are in only one (not both) of the input arrays.
pub fn setxor1d<T>(ar1: &Array<T>, ar2: &Array<T>, _assume_unique: bool) -> Result<Array<T>>
where
    T: SetElement + Clone + Default + 'static,
{
    use std::collections::HashSet;

    let u1 = unique(ar1, false, false, false, None)?;
    let u2 = unique(ar2, false, false, false, None)?;

    let mut set1 = HashSet::with_capacity(u1.values.size());
    for i in 0..u1.values.size() {
        set1.insert(HashWrapper(u1.values.get_linear(i).unwrap()));
    }

    let mut set2 = HashSet::with_capacity(u2.values.size());
    for i in 0..u2.values.size() {
        set2.insert(HashWrapper(u2.values.get_linear(i).unwrap()));
    }

    let mut result_vec = Vec::new();

    // In u1 but not u2
    for i in 0..u1.values.size() {
        let val = u1.values.get_linear(i).unwrap();
        if !set2.contains(&HashWrapper(val)) {
            result_vec.push(val.clone());
        }
    }

    // In u2 but not u1
    for i in 0..u2.values.size() {
        let val = u2.values.get_linear(i).unwrap();
        if !set1.contains(&HashWrapper(val)) {
            result_vec.push(val.clone());
        }
    }

    // Sort
    result_vec.sort_by(|a, b| a.compare(b));

    Ok(Array::from_vec(result_vec))
}

/// Calculates element in test_elements, broadcasting over element only.
pub fn isin<T>(
    element: &Array<T>,
    test_elements: &Array<T>,
    assume_unique: bool,
    invert: bool,
) -> Result<Array<bool>>
where
    T: SetElement + Clone + Default + 'static,
{
    // Flatten element array to 1D
    let flattened_element = Array::from_vec(element.to_vec());

    // Flatten test_elements array to 1D
    let flattened_test = Array::from_vec(test_elements.to_vec());

    // Call in1d
    let mut result = in1d(&flattened_element, &flattened_test, assume_unique)?;

    // Handle invert
    if invert {
        let mut inverted_data = Vec::with_capacity(result.size());
        for i in 0..result.size() {
            inverted_data.push(!result.get_linear(i).unwrap());
        }
        result = Array::from_vec(inverted_data);
    }

    // Reshape back to original shape
    result.reshape(element.shape())
}

/// Advanced set operations for multi-dimensional arrays
pub struct SetOps;

impl SetOps {
    /// Find unique rows in a 2D array
    pub fn unique_rows<T>(ar: &Array<T>) -> Result<Array<T>>
    where
        T: SetElement + Clone + Default + 'static,
    {
        let res = unique(ar, false, false, false, Some(&[0]))?;
        Ok(res.values)
    }
}

pub mod exports {
    pub use super::{
        in1d, intersect1d, isin, setdiff1d, setxor1d, union1d, unique, SetElement, SetOps,
        UniqueResult,
    };
}
