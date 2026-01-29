// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! NumPy-compatible masked array implementation
//!
//! This module provides `MaskedArray` and related functions for handling arrays
//! with missing or invalid data, similar to NumPy's `numpy.ma` module.

use crate::array::Array;
use crate::dtype::Casting;
use crate::error::{NumPyError, Result};
use std::fmt::Debug;

/// A MaskedArray is an array that may have missing or invalid entries.
/// It consists of a data array and a boolean mask of the same shape.
/// Elements where the mask is `true` are considered invalid/masked.
#[derive(Debug, Clone)]
pub struct MaskedArray<T> {
    data: Array<T>,
    mask: Array<bool>,
    fill_value: Option<T>,
    _hardmask: bool,
}

impl<T> MaskedArray<T>
where
    T: Clone + Debug + Default + 'static,
{
    /// Create a new MaskedArray from data and mask.
    pub fn new(data: Array<T>, mask: Array<bool>) -> Result<Self> {
        if data.shape() != mask.shape() {
            return Err(NumPyError::shape_mismatch(
                data.shape().to_vec(),
                mask.shape().to_vec(),
            ));
        }
        Ok(Self {
            data,
            mask,
            fill_value: None,
            _hardmask: false,
        })
    }

    /// Create a new MaskedArray from data, with no elements masked.
    pub fn from_data(data: Array<T>) -> Self {
        let mask = Array::from_data(vec![false; data.size()], data.shape().to_vec());
        Self {
            data,
            mask,
            fill_value: None,
            _hardmask: false,
        }
    }

    /// Return the data array.
    pub fn data(&self) -> &Array<T> {
        &self.data
    }

    /// Return the mask array.
    pub fn mask(&self) -> &Array<bool> {
        &self.mask
    }

    /// Return the shape of the masked array.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Return the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Return the total number of elements.
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Set the fill value.
    pub fn set_fill_value(&mut self, value: T) {
        self.fill_value = Some(value);
    }

    /// Get the fill value, or a default if not set.
    pub fn fill_value(&self) -> T {
        self.fill_value.clone().unwrap_or_else(T::default)
    }

    /// Return an array where masked values are replaced by the fill value.
    pub fn filled(&self) -> Array<T> {
        let mut result = self.data.clone();
        let fill = self.fill_value();

        let mask_data = self.mask.data();

        for (i, &is_masked) in mask_data.iter().enumerate() {
            if is_masked {
                result.set_linear(i, fill.clone());
            }
        }
        result
    }

    /// Sum of array elements over a given axis, respecting the mask.
    pub fn sum(&self) -> Result<T>
    where
        T: std::iter::Sum + Clone,
    {
        let mask_data = self.mask.data();
        let data = self.data.data();

        let filtered: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| !m)
            .map(|(d, _)| d.clone())
            .collect();

        Ok(filtered.into_iter().sum())
    }

    /// Mean of array elements, respecting the mask.
    pub fn mean(&self) -> Result<T>
    where
        T: std::iter::Sum + Clone + num_traits::FromPrimitive + std::ops::Div<Output = T> + Default,
    {
        let mask_data = self.mask.data();
        let data = self.data.data();

        let filtered: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| !m)
            .map(|(d, _)| d.clone())
            .collect();

        let count = filtered.len();
        if count == 0 {
            return Ok(T::default());
        }

        let sum: T = filtered.into_iter().sum();
        let count_t = T::from_usize(count)
            .ok_or_else(|| NumPyError::value_error("Failed to convert count to type T", "usize"))?;

        Ok(sum / count_t)
    }

    /// Variance of array elements, respecting the mask.
    pub fn var(&self, ddof: Option<usize>) -> Result<T>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + num_traits::Float
            + num_traits::FromPrimitive
            + num_traits::NumCast
            + std::iter::Sum
            + 'static,
    {
        let mean_val = self.mean()?;
        let mask_data = self.mask.data();
        let data = self.data.data();

        let filtered: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| !m)
            .map(|(d, _)| d.clone())
            .collect();

        let count = filtered.len();
        let ddof = ddof.unwrap_or(0);
        if count <= ddof {
            return Ok(T::nan());
        }

        let sum_sq_diff: T = filtered
            .iter()
            .map(|x| {
                let diff = *x - mean_val.clone();
                diff * diff
            })
            .sum();

        let denominator = T::from_usize(count - ddof).unwrap_or_else(T::one);
        Ok(sum_sq_diff / denominator)
    }

    /// Standard deviation of array elements, respecting the mask.
    pub fn std(&self, ddof: Option<usize>) -> Result<T>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + num_traits::Float
            + num_traits::FromPrimitive
            + std::iter::Sum
            + 'static,
    {
        let variance = self.var(ddof)?;
        Ok(variance.sqrt())
    }

    /// Median of array elements, respecting the mask.
    pub fn median(&self) -> Result<T>
    where
        T: Clone
            + Default
            + std::fmt::Debug
            + PartialOrd
            + num_traits::FromPrimitive
            + num_traits::NumCast
            + 'static,
    {
        let mask_data = self.mask.data();
        let data = self.data.data();

        let mut filtered: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| !m)
            .map(|(d, _)| d.clone())
            .collect();

        let count = filtered.len();
        if count == 0 {
            return Ok(T::default());
        }

        filtered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if count % 2 == 0 {
            let mid1 = &filtered[count / 2 - 1];
            let mid2 = &filtered[count / 2];

            if let (Some(a), Some(b)) = (
                num_traits::cast::<T, f64>(mid1.clone()),
                num_traits::cast::<T, f64>(mid2.clone()),
            ) {
                let avg = (a + b) / 2.0;
                if let Some(result) = num_traits::cast::<f64, T>(avg) {
                    return Ok(result);
                }
            }

            Ok(mid1.clone())
        } else {
            Ok(filtered[count / 2].clone())
        }
    }

    /// Maximum of unmasked elements.
    pub fn max(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        let mut max_val: Option<&T> = None;
        for (data_val, &mask_val) in self.data.data().iter().zip(self.mask.data().iter()) {
            if !mask_val {
                match max_val {
                    None => max_val = Some(data_val),
                    Some(current_max) => {
                        if data_val > current_max {
                            max_val = Some(data_val);
                        }
                    }
                }
            }
        }
        max_val
    }

    /// Minimum of unmasked elements.
    pub fn min(&self) -> Option<&T>
    where
        T: PartialOrd,
    {
        let mut min_val: Option<&T> = None;
        for (data_val, &mask_val) in self.data.data().iter().zip(self.mask.data().iter()) {
            if !mask_val {
                match min_val {
                    None => min_val = Some(data_val),
                    Some(current_min) => {
                        if data_val < current_min {
                            min_val = Some(data_val);
                        }
                    }
                }
            }
        }
        min_val
    }

    /// Product of unmasked elements.
    pub fn prod(&self) -> T
    where
        T: std::ops::Mul<Output = T> + Clone + Default + num_traits::One,
    {
        let mut product = T::one();
        let mut has_unmasked = false;

        for (data_val, &mask_val) in self.data.data().iter().zip(self.mask.data().iter()) {
            if !mask_val {
                product = product * data_val.clone();
                has_unmasked = true;
            }
        }

        if !has_unmasked {
            T::default()
        } else {
            product
        }
    }

    /// Count the non-masked elements of the array.
    pub fn count(&self) -> usize {
        self.mask.data().iter().filter(|&&m| !m).count()
    }

    /// Return True if all unmasked elements evaluate to True.
    pub fn all(&self) -> bool
    where
        T: Into<bool>,
    {
        for (data_val, &mask_val) in self.data.data().iter().zip(self.mask.data().iter()) {
            if !mask_val && !data_val.clone().into() {
                return false;
            }
        }
        true
    }

    /// Return True if any unmasked element evaluates to True.
    pub fn any(&self) -> bool
    where
        T: Into<bool>,
    {
        for (data_val, &mask_val) in self.data.data().iter().zip(self.mask.data().iter()) {
            if !mask_val && data_val.clone().into() {
                return true;
            }
        }
        false
    }

    /// Create a copy of the MaskedArray.
    pub fn copy(&self) -> Self {
        Self {
            data: self.data.clone(),
            mask: self.mask.clone(),
            fill_value: self.fill_value.clone(),
            _hardmask: self._hardmask,
        }
    }

    /// Return a flattened copy of the MaskedArray.
    pub fn flatten(&self) -> Self {
        let flat_data = self.data.flatten();
        let flat_mask = self.mask.flatten();
        Self {
            data: flat_data,
            mask: flat_mask,
            fill_value: self.fill_value.clone(),
            _hardmask: self._hardmask,
        }
    }

    /// Return a reshaped copy of the MaskedArray.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.size() {
            return Err(NumPyError::invalid_value(
                "Cannot reshape: total elements must remain the same",
            ));
        }

        let reshaped_data = Array::from_shape_vec(new_shape.clone(), self.data.data().to_vec())?;
        let reshaped_mask = Array::from_shape_vec(new_shape, self.mask.data().to_vec())?;

        Ok(Self {
            data: reshaped_data,
            mask: reshaped_mask,
            fill_value: self.fill_value.clone(),
            _hardmask: self._hardmask,
        })
    }

    /// Remove single-dimensional entries from the shape.
    pub fn squeeze(&self) -> Self {
        let squeezed_data = self.data.squeeze();
        let squeezed_mask = self.mask.squeeze();

        Self {
            data: squeezed_data,
            mask: squeezed_mask,
            fill_value: self.fill_value.clone(),
            _hardmask: self._hardmask,
        }
    }

    /// Return the transposed MaskedArray.
    pub fn transpose(&self) -> Self {
        Self {
            data: self.data.transpose(),
            mask: self.mask.transpose(),
            fill_value: self.fill_value.clone(),
            _hardmask: self._hardmask,
        }
    }

    /// Take elements from the MaskedArray along an axis.
    pub fn take(&self, indices: &[usize], axis: Option<usize>) -> Result<Self> {
        let taken_data = self.data.take(indices, axis)?;
        let taken_mask = self.mask.take(indices, axis)?;

        Ok(Self {
            data: taken_data,
            mask: taken_mask,
            fill_value: self.fill_value.clone(),
            _hardmask: self._hardmask,
        })
    }

    /// Compress masked array along given axis.
    pub fn compress(&self, condition: &Array<bool>, _axis: Option<usize>) -> Result<MaskedArray<T>>
    where
        T: Clone + Default + 'static,
    {
        let cond_data = condition.data();
        let data = self.data.data();
        let mask_data = self.mask.data();

        let filtered_data: Vec<T> = data
            .iter()
            .zip(mask_data.iter())
            .zip(cond_data.iter())
            .filter(|((_, &m), &c)| !m && c)
            .map(|((d, _), _)| d.clone())
            .collect();

        let len = filtered_data.len();
        let filtered_mask: Vec<bool> = vec![false; len];

        Ok(MaskedArray::new(
            Array::from_vec(filtered_data),
            Array::from_data(filtered_mask, vec![len]),
        )?)
    }

    /// Mask values equal to a given value.
    pub fn masked_values(data: Array<T>, value: T) -> Result<Self>
    where
        T: PartialEq + Clone + Default + Debug + 'static,
    {
        let mask_data: Vec<bool> = data.data().iter().map(|x| *x == value).collect();
        let mask = Array::from_data(mask_data, data.shape().to_vec());
        let mut ma = Self::new(data, mask)?;
        ma.set_fill_value(value);
        Ok(ma)
    }

    /// Mask values outside a given range [min, max].
    pub fn masked_outside(data: Array<T>, min: T, max: T) -> Result<Self>
    where
        T: PartialOrd + Clone + Default + Debug + 'static,
    {
        let mask_data: Vec<bool> = data.data().iter().map(|x| *x < min || *x > max).collect();
        let mask = Array::from_data(mask_data, data.shape().to_vec());
        Self::new(data, mask)
    }

    /// Mask values inside a given range [min, max].
    pub fn masked_inside(data: Array<T>, min: T, max: T) -> Result<Self>
    where
        T: PartialOrd + Clone + Default + Debug + 'static,
    {
        let mask_data: Vec<bool> = data.data().iter().map(|x| *x >= min && *x <= max).collect();
        let mask = Array::from_data(mask_data, data.shape().to_vec());
        Self::new(data, mask)
    }

    /// Hardens the mask to prevent unmasking.
    pub fn harden_mask(&mut self) {
        self._hardmask = true;
    }

    /// Softens the mask to allow unmasking.
    pub fn soften_mask(&mut self) {
        self._hardmask = false;
    }

    /// Shrinks the mask to include only masked data.
    pub fn shrink_mask(&mut self) {
        // If all mask values are false, set to empty mask
        if self.mask.data().iter().all(|&m| !m) {
            let shape = self.mask.shape().to_vec();
            self.mask = Array::from_data(vec![], shape);
        }
    }
}

// ===== Module-level numpy.ma functions =====

/// The mask value - a special sentinel indicating no mask
pub const NOMASK: bool = false;

/// Create a masked array from data with optional mask
pub fn masked_array<T>(data: Array<T>, mask: Option<Array<bool>>) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    match mask {
        Some(m) => MaskedArray::new(data, m),
        None => Ok(MaskedArray::from_data(data)),
    }
}

/// Create an empty masked array with the given shape
pub fn empty<T>(shape: &[usize]) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    let data = Array::from_shape_vec(shape.to_vec(), vec![T::default(); shape.iter().product()])?;
    Ok(MaskedArray::from_data(data))
}

/// Create a masked array filled with zeros
pub fn zeros<T>(shape: &[usize]) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    let data = Array::zeros(shape.to_vec())?;
    Ok(MaskedArray::from_data(data))
}

/// Create a masked array filled with ones
pub fn ones<T>(shape: &[usize]) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    let data = Array::ones(shape.to_vec())?;
    Ok(MaskedArray::from_data(data))
}

/// Create a masked array filled with a fill value
pub fn full<T>(shape: &[usize], fill_value: T) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    let data = Array::from_shape_vec(shape.to_vec(), vec![fill_value; shape.iter().product()])?;
    Ok(MaskedArray::from_data(data))
}

/// Create a full-like masked array
pub fn full_like<T>(a: &Array<T>, fill_value: T) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    Ok(MaskedArray::from_data(a.clone()))
}

/// Convert data to a masked array
pub fn array<T>(data: Array<T>) -> MaskedArray<T>
where
    T: Clone + Debug + Default + 'static,
{
    MaskedArray::from_data(data)
}

/// Get the mask of a masked array
pub fn getmask<T>(ma: &MaskedArray<T>) -> &Array<bool> {
    ma.mask()
}

/// Get the mask of a masked array, or an array of False if no mask exists
pub fn getmaskarray<T>(ma: &MaskedArray<T>) -> Array<bool> {
    ma.mask().clone()
}

/// Create a mask for values equal to a given value
pub fn masked_values<T>(data: Array<T>, value: T) -> Result<MaskedArray<T>>
where
    T: PartialEq + Clone + Debug + Default + 'static,
{
    MaskedArray::masked_values(data, value)
}

/// Create a mask for values outside a range
pub fn masked_outside<T>(data: Array<T>, min: T, max: T) -> Result<MaskedArray<T>>
where
    T: PartialOrd + Clone + Debug + Default + 'static,
{
    MaskedArray::masked_outside(data, min, max)
}

/// Create a mask for values inside a range
pub fn masked_inside<T>(data: Array<T>, min: T, max: T) -> Result<MaskedArray<T>>
where
    T: PartialOrd + Clone + Debug + Default + 'static,
{
    MaskedArray::masked_inside(data, min, max)
}

/// Mask elements where a condition is true
pub fn masked_where<T>(condition: Array<bool>, data: Array<T>) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    MaskedArray::new(data, condition)
}

/// Mask elements greater than a value
pub fn masked_greater<T>(data: Array<T>, value: T) -> Result<MaskedArray<T>>
where
    T: PartialOrd + Clone + Debug + Default + 'static,
{
    let mask: Vec<bool> = data.data().iter().map(|x| *x > value).collect();
    let mask_arr = Array::from_data(mask, data.shape().to_vec());
    MaskedArray::new(data, mask_arr)
}

/// Mask elements greater than or equal to a value
pub fn masked_greater_equal<T>(data: Array<T>, value: T) -> Result<MaskedArray<T>>
where
    T: PartialOrd + Clone + Debug + Default + 'static,
{
    let mask: Vec<bool> = data.data().iter().map(|x| *x >= value).collect();
    let mask_arr = Array::from_data(mask, data.shape().to_vec());
    MaskedArray::new(data, mask_arr)
}

/// Mask elements less than a value
pub fn masked_less<T>(data: Array<T>, value: T) -> Result<MaskedArray<T>>
where
    T: PartialOrd + Clone + Debug + Default + 'static,
{
    let mask: Vec<bool> = data.data().iter().map(|x| *x < value).collect();
    let mask_arr = Array::from_data(mask, data.shape().to_vec());
    MaskedArray::new(data, mask_arr)
}

/// Mask elements less than or equal to a value
pub fn masked_less_equal<T>(data: Array<T>, value: T) -> Result<MaskedArray<T>>
where
    T: PartialOrd + Clone + Debug + Default + 'static,
{
    let mask: Vec<bool> = data.data().iter().map(|x| *x <= value).collect();
    let mask_arr = Array::from_data(mask, data.shape().to_vec());
    MaskedArray::new(data, mask_arr)
}

/// Mask elements equal to a value
pub fn masked_equal<T>(data: Array<T>, value: T) -> Result<MaskedArray<T>>
where
    T: PartialEq + Clone + Debug + Default + 'static,
{
    let mask: Vec<bool> = data.data().iter().map(|x| *x == value).collect();
    let mask_arr = Array::from_data(mask, data.shape().to_vec());
    MaskedArray::new(data, mask_arr)
}

/// Mask elements not equal to a value
pub fn masked_not_equal<T>(data: Array<T>, value: T) -> Result<MaskedArray<T>>
where
    T: PartialEq + Clone + Debug + Default + 'static,
{
    let mask: Vec<bool> = data.data().iter().map(|x| *x != value).collect();
    let mask_arr = Array::from_data(mask, data.shape().to_vec());
    MaskedArray::new(data, mask_arr)
}

/// Mask element if value is contained in the data
pub fn masked_object<T>(data: Array<T>, value: T) -> Result<MaskedArray<T>>
where
    T: PartialEq + Clone + Debug + Default + 'static,
{
    masked_equal(data, value)
}

/// Test if all masked array elements evaluate to True
pub fn all<T>(ma: &MaskedArray<T>) -> bool
where
    T: Into<bool>,
{
    ma.all()
}

/// Test if any masked array element evaluates to True
pub fn any<T>(ma: &MaskedArray<T>) -> bool
where
    T: Into<bool>,
{
    ma.any()
}

/// Sum of masked array elements
pub fn sum<T>(ma: &MaskedArray<T>) -> Result<T>
where
    T: std::iter::Sum + Clone,
{
    ma.sum()
}

/// Mean of masked array elements
pub fn mean<T>(ma: &MaskedArray<T>) -> Result<T>
where
    T: std::iter::Sum + Clone + num_traits::FromPrimitive + std::ops::Div<Output = T> + Default,
{
    ma.mean()
}

/// Variance of masked array elements
pub fn var<T>(ma: &MaskedArray<T>, ddof: Option<usize>) -> Result<T>
where
    T: Clone + Default + std::fmt::Debug + num_traits::Float + num_traits::FromPrimitive + std::iter::Sum + 'static,
{
    ma.var(ddof)
}

/// Standard deviation of masked array elements
pub fn std<T>(ma: &MaskedArray<T>, ddof: Option<usize>) -> Result<T>
where
    T: Clone + Default + std::fmt::Debug + num_traits::Float + num_traits::FromPrimitive + std::iter::Sum + 'static,
{
    ma.std(ddof)
}

/// Median of masked array elements
pub fn median<T>(ma: &MaskedArray<T>) -> Result<T>
where
    T: Clone + Default + std::fmt::Debug + PartialOrd + num_traits::FromPrimitive + num_traits::NumCast + 'static,
{
    ma.median()
}

/// Maximum of masked array elements
pub fn max<T>(ma: &MaskedArray<T>) -> Option<T>
where
    T: Clone + PartialOrd,
{
    ma.max().cloned()
}

/// Minimum of masked array elements
pub fn min<T>(ma: &MaskedArray<T>) -> Option<T>
where
    T: Clone + PartialOrd,
{
    ma.min().cloned()
}

/// Product of masked array elements
pub fn prod<T>(ma: &MaskedArray<T>) -> T
where
    T: std::ops::Mul<Output = T> + Clone + Default + num_traits::One,
{
    ma.prod()
}

/// Cumulative sum of masked array elements
pub fn cumsum<T>(ma: &MaskedArray<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + std::ops::Add<Output = T>,
{
    // Simplified: ignore mask for cumsum
    ma.data().cumsum(axis)
}

/// Cumulative product of masked array elements
pub fn cumprod<T>(ma: &MaskedArray<T>, axis: Option<isize>) -> Result<Array<T>>
where
    T: Clone + Default + std::ops::Mul<Output = T>,
{
    // Simplified: ignore mask for cumprod
    ma.data().cumprod(axis)
}

/// Peak-to-peak (max - min) of masked array elements
pub fn ptp<T>(ma: &MaskedArray<T>) -> Option<f64>
where
    T: PartialOrd + Clone + Into<f64>,
{
    if ma.count() == 0 {
        return None;
    }
    let max_val = ma.max()?.clone().into();
    let min_val = ma.min()?.clone().into();
    Some(max_val - min_val)
}

/// Count the number of unmasked elements
pub fn count<T>(ma: &MaskedArray<T>) -> usize {
    ma.count()
}

/// Return the underlying data of a masked array
pub fn getdata<T>(ma: &MaskedArray<T>) -> &Array<T> {
    ma.data()
}

/// Copy a masked array
pub fn copy<T>(ma: &MaskedArray<T>) -> MaskedArray<T> {
    ma.copy()
}

/// Reshape a masked array
pub fn reshape<T>(ma: &MaskedArray<T>, new_shape: &[usize]) -> Result<MaskedArray<T>> {
    ma.reshape(new_shape.to_vec())
}

/// Transpose a masked array
pub fn transpose<T>(ma: &MaskedArray<T>) -> MaskedArray<T> {
    ma.transpose()
}

/// Swap axes of a masked array
pub fn swapaxes<T>(ma: &MaskedArray<T>, axis1: isize, axis2: isize) -> MaskedArray<T> {
    // Simplified implementation
    ma.transpose()
}

/// Take elements from a masked array
pub fn take<T>(ma: &MaskedArray<T>, indices: &[usize], axis: Option<usize>) -> Result<MaskedArray<T>> {
    ma.take(indices, axis)
}

/// Fill the masked array with a given value
pub fn filled<T>(ma: &MaskedArray<T>) -> Array<T> {
    ma.filled()
}

/// Set the fill value of a masked array
pub fn set_fill_value<T>(ma: &mut MaskedArray<T>, value: T) {
    ma.set_fill_value(value)
}

/// Get the fill value of a masked array
pub fn get_fill_value<T>(ma: &MaskedArray<T>) -> T {
    ma.fill_value()
}

/// Hardens the mask to prevent unmasking
pub fn harden_mask<T>(ma: &mut MaskedArray<T>) {
    ma.harden_mask()
}

/// Softens the mask to allow unmasking
pub fn soften_mask<T>(ma: &mut MaskedArray<T>) {
    ma.soften_mask()
}

/// Shrinks the mask to include only masked data
pub fn shrink_mask<T>(ma: &mut MaskedArray<T>) {
    ma.shrink_mask()
}

/// Check if two masked arrays are close
pub fn allclose<T>(a: &MaskedArray<T>, b: &MaskedArray<T>, rtol: Option<T>, atol: Option<T>) -> bool
where
    T: num_traits::Float + Clone + PartialEq,
{
    let rtol_val = rtol.unwrap_or_else(|| num_traits::Float::epsilon() * T::from(10.0).unwrap_or(T::one()));
    let atol_val = atol.unwrap_or_else(num_traits::Float::epsilon);

    let a_data = a.data().data();
    let b_data = b.data().data();
    let a_mask = a.mask().data();
    let b_mask = b.mask().data();

    if a_data.len() != b_data.len() {
        return false;
    }

    for ((a_val, b_val), (&a_m, &b_m)) in a_data.iter().zip(b_data.iter()).zip(a_mask.iter().zip(b_mask.iter())) {
        if a_m || b_m {
            continue;
        }
        let diff = (*a_val - b_val).abs();
        let tolerance = atol_val + rtol_val * b_val.abs();
        if diff > tolerance {
            return false;
        }
    }
    true
}

/// Check if masked array elements are close to each other
pub fn isclose<T>(a: &MaskedArray<T>, b: &MaskedArray<T>, rtol: Option<T>, atol: Option<T>) -> Result<MaskedArray<bool>>
where
    T: num_traits::Float + Clone + PartialEq + 'static,
{
    let rtol_val = rtol.unwrap_or_else(|| num_traits::Float::epsilon() * T::from(10.0).unwrap_or(T::one()));
    let atol_val = atol.unwrap_or_else(num_traits::Float::epsilon);

    let a_data = a.data().data();
    let b_data = b.data().data();
    let a_mask = a.mask().data();
    let b_mask = b.mask().data();

    let mut result_data = Vec::with_capacity(a_data.len());
    let mut result_mask = Vec::with_capacity(a_data.len());

    for ((a_val, b_val), (&a_m, &b_m)) in a_data.iter().zip(b_data.iter()).zip(a_mask.iter().zip(b_mask.iter())) {
        let is_masked = a_m || b_m;
        if is_masked {
            result_data.push(false);
            result_mask.push(false);
        } else {
            let diff = (*a_val - b_val).abs();
            let tolerance = atol_val + rtol_val * b_val.abs();
            result_data.push(diff <= tolerance);
            result_mask.push(false);
        }
    }

    let result = Array::from_data(result_data, a.shape().to_vec());
    let result_mask_arr = Array::from_data(result_mask, a.shape().to_vec());
    MaskedArray::new(result, result_mask_arr)
}

/// Check if all unmasked elements are equal
pub fn allequal<T>(a: &MaskedArray<T>, b: &MaskedArray<T>) -> bool
where
    T: PartialEq + Clone,
{
    if a.shape() != b.shape() {
        return false;
    }

    let a_data = a.data().data();
    let b_data = b.data().data();
    let a_mask = a.mask().data();
    let b_mask = b.mask().data();

    for ((a_val, b_val), (&a_m, &b_m)) in a_data.iter().zip(b_data.iter()).zip(a_mask.iter().zip(b_mask.iter())) {
        if a_m || b_m {
            continue;
        }
        if a_val != b_val {
            return false;
        }
    }
    true
}

/// Check if a value is a MaskedArray
pub fn is_masked_array<T>(_: &T) -> bool {
    false
}

/// Return the average of masked array elements
pub fn average<T>(ma: &MaskedArray<T>) -> Result<T>
where
    T: std::iter::Sum + Clone + num_traits::FromPrimitive + std::ops::Div<Output = T> + Default,
{
    ma.mean()
}

/// Append values to the end of a masked array
pub fn append<T>(ma: &MaskedArray<T>, values: &Array<T>, axis: Option<isize>) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    // Simplified: ignore mask for append
    let new_data = ma.data().append(values, axis)?;
    Ok(MaskedArray::from_data(new_data))
}

/// Concatenate masked arrays along an axis
pub fn concatenate<T>(arrays: &[&MaskedArray<T>], axis: isize) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    if arrays.is_empty() {
        return Err(NumPyError::invalid_value("Cannot concatenate empty array"));
    }

    let first_shape = arrays[0].shape();
    let normalized_axis = if axis < 0 {
        first_shape.len() as isize + axis
    } else {
        axis
    } as usize;

    // Extract data and masks
    let mut all_data = Vec::new();
    let mut all_mask = Vec::new();
    let mut output_shape = first_shape.to_vec();

    for (i, ma) in arrays.iter().enumerate() {
        let data = ma.data().data().clone();
        let mask = ma.mask().data().clone();

        // Adjust shape for concatenation
        if i > 0 {
            output_shape[normalized_axis] += ma.shape()[normalized_axis];
        }

        all_data.push(data);
        all_mask.push(mask);
    }

    // Flatten and interleave
    let flat_data: Vec<T> = all_data.into_iter().flatten().collect();
    let flat_mask: Vec<bool> = all_mask.into_iter().flatten().collect();

    let result = Array::from_data(flat_data, output_shape.clone());
    let result_mask = Array::from_data(flat_mask, output_shape);

    MaskedArray::new(result, result_mask)
}

/// Resize a masked array
pub fn resize<T>(ma: &MaskedArray<T>, new_shape: &[usize]) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + 'static,
{
    let total_new: usize = new_shape.iter().product();
    let total_old = ma.size();

    let mut new_data = Vec::with_capacity(total_new);
    let mut new_mask = Vec::with_capacity(total_new);

    let old_data = ma.data().data();
    let old_mask = ma.mask().data();

    for i in 0..total_new {
        let idx = i % total_old;
        new_data.push(old_data[idx].clone());
        new_mask.push(old_mask[idx]);
    }

    let result = Array::from_data(new_data, new_shape.to_vec())?;
    let result_mask = Array::from_data(new_mask, new_shape.to_vec());
    MaskedArray::new(result, result_mask)
}

/// Sort a masked array along a given axis
pub fn sort<T>(ma: &MaskedArray<T>, axis: Option<isize>) -> Result<MaskedArray<T>>
where
    T: Clone + Debug + Default + PartialOrd + 'static,
{
    // Simplified: sort ignoring mask
    let sorted_data = ma.data().sort(axis)?;
    Ok(MaskedArray::from_data(sorted_data))
}

/// Argsort of masked array
pub fn argsort<T>(ma: &MaskedArray<T>, axis: Option<isize>) -> Result<Array<usize>>
where
    T: Clone + Debug + Default + PartialOrd + 'static,
{
    // Simplified: argsort ignoring mask
    let sorted_indices = ma.data().argsort(axis)?;
    Ok(sorted_indices)
}

/// Argmax of masked array
pub fn argmax<T>(ma: &MaskedArray<T>, axis: Option<isize>) -> Result<Array<usize>>
where
    T: Clone + Debug + Default + PartialOrd + 'static,
{
    ma.data().argmax(axis)
}

/// Argmin of masked array
pub fn argmin<T>(ma: &MaskedArray<T>, axis: Option<isize>) -> Result<Array<usize>>
where
    T: Clone + Debug + Default + PartialOrd + 'static,
{
    ma.data().argmin(axis)
}

/// Where condition for masked arrays
pub fn where_<T>(condition: Array<bool>, x: Array<T>, y: Array<T>) -> Result<Array<T>>
where
    T: Clone + Debug + Default + 'static,
{
    let cond_data = condition.data();
    let x_data = x.data();
    let y_data = y.data();

    let mut result = Vec::with_capacity(cond_data.len());
    for (i, &c) in cond_data.iter().enumerate() {
        if c {
            result.push(x_data[i].clone());
        } else {
            result.push(y_data[i].clone());
        }
    }

    Array::from_vec(result)
}

/// Put values into masked array
pub fn put<T>(ma: &mut MaskedArray<T>, indices: &[usize], values: &[T]) -> Result<()>
where
    T: Clone + Debug + 'static,
{
    for (idx, val) in indices.iter().zip(values.iter()) {
        ma.data().set_linear(*idx, val.clone());
    }
    Ok(())
}

/// Place values into masked array
pub fn place<T>(ma: &mut MaskedArray<T>, condition: &Array<bool>, values: &[T]) -> Result<()>
where
    T: Clone + Debug + 'static,
{
    let cond_data = condition.data();
    let mut data = ma.data().clone();
    let mut val_idx = 0;

    for (i, &c) in cond_data.iter().enumerate() {
        if c {
            if val_idx < values.len() {
                data.set_linear(i, values[val_idx].clone());
                val_idx += 1;
            }
        }
    }

    Ok(())
}

/// Fill masked array with a scalar value
pub fn fill<T>(ma: &mut MaskedArray<T>, value: T)
where
    T: Clone,
{
    let shape = ma.shape().to_vec();
    let new_data = Array::from_shape_vec(shape, vec![value; ma.size()]).unwrap();
    ma.data = new_data;
}

/// Check if array contains NaN values
pub fn isnan<T>(a: &Array<T>) -> Array<bool>
where
    T: num_traits::Float + Clone + 'static,
{
    let data = a.data();
    let nan_mask: Vec<bool> = data.iter().map(|x| x.is_nan()).collect();
    Array::from_data(nan_mask, a.shape().to_vec())
}

/// Check if object is a MaskedArray
pub fn is_masked_array_type<T>(_: &T) -> bool {
    false
}

pub mod exports {
    pub use super::{
        all, allequal, any, append, argmax, argmin, argsort, array, average, concatenate,
        copy, count, cumprod, cumsum, empty, filled, full, full_like, getdata,
        getmask, getmaskarray, harden_mask, isclose, is_masked_array, is_masked_array_type,
        isnan, masked_array, masked_equal, masked_greater, masked_greater_equal,
        masked_inside, masked_less, masked_less_equal, masked_not_equal, masked_object,
        masked_outside, masked_values, masked_where, MaskedArray, max, mean, median, min,
        ones, place, prod, ptp, put, reshape, resize, set_fill_value, shrink_mask,
        sort, std, sum, swapaxes, take, transpose, var, where_, zeros,
    };
}

#[cfg(test)]
mod additional_tests;
