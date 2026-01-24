//! Array implementation for Rust NumPy port
//!
//! This module provides a basic Array struct that demonstrates
//! how to implement NumPy-compatible arrays in Rust.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Array {
    /// Create a new 1D array from a vector
    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self {
            data,
            shape: vec![len],
            strides: vec![1],
        }
    }

    /// Create a new array with the given shape and data
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        let total_elements: usize = shape.iter().product();
        if total_elements != data.len() {
            return Err(format!(
                "Data length {} doesn't match shape {:?}",
                data.len(),
                shape
            ));
        }

        // Calculate strides
        let mut strides = vec![1usize; shape.len()];
        for i in (1..shape.len()).rev() {
            strides[i - 1] = strides[i] * shape[i];
        }

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    /// Reshape the array to new dimensions
    pub fn reshape(mut self, new_shape: &[usize]) -> Result<Self, String> {
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.data.len() {
            return Err(format!(
                "Cannot reshape array of size {} to shape {:?}",
                self.data.len(),
                new_shape
            ));
        }

        // Calculate new strides
        let mut strides = vec![1usize; new_shape.len()];
        for i in (1..new_shape.len()).rev() {
            strides[i - 1] = strides[i] * new_shape[i];
        }

        self.shape = new_shape.to_vec();
        self.strides = strides;
        Ok(self)
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get iterator over the data
    pub fn iter(&self) -> std::slice::Iter<f64> {
        self.data.iter()
    }

    /// Get mutable iterator over the data
    pub fn iter_mut(&mut self) -> std::slice::IterMut<f64> {
        self.data.iter_mut()
    }

    /// Get a reference to the underlying data
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Calculate the total size of the array
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if two arrays are approximately equal
    pub fn all_close(&self, other: &Array, tolerance: f64) -> bool {
        if self.shape != other.shape {
            return false;
        }

        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(&a, &b)| (a - b).abs() < tolerance)
    }

    /// Create an array filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![0.0; total_elements];
        Self::new(data, shape.to_vec()).unwrap()
    }

    /// Create an array filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![1.0; total_elements];
        Self::new(data, shape.to_vec()).unwrap()
    }

    /// Create an identity matrix
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::new(data, vec![n, n]).unwrap()
    }

    /// Create a range array (equivalent to numpy.arange)
    pub fn arange(start: isize, stop: isize, step: isize) -> Self {
        let mut data = Vec::new();
        let mut current = start;

        if step > 0 {
            while current < stop {
                data.push(current as f64);
                current += step;
            }
        } else if step < 0 {
            while current > stop {
                data.push(current as f64);
                current += step;
            }
        }

        Self::from_vec(data)
    }

    /// Create a linearly spaced array (equivalent to numpy.linspace)
    pub fn linspace(start: f64, stop: f64, num: usize) -> Self {
        if num == 0 {
            return Self::from_vec(vec![]);
        }
        if num == 1 {
            return Self::from_vec(vec![start]);
        }

        let step = (stop - start) / (num - 1) as f64;
        let mut data = Vec::with_capacity(num);

        for i in 0..num {
            data.push(start + i as f64 * step);
        }

        Self::from_vec(data)
    }

    /// Transpose the array (for 2D arrays)
    pub fn transpose(&self) -> Result<Self, String> {
        if self.ndim() != 2 {
            return Err("Transpose only implemented for 2D arrays".to_string());
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut transposed_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Self::new(transposed_data, vec![cols, rows])
    }

    /// Flatten the array to 1D
    pub fn flatten(&self) -> Self {
        Self::from_vec(self.data.clone())
    }

    /// Compute the sum of all elements
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// Compute the mean of all elements
    pub fn mean(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        self.sum() / self.len() as f64
    }

    /// Find the minimum value
    pub fn min(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        self.data.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    }

    /// Find the maximum value
    pub fn max(&self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        self.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Find the index of the minimum value
    pub fn argmin(&self) -> usize {
        if self.is_empty() {
            return 0;
        }

        let mut min_index = 0;
        let mut min_value = self.data[0];

        for (i, &value) in self.data.iter().enumerate() {
            if value < min_value {
                min_value = value;
                min_index = i;
            }
        }

        min_index
    }

    /// Find the index of the maximum value
    pub fn argmax(&self) -> usize {
        if self.is_empty() {
            return 0;
        }

        let mut max_index = 0;
        let mut max_value = self.data[0];

        for (i, &value) in self.data.iter().enumerate() {
            if value > max_value {
                max_value = value;
                max_index = i;
            }
        }

        max_index
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array(shape={:?}, data=[", self.shape)?;

        for (i, &value) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.6}", value)?;
        }

        write!(f, "])")
    }
}

/// Create an array from a vector (convenience function)
pub fn array(data: Vec<f64>) -> Result<Array, String> {
    Ok(Array::from_vec(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let arr = Array::from_vec(data);

        assert_eq!(arr.shape(), &[4]);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_array_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::from_vec(data).reshape(&[2, 3]).unwrap();

        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.len(), 6);
    }

    #[test]
    fn test_zeros() {
        let arr = Array::zeros(&[3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.sum(), 0.0);
    }

    #[test]
    fn test_ones() {
        let arr = Array::ones(&[2, 3]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.sum(), 6.0);
    }

    #[test]
    fn test_eye() {
        let arr = Array::eye(3);
        assert_eq!(arr.shape(), &[3, 3]);
        assert_eq!(arr.sum(), 3.0);
    }

    #[test]
    fn test_arange() {
        let arr = Array::arange(0, 10, 2);
        assert_eq!(arr.shape(), &[5]);
        assert_eq!(arr.data(), &[0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_linspace() {
        let arr = Array::linspace(0.0, 1.0, 5);
        assert_eq!(arr.shape(), &[5]);
        assert!(arr.all_close(&Array::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]), 1e-10));
    }

    #[test]
    fn test_transpose() {
        let arr = Array::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let transposed = arr.transpose().unwrap();

        assert_eq!(transposed.shape(), &[2, 2]);
        assert_eq!(transposed.data(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_statistical_functions() {
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(arr.sum(), 15.0);
        assert_eq!(arr.mean(), 3.0);
        assert_eq!(arr.min(), 1.0);
        assert_eq!(arr.max(), 5.0);
        assert_eq!(arr.argmin(), 0);
        assert_eq!(arr.argmax(), 4);
    }
}
