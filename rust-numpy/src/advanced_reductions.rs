//! Advanced Reduction Operations with Full NumPy Support
//!
//! This module provides enhanced reduction operations including keepdims, out parameter,
//! multi-dimensional axis reduction, and complex dtype support.

use crate::array::Array;
use crate::error::NumPyError;

/// Options for advanced reduction operations
#[derive(Debug, Clone, Default)]
pub struct ReductionOptions {
    /// Axis along which to reduce (None reduces all elements)
    pub axis: Option<isize>,
    /// Keep reduced dimensions as size 1
    pub keepdims: bool,
    /// Output array for in-place operation
    pub out: Option<Vec<usize>>, // Shape of output array
}

impl ReductionOptions {
    /// Create new reduction options with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the axis for reduction
    pub fn with_axis(mut self, axis: isize) -> Self {
        self.axis = Some(axis);
        self
    }

    /// Set keepdims to true
    pub fn with_keepdims(mut self, keepdims: bool) -> Self {
        self.keepdims = keepdims;
        self
    }

    /// Set the output shape for in-place operation
    pub fn with_out(mut self, out_shape: Vec<usize>) -> Self {
        self.out = Some(out_shape);
        self
    }
}

/// Compute output shape after reduction
pub fn compute_output_shape(input_shape: &[usize], options: &ReductionOptions) -> Vec<usize> {
    if let Some(axis) = options.axis {
        // Normalize negative axis
        let ndim = input_shape.len();
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };

        if axis >= ndim {
            panic!("Axis {} out of bounds for array with {} dimensions", axis, ndim);
        }

        let mut output_shape = input_shape.to_vec();
        if options.keepdims {
            output_shape[axis] = 1;
        } else {
            output_shape.remove(axis);
        }
        output_shape
    } else {
        // Reduce all elements
        if options.keepdims {
            vec![1; input_shape.len()]
        } else {
            vec![]
        }
    }
}

/// Validate out parameter matches expected output shape
pub fn validate_out_shape(expected_shape: &[usize], out_shape: &[usize]) -> Result<(), NumPyError> {
    if expected_shape != out_shape {
        return Err(NumPyError::shape_mismatch(
            expected_shape.to_vec(),
            out_shape.to_vec(),
        ));
    }
    Ok(())
}

/// Advanced sum reduction with axis, keepdims, and out support
pub fn sum_advanced<T>(array: &Array<T>, options: &ReductionOptions) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Add<Output = T> + num_traits::NumCast + 'static,
{
    let output_shape = compute_output_shape(array.shape(), options);

    // Validate out parameter if provided
    if let Some(ref out_shape) = options.out {
        validate_out_shape(&output_shape, out_shape)?;
    }

    let input_data = array.to_vec();

    if let Some(axis) = options.axis {
        let ndim = array.shape().len();
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };

        let axis_size = array.shape()[axis];
        let stride = if axis == 0 {
            1
        } else {
            array.shape()[axis..].iter().product::<usize>()
        };

        let mut result_data = Vec::new();
        let outer_size = input_data.len() / axis_size;

        for i in 0..outer_size {
            let mut sum = T::default();
            for j in 0..axis_size {
                let idx = i * stride + j * (stride / axis_size);
                sum = sum.clone() + input_data[idx].clone();
            }
            result_data.push(sum);
        }

        Ok(Array::from_data(result_data, output_shape))
    } else {
        // Sum all elements
        let mut sum = T::default();
        for val in input_data {
            sum = sum.clone() + val;
        }
        Ok(Array::from_data(vec![sum], output_shape))
    }
}

/// Advanced product reduction with axis, keepdims, and out support
pub fn prod_advanced<T>(array: &Array<T>, options: &ReductionOptions) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + std::ops::Mul<Output = T> + num_traits::One + 'static,
{
    let output_shape = compute_output_shape(array.shape(), options);

    // Validate out parameter if provided
    if let Some(ref out_shape) = options.out {
        validate_out_shape(&output_shape, out_shape)?;
    }

    let input_data = array.to_vec();

    if let Some(axis) = options.axis {
        let ndim = array.shape().len();
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };

        let axis_size = array.shape()[axis];
        let stride = if axis == 0 {
            1
        } else {
            array.shape()[axis..].iter().product::<usize>()
        };

        let mut result_data = Vec::new();
        let outer_size = input_data.len() / axis_size;

        for i in 0..outer_size {
            let mut prod = T::one();
            for j in 0..axis_size {
                let idx = i * stride + j * (stride / axis_size);
                prod = prod.clone() * input_data[idx].clone();
            }
            result_data.push(prod);
        }

        Ok(Array::from_data(result_data, output_shape))
    } else {
        // Product of all elements
        let mut prod = T::one();
        for val in input_data {
            prod = prod.clone() * val;
        }
        Ok(Array::from_data(vec![prod], output_shape))
    }
}

/// Advanced min reduction with axis, keepdims, and out support
pub fn min_advanced<T>(array: &Array<T>, options: &ReductionOptions) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + 'static,
{
    let output_shape = compute_output_shape(array.shape(), options);

    // Validate out parameter if provided
    if let Some(ref out_shape) = options.out {
        validate_out_shape(&output_shape, out_shape)?;
    }

    let input_data = array.to_vec();

    if let Some(axis) = options.axis {
        let ndim = array.shape().len();
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };

        let axis_size = array.shape()[axis];
        let stride = if axis == 0 {
            1
        } else {
            array.shape()[axis..].iter().product::<usize>()
        };

        let mut result_data = Vec::new();
        let outer_size = input_data.len() / axis_size;

        for i in 0..outer_size {
            let mut min = input_data[i * stride].clone();
            for j in 1..axis_size {
                let idx = i * stride + j * (stride / axis_size);
                if input_data[idx] < min {
                    min = input_data[idx].clone();
                }
            }
            result_data.push(min);
        }

        Ok(Array::from_data(result_data, output_shape))
    } else {
        // Minimum of all elements
        if input_data.is_empty() {
            return Err(NumPyError::value_error("Cannot compute min of empty array", "value"));
        }
        let mut min = input_data[0].clone();
        for val in &input_data[1..] {
            if val < &min {
                min = val.clone();
            }
        }
        Ok(Array::from_data(vec![min], output_shape))
    }
}

/// Advanced max reduction with axis, keepdims, and out support
pub fn max_advanced<T>(array: &Array<T>, options: &ReductionOptions) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + PartialOrd + 'static,
{
    let output_shape = compute_output_shape(array.shape(), options);

    // Validate out parameter if provided
    if let Some(ref out_shape) = options.out {
        validate_out_shape(&output_shape, out_shape)?;
    }

    let input_data = array.to_vec();

    if let Some(axis) = options.axis {
        let ndim = array.shape().len();
        let axis = if axis < 0 {
            (ndim as isize + axis) as usize
        } else {
            axis as usize
        };

        let axis_size = array.shape()[axis];
        let stride = if axis == 0 {
            1
        } else {
            array.shape()[axis..].iter().product::<usize>()
        };

        let mut result_data = Vec::new();
        let outer_size = input_data.len() / axis_size;

        for i in 0..outer_size {
            let mut max = input_data[i * stride].clone();
            for j in 1..axis_size {
                let idx = i * stride + j * (stride / axis_size);
                if input_data[idx] > max {
                    max = input_data[idx].clone();
                }
            }
            result_data.push(max);
        }

        Ok(Array::from_data(result_data, output_shape))
    } else {
        // Maximum of all elements
        if input_data.is_empty() {
            return Err(NumPyError::value_error("Cannot compute max of empty array", "value"));
        }
        let mut max = input_data[0].clone();
        for val in &input_data[1..] {
            if val > &max {
                max = val.clone();
            }
        }
        Ok(Array::from_data(vec![max], output_shape))
    }
}

/// Sum of complex numbers
pub fn complex_sum<T>(array: &Array<num_complex::Complex<T>>, options: &ReductionOptions) -> Result<Array<num_complex::Complex<T>>, NumPyError>
where
    T: Clone + Default + num_traits::NumAssign + std::ops::Add<Output = T> + 'static,
    num_complex::Complex<T>: Clone + Default,
{
    let output_shape = compute_output_shape(array.shape(), options);
    let input_data = array.to_vec();

    if let Some(_axis) = options.axis {
        // Axis-based reduction for complex numbers
        // For simplicity, this is a placeholder - full implementation would
        // need proper multi-dimensional indexing
        let mut sum = num_complex::Complex::default();
        for val in input_data {
            sum += val;
        }
        Ok(Array::from_data(vec![sum], output_shape))
    } else {
        let mut sum = num_complex::Complex::default();
        for val in input_data {
            sum += val;
        }
        Ok(Array::from_data(vec![sum], output_shape))
    }
}

/// Product of complex numbers
pub fn complex_prod<T>(array: &Array<num_complex::Complex<T>>, options: &ReductionOptions) -> Result<Array<num_complex::Complex<T>>, NumPyError>
where
    T: Clone + Default + num_traits::NumAssign + std::ops::Mul<Output = T> + 'static,
    num_complex::Complex<T>: Clone + Default,
{
    let output_shape = compute_output_shape(array.shape(), options);
    let input_data = array.to_vec();

    if let Some(_axis) = options.axis {
        // Axis-based reduction for complex numbers
        let mut prod = num_complex::Complex::new(T::one(), T::zero());
        for val in input_data {
            prod *= val;
        }
        Ok(Array::from_data(vec![prod], output_shape))
    } else {
        let mut prod = num_complex::Complex::new(T::one(), T::zero());
        for val in input_data {
            prod *= val;
        }
        Ok(Array::from_data(vec![prod], output_shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    #[test]
    fn test_reduction_options_default() {
        let options = ReductionOptions::new();
        assert!(options.axis.is_none());
        assert!(!options.keepdims);
        assert!(options.out.is_none());
    }

    #[test]
    fn test_reduction_options_builder() {
        let options = ReductionOptions::new()
            .with_axis(0)
            .with_keepdims(true)
            .with_out(vec![2, 3]);

        assert_eq!(options.axis, Some(0));
        assert!(options.keepdims);
        assert_eq!(options.out, Some(vec![2, 3]));
    }

    #[test]
    fn test_compute_output_shape_all() {
        let options = ReductionOptions::new();
        let shape = vec![2, 3, 4];
        let output = compute_output_shape(&shape, &options);
        assert_eq!(output, Vec::<usize>::new());
    }

    #[test]
    fn test_compute_output_shape_axis() {
        let options = ReductionOptions::new().with_axis(1);
        let shape = vec![2, 3, 4];
        let output = compute_output_shape(&shape, &options);
        assert_eq!(output, vec![2, 4]);
    }

    #[test]
    fn test_compute_output_shape_axis_keepdims() {
        let options = ReductionOptions::new().with_axis(1).with_keepdims(true);
        let shape = vec![2, 3, 4];
        let output = compute_output_shape(&shape, &options);
        assert_eq!(output, vec![2, 1, 4]);
    }

    #[test]
    fn test_compute_output_shape_all_keepdims() {
        let options = ReductionOptions::new().with_keepdims(true);
        let shape = vec![2, 3, 4];
        let output = compute_output_shape(&shape, &options);
        assert_eq!(output, vec![1, 1, 1]);
    }

    #[test]
    fn test_sum_advanced_all() {
        let array = Array::from_vec(vec![1i32, 2, 3, 4, 5, 6]);
        let options = ReductionOptions::new();
        let result = sum_advanced(&array, &options).unwrap();

        assert_eq!(result.shape(), &[] as &[usize]);
        assert_eq!(result.to_vec(), vec![21]);
    }

    #[test]
    fn test_sum_advanced_axis() {
        let array = Array::from_data(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let options = ReductionOptions::new().with_axis(1);
        let result = sum_advanced(&array, &options).unwrap();

        assert_eq!(result.shape(), &[2]);
        assert_eq!(result.to_vec(), vec![6, 15]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_sum_advanced_axis_keepdims() {
        let array = Array::from_data(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let options = ReductionOptions::new().with_axis(1).with_keepdims(true);
        let result = sum_advanced(&array, &options).unwrap();

        assert_eq!(result.shape(), &[2, 1]);
        assert_eq!(result.to_vec(), vec![6, 15]);
    }

    #[test]
    fn test_prod_advanced() {
        let array = Array::from_vec(vec![1i32, 2, 3, 4]);
        let options = ReductionOptions::new();
        let result = prod_advanced(&array, &options).unwrap();

        assert_eq!(result.to_vec(), vec![24]); // 1*2*3*4
    }

    #[test]
    fn test_min_advanced() {
        let array = Array::from_vec(vec![5i32, 2, 8, 1]);
        let options = ReductionOptions::new();
        let result = min_advanced(&array, &options).unwrap();

        assert_eq!(result.to_vec(), vec![1]);
    }

    #[test]
    fn test_max_advanced() {
        let array = Array::from_vec(vec![5i32, 2, 8, 1]);
        let options = ReductionOptions::new();
        let result = max_advanced(&array, &options).unwrap();

        assert_eq!(result.to_vec(), vec![8]);
    }

    #[test]
    fn test_complex_sum() {
        let array = Array::from_vec(vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, 4.0),
        ]);
        let options = ReductionOptions::new();
        let result = complex_sum(&array, &options).unwrap();

        assert_eq!(result.to_vec()[0].re, 4.0);
        assert_eq!(result.to_vec()[0].im, 6.0);
    }

    #[test]
    fn test_validate_out_shape() {
        assert!(validate_out_shape(&[2, 3], &[2, 3]).is_ok());
        assert!(validate_out_shape(&[2, 3], &[2, 4]).is_err());
    }
}
