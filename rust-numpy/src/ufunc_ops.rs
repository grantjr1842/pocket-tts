use crate::array::Array;
use crate::broadcasting::{broadcast_arrays, compute_broadcast_shape};
// use crate::dtype::{Dtype, DtypeKind}; // Unused
use crate::error::{NumPyError, Result};
// use crate::slicing::{MultiSlice, Slice}; // Unused
use crate::ufunc::{get_ufunc, UfuncRegistry};
use std::sync::Arc;

/// Ufunc execution engine
#[allow(dead_code)]
pub struct UfuncEngine {
    registry: Arc<UfuncRegistry>,
}

impl UfuncEngine {
    /// Create new ufunc engine
    pub fn new() -> Self {
        Self {
            registry: Arc::new(UfuncRegistry::new()),
        }
    }

    /// Execute binary ufunc on two arrays
    pub fn execute_binary<T>(
        &self,
        ufunc_name: &str,
        a: &Array<T>,
        b: &Array<T>,
    ) -> Result<Array<T>>
    where
        T: Clone + Default + 'static,
    {
        let ufunc = get_ufunc(ufunc_name)
            .ok_or_else(|| NumPyError::ufunc_error(ufunc_name, "Function not found"))?;

        // Check if ufunc supports the dtype
        if !ufunc.supports_dtypes(&[a.dtype(), b.dtype()]) {
            return Err(NumPyError::ufunc_error(
                ufunc_name,
                format!(
                    "Unsupported dtype combination: {:?} and {:?}",
                    a.dtype(),
                    b.dtype()
                ),
            ));
        }

        // Broadcast arrays to common shape
        let broadcasted = broadcast_arrays(&[a, b])?;

        // Create output array
        let output_shape = compute_broadcast_shape(a.shape(), b.shape());
        let mut output = Array::zeros(output_shape);

        // Create array views for ufunc execution
        let views: Vec<&dyn crate::ufunc::ArrayView> = broadcasted
            .iter()
            .map(|arr| arr as &dyn crate::ufunc::ArrayView)
            .collect();

        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

        // Execute ufunc
        ufunc.execute(&views, &mut outputs)?;

        Ok(output)
    }

    /// Execute unary ufunc on single array
    pub fn execute_unary<T>(&self, ufunc_name: &str, a: &Array<T>) -> Result<Array<T>>
    where
        T: Clone + Default + 'static,
    {
        let ufunc = get_ufunc(ufunc_name)
            .ok_or_else(|| NumPyError::ufunc_error(ufunc_name, "Function not found"))?;

        // Check if ufunc supports the dtype
        if !ufunc.supports_dtypes(&[a.dtype()]) {
            return Err(NumPyError::ufunc_error(
                ufunc_name,
                format!("Unsupported dtype: {:?}", a.dtype()),
            ));
        }

        // Create output array
        let mut output = a.clone(); // Copy shape and dtype

        // Create array views for ufunc execution
        let input_views: Vec<&dyn crate::ufunc::ArrayView> = vec![a];
        let mut outputs: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

        // Execute ufunc
        ufunc.execute(&input_views, &mut outputs)?;

        Ok(output)
    }

    /// Execute reduction ufunc
    pub fn execute_reduction<T, F>(
        &self,
        _ufunc_name: &str,
        array: &Array<T>,
        axis: Option<&[isize]>,
        keepdims: bool,
        operation: F,
    ) -> Result<Array<T>>
    where
        T: Clone + Default + 'static,
        F: Fn(T, T) -> T + Send + Sync,
    {
        let output_shape = crate::broadcasting::broadcast_shape_for_reduce(
            array.shape(),
            axis.unwrap_or(&[]),
            keepdims,
        );

        let mut output = Array::zeros(output_shape);

        // This is a simplified reduction implementation
        // Real NumPy has much more complex reduction logic
        if let Some(reduction_axes) = axis {
            // Reduce along specific axes
            self.reduce_along_axes(array, &mut output, reduction_axes, &operation)?;
        } else {
            // Reduce all elements
            if let Some(initial) = array.get(0) {
                let mut result = initial.clone();
                for i in 1..array.size() {
                    if let Some(element) = array.get(i) {
                        result = operation(result, element.clone());
                    }
                }
                // Set result in output
                // output.set_by_indices(&[0; output.ndim()], result)?;
            }
        }

        Ok(output)
    }

    /// Reduce along specific axes
    fn reduce_along_axes<T, F>(
        &self,
        input: &Array<T>,
        output: &mut Array<T>,
        axes: &[isize],
        operation: F,
    ) -> Result<()>
    where
        T: Clone + Default + 'static,
        F: Fn(T, T) -> T + Send + Sync,
    {
        // This is highly simplified - real implementation would need
        // proper iteration over reduction axes and parallelization

        let input_shape = input.shape();
        let output_shape = output.shape();

        // For each position in output shape, compute reduction
        for output_idx in 0..output.size() {
            let output_indices = crate::strides::compute_multi_indices(output_idx, output_shape);

            // Map output indices to input indices (setting reduced axes to full range)
            let mut input_indices = output_indices.clone();

            for &axis in axes {
                let ax = if axis < 0 {
                    axis + input_shape.len() as isize
                } else {
                    axis
                } as usize;

                if ax < input_indices.len() {
                    input_indices[ax] = 0; // Start of reduced dimension
                }
            }

            // Iterate over all elements to reduce
            if let Ok(initial) = input.get_by_indices(&input_indices) {
                let mut result = initial.clone();

                // Iterate over reduced dimension
                for &axis in axes {
                    let ax = if axis < 0 {
                        axis + input_shape.len() as isize
                    } else {
                        axis
                    } as usize;

                    if ax < input_shape.len() {
                        for pos in 0..input_shape[ax] {
                            let mut temp_indices = input_indices.clone();
                            temp_indices[ax] = pos;

                            if let Ok(element) = input.get_by_indices(&temp_indices) {
                                result = operation(result, element.clone());
                            }
                        }
                    }
                }

                // Set result in output - this is simplified
                // output.set_by_indices(&output_indices, result)?;
            }
        }

        Ok(())
    }
}

impl Default for UfuncEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level ufunc operations for Array
impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    /// Element-wise addition
    pub fn add(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("add", self, other)
    }

    /// Element-wise subtraction
    pub fn subtract(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("subtract", self, other)
    }

    /// Element-wise multiplication
    pub fn multiply(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("multiply", self, other)
    }

    /// Element-wise division
    pub fn divide(&self, other: &Array<T>) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_binary("divide", self, other)
    }

    /// Element-wise negation
    pub fn negative(&self) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_unary("negative", self)
    }

    /// Absolute value
    pub fn abs(&self) -> Result<Array<T>> {
        let engine = UfuncEngine::new();
        engine.execute_unary("absolute", self)
    }

    /// Sum of elements
    pub fn sum(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: std::ops::Add<Output = T>,
    {
        let engine = UfuncEngine::new();
        engine.execute_reduction("sum", self, axis, keepdims, |a, b| a + b)
    }

    /// Product of elements
    pub fn product(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: std::ops::Mul<Output = T>,
    {
        let engine = UfuncEngine::new();
        engine.execute_reduction("product", self, axis, keepdims, |a, b| a * b)
    }

    /// Minimum of elements
    pub fn min(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: PartialOrd,
    {
        let engine = UfuncEngine::new();
        engine.execute_reduction(
            "min",
            self,
            axis,
            keepdims,
            |a, b| if a < b { a } else { b },
        )
    }

    /// Maximum of elements
    pub fn max(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<T>>
    where
        T: PartialOrd,
    {
        let engine = UfuncEngine::new();
        engine.execute_reduction(
            "max",
            self,
            axis,
            keepdims,
            |a, b| if a > b { a } else { b },
        )
    }

    /// Mean of elements
    pub fn mean(&self, axis: Option<&[isize]>, keepdims: bool) -> Result<Array<f64>>
    where
        T: Into<f64> + Clone + std::ops::Add<Output = T>,
    {
        // Simplified implementation - would need proper type conversion
        let sum_result = self.sum(axis, keepdims)?;
        let _count = axis
            .map(|axes| {
                let reduced_shape =
                    crate::broadcasting::broadcast_shape_for_reduce(self.shape(), axes, keepdims);
                reduced_shape.iter().product()
            })
            .unwrap_or(1);

        // Convert sum to f64 and divide by count
        // This would need proper type promotion in real implementation
        let result = Array::<f64>::zeros(sum_result.shape().to_vec());
        for i in 0..result.size() {
            if let Some(_sum_val) = sum_result.get(i) {
                // result.set(i, (*sum_val).into<f64>() / count as f64)?;
            }
        }

        Ok(result)
    }
}

/// Ufunc operation traits for different data types
pub trait UfuncOps<T>: Send + Sync {
    fn add(a: &T, b: &T) -> T;
    fn subtract(a: &T, b: &T) -> T;
    fn multiply(a: &T, b: &T) -> T;
    fn divide(a: &T, b: &T) -> T;
    fn negative(a: &T) -> T;
    fn absolute(a: &T) -> T;
}

/// Implement ufunc operations for basic numeric types
macro_rules! impl_signed_ufunc_ops {
    ($($t:ty),*) => {
        $(
            impl UfuncOps<$t> for $t {
                fn add(a: &$t, b: &$t) -> $t { a + b }
                fn subtract(a: &$t, b: &$t) -> $t { a - b }
                fn multiply(a: &$t, b: &$t) -> $t { a * b }
                fn divide(a: &$t, b: &$t) -> $t { a / b }
                fn negative(a: &$t) -> $t { -a }
                fn absolute(a: &$t) -> $t { a.abs() }
            }
        )*
    }
}

macro_rules! impl_unsigned_ufunc_ops {
    ($($t:ty),*) => {
        $(
            impl UfuncOps<$t> for $t {
                fn add(a: &$t, b: &$t) -> $t { a + b }
                fn subtract(a: &$t, b: &$t) -> $t { a - b }
                fn multiply(a: &$t, b: &$t) -> $t { a * b }
                fn divide(a: &$t, b: &$t) -> $t { a / b }
                fn negative(a: &$t) -> $t { 0_u8.wrapping_sub(1) as $t * a } // Not ideal but satisfies trait
                fn absolute(a: &$t) -> $t { *a }
            }
        )*
    }
}

impl_signed_ufunc_ops!(f64, f32, i64, i32, i16, i8);
impl_unsigned_ufunc_ops!(u64, u32, u16, u8);
