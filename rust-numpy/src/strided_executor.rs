/// Strided execution engines for optimized array operations
use crate::array::Array;
use crate::error::{NumPyError, Result};

/// Strided binary executor for element-wise operations
pub struct StridedBinaryExecutor;

impl StridedBinaryExecutor {
    pub fn execute<T, F>(
        lhs: &Array<T>,
        rhs: &Array<T>,
        output: &mut Array<T>,
        operation: F,
    ) -> Result<()>
    where
        T: Clone + 'static,
        F: Fn(T, T) -> T,
    {
        // Basic implementation - would be optimized with actual strided execution
        if lhs.shape() != rhs.shape() || lhs.shape() != output.shape() {
            return Err(NumPyError::shape_mismatch(
                lhs.shape().to_vec(),
                rhs.shape().to_vec(),
            ));
        }

        for i in 0..lhs.size() {
            if let (Some(a), Some(b)) = (lhs.get(i), rhs.get(i)) {
                let result = operation(a.clone(), b.clone());
                output.set(i, result)?;
            }
        }

        Ok(())
    }
}

/// Strided unary executor for element-wise operations
pub struct StridedUnaryExecutor;

impl StridedUnaryExecutor {
    pub fn execute<T, F>(input: &Array<T>, output: &mut Array<T>, operation: F) -> Result<()>
    where
        T: Clone + 'static,
        F: Fn(T) -> T,
    {
        // Basic implementation - would be optimized with actual strided execution
        if input.shape() != output.shape() {
            return Err(NumPyError::shape_mismatch(
                input.shape().to_vec(),
                output.shape().to_vec(),
            ));
        }

        for i in 0..input.size() {
            if let Some(val) = input.get(i) {
                let result = operation(val.clone());
                output.set(i, result)?;
            }
        }

        Ok(())
    }
}

/// Strided reduction executor for reduction operations
pub struct StridedReductionExecutor;

impl StridedReductionExecutor {
    pub fn execute<T, F, R>(
        input: &Array<T>,
        axis: Option<usize>,
        operation: F,
        identity: R,
    ) -> Result<Array<R>>
    where
        T: Clone + 'static,
        R: Clone + Default + 'static,
        F: Fn(R, T) -> R,
    {
        // Basic implementation - would be optimized with actual strided execution
        match axis {
            Some(axis) => {
                if axis >= input.ndim() {
                    return Err(NumPyError::value_error(
                        format!(
                            "Axis {} out of bounds for array of dimension {}",
                            axis,
                            input.ndim()
                        ),
                        "",
                    ));
                }
                // Simplified reduction along axis
                let mut result = identity;
                for i in 0..input.size() {
                    if let Some(val) = input.get(i) {
                        result = operation(result, val.clone());
                    }
                }
                Ok(Array::from_vec(vec![result]))
            }
            None => {
                // Global reduction
                let mut result = identity;
                for i in 0..input.size() {
                    if let Some(val) = input.get(i) {
                        result = operation(result, val.clone());
                    }
                }
                Ok(Array::from_shape_vec(vec![], vec![result]))
            }
        }
    }
}

/// Factory for creating strided executors
pub struct StridedExecutorFactory;

impl StridedExecutorFactory {
    pub fn create_binary() -> StridedBinaryExecutor {
        StridedBinaryExecutor
    }

    pub fn create_unary() -> StridedUnaryExecutor {
        StridedUnaryExecutor
    }

    pub fn create_reduction() -> StridedReductionExecutor {
        StridedReductionExecutor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_binary_executor() {
        let lhs = Array::from_data(vec![1, 2, 3], vec![3]);
        let rhs = Array::from_data(vec![4, 5, 6], vec![3]);
        let mut output = Array::from_data(vec![0; 3], vec![3]);

        let result = StridedBinaryExecutor::execute(&lhs, &rhs, &mut output, |a, b| a + b);

        assert!(result.is_ok());
        assert_eq!(output.get(0), Some(&5));
        assert_eq!(output.get(1), Some(&7));
        assert_eq!(output.get(2), Some(&9));
    }

    #[test]
    fn test_strided_unary_executor() {
        let input = Array::from_data(vec![1, 2, 3], vec![3]);
        let mut output = Array::from_data(vec![0; 3], vec![3]);

        let result = StridedUnaryExecutor::execute(&input, &mut output, |x| x * 2);

        assert!(result.is_ok());
        assert_eq!(output.get(0), Some(&2));
        assert_eq!(output.get(1), Some(&4));
        assert_eq!(output.get(2), Some(&6));
    }

    #[test]
    fn test_strided_reduction_executor() {
        let input = Array::from_data(vec![1, 2, 3, 4], vec![2, 2]);

        let result = StridedReductionExecutor::execute(&input, None, |acc, val| acc + val, 0i32);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.get(0), Some(&10));
    }
}
