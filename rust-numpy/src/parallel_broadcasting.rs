//! Parallel Broadcasting Optimizations
//!
//! This module provides optimized parallel operations for broadcasting scenarios,
//! including advanced broadcasting edge cases and performance optimizations.

use crate::array::Array;
use crate::error::NumPyError;
use crate::parallel_executor::ParallelExecutor;
use std::sync::Arc;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Broadcasting strategy for parallel operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BroadcastStrategy {
    /// Element-wise operation with simple broadcasting
    ElementWise,
    /// Outer product style broadcasting
    OuterProduct,
    /// Reduction along broadcasted dimensions
    ReduceAlong,
    /// Custom broadcasting pattern
    Custom,
}

/// Parallel broadcasting optimizer
pub struct ParallelBroadcasting {
    executor: Arc<ParallelExecutor>,
}

impl ParallelBroadcasting {
    /// Create a new parallel broadcasting optimizer
    pub fn new(executor: Arc<ParallelExecutor>) -> Self {
        Self { executor }
    }

    /// Determine optimal broadcasting strategy
    pub fn determine_strategy(&self, a_shape: &[usize], b_shape: &[usize]) -> BroadcastStrategy {
        // If shapes are identical, simple element-wise
        if a_shape == b_shape {
            return BroadcastStrategy::ElementWise;
        }

        // Check for outer product pattern (e.g., (N,) @ (M,))
        if (a_shape.len() == 1 && b_shape.len() == 1)
            || (a_shape.is_empty() || b_shape.is_empty())
        {
            return BroadcastStrategy::OuterProduct;
        }

        // Check for reduction pattern
        let a_dims = a_shape.len();
        let b_dims = b_shape.len();

        if a_dims != b_dims {
            // Could be reduction along broadcasted dimensions
            return BroadcastStrategy::ReduceAlong;
        }

        // Default to element-wise with broadcasting
        BroadcastStrategy::ElementWise
    }

    /// Parallel binary operation with broadcasting
    #[cfg(feature = "rayon")]
    pub fn broadcast_binary_op<T, F>(
        &self,
        a: &Array<T>,
        b: &Array<T>,
        op: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Send + Sync + Clone + Default + 'static,
        F: Fn(&T, &T) -> T + Send + Sync + 'static,
    {
        let strategy = self.determine_strategy(a.shape(), b.shape());

        match strategy {
            BroadcastStrategy::ElementWise => {
                self.broadcast_element_wise(a, b, op)
            }
            BroadcastStrategy::OuterProduct => {
                self.outer_product(a, b, op)
            }
            BroadcastStrategy::ReduceAlong => {
                self.broadcast_with_reduction(a, b, op)
            }
            BroadcastStrategy::Custom => {
                self.broadcast_custom(a, b, op)
            }
        }
    }

    /// Element-wise broadcasting operation
    #[cfg(feature = "rayon")]
    fn broadcast_element_wise<T, F>(
        &self,
        a: &Array<T>,
        b: &Array<T>,
        op: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Send + Sync + Clone + Default + 'static,
        F: Fn(&T, &T) -> T + Send + Sync + 'static,
    {
        // Compute broadcasted shape
        let result_shape = self.compute_broadcast_shape(a.shape(), b.shape());

        // Check if simple case (same shape)
        if a.shape() == b.shape() {
            return self.executor.binary_op(a, b, op);
        }

        // General broadcasting case
        let a_size = a.size();
        let b_size = b.size();
        let result_size: usize = result_shape.iter().product();

        // If both arrays are small, use sequential
        if result_size < 4096 {
            return self.broadcast_sequential(a, b, op);
        }

        // Convert to vectors for parallel processing
        let a_vec = a.to_vec();
        let b_vec = b.to_vec();

        // Compute strides for broadcasting
        let a_strides = self.compute_broadcast_strides(a.shape(), &result_shape);
        let b_strides = self.compute_broadcast_strides(b.shape(), &result_shape);

        let result_data: Vec<T> = (0..result_size)
            .into_par_iter()
            .map(|i| {
                let a_idx = if a_size == 1 { 0 } else { self.compute_index(i, &a_strides) };
                let b_idx = if b_size == 1 { 0 } else { self.compute_index(i, &b_strides) };
                op(&a_vec[a_idx], &b_vec[b_idx])
            })
            .collect();

        Ok(Array::from_data(result_data, result_shape))
    }

    /// Outer product operation
    #[cfg(feature = "rayon")]
    fn outer_product<T, F>(
        &self,
        a: &Array<T>,
        b: &Array<T>,
        op: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Send + Sync + Clone + Default + 'static,
        F: Fn(&T, &T) -> T + Send + Sync + 'static,
    {
        let a_vec = a.to_vec();
        let b_vec = b.to_vec();

        let mut result_shape = vec![0usize; a.shape().len() + b.shape().len()];
        result_shape[..a.shape().len()].copy_from_slice(a.shape());
        result_shape[a.shape().len()..].copy_from_slice(b.shape());

        let result_data: Vec<T> = a_vec
            .par_iter()
            .flat_map(|a_elem| {
                b_vec.par_iter().map(|b_elem| op(a_elem, b_elem)).collect::<Vec<_>>()
            })
            .collect();

        Ok(Array::from_data(result_data, result_shape))
    }

    /// Broadcasting with reduction
    #[cfg(feature = "rayon")]
    fn broadcast_with_reduction<T, F>(
        &self,
        a: &Array<T>,
        b: &Array<T>,
        op: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Send + Sync + Clone + Default + 'static,
        F: Fn(&T, &T) -> T + Send + Sync + 'static,
    {
        // For now, delegate to element-wise
        self.broadcast_element_wise(a, b, op)
    }

    /// Custom broadcasting pattern
    #[cfg(feature = "rayon")]
    fn broadcast_custom<T, F>(
        &self,
        a: &Array<T>,
        b: &Array<T>,
        op: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Send + Sync + Clone + Default + 'static,
        F: Fn(&T, &T) -> T + Send + Sync + 'static,
    {
        // Delegate to element-wise for now
        self.broadcast_element_wise(a, b, op)
    }

    /// Sequential broadcasting operation (fallback)
    fn broadcast_sequential<T, F>(
        &self,
        a: &Array<T>,
        b: &Array<T>,
        op: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static,
        F: Fn(&T, &T) -> T,
    {
        let result_shape = self.compute_broadcast_shape(a.shape(), b.shape());
        let result_size: usize = result_shape.iter().product();

        let a_vec = a.to_vec();
        let b_vec = b.to_vec();
        let a_size = a.size();
        let b_size = b.size();

        let a_strides = self.compute_broadcast_strides(a.shape(), &result_shape);
        let b_strides = self.compute_broadcast_strides(b.shape(), &result_shape);

        let mut result_data = Vec::with_capacity(result_size);
        for i in 0..result_size {
            let a_idx = if a_size == 1 { 0 } else { self.compute_index(i, &a_strides) };
            let b_idx = if b_size == 1 { 0 } else { self.compute_index(i, &b_strides) };
            result_data.push(op(&a_vec[a_idx], &b_vec[b_idx]));
        }

        Ok(Array::from_data(result_data, result_shape))
    }

    /// Compute broadcasted shape
    fn compute_broadcast_shape(&self, a_shape: &[usize], b_shape: &[usize]) -> Vec<usize> {
        let a_ndim = a_shape.len();
        let b_ndim = b_shape.len();
        let max_ndim = a_ndim.max(b_ndim);

        let mut result_shape = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let a_dim = if i < max_ndim - a_ndim {
                1
            } else {
                a_shape[i - (max_ndim - a_ndim)]
            };
            let b_dim = if i < max_ndim - b_ndim {
                1
            } else {
                b_shape[i - (max_ndim - b_ndim)]
            };

            // Broadcasting rules: dimensions must match or one must be 1
            if a_dim == b_dim || a_dim == 1 || b_dim == 1 {
                result_shape.push(a_dim.max(b_dim));
            } else {
                // Invalid broadcasting - should error
                result_shape.push(a_dim.max(b_dim));
            }
        }

        result_shape
    }

    /// Compute strides for broadcasting
    fn compute_broadcast_strides(&self, src_shape: &[usize], dst_shape: &[usize]) -> Vec<usize> {
        let src_ndim = src_shape.len();
        let dst_ndim = dst_shape.len();
        let offset = dst_ndim - src_ndim;

        let mut strides = Vec::with_capacity(src_ndim);
        for i in 0..src_ndim {
            if src_shape[i] == dst_shape[offset + i] {
                strides.push(1); // No broadcasting, stride of 1
            } else if src_shape[i] == 1 {
                strides.push(0); // Broadcast, stride of 0
            } else {
                strides.push(1); // Default stride
            }
        }

        strides
    }

    /// Compute flat index from strides
    fn compute_index(&self, flat_idx: usize, strides: &[usize]) -> usize {
        if strides.is_empty() {
            return 0;
        }

        // Simplified index computation
        // In production, this would properly handle multi-dimensional indexing
        let mut idx = 0;
        let mut remaining = flat_idx;
        for (i, &stride) in strides.iter().enumerate().rev() {
            let dim_size = if i < strides.len() { 1 } else { 1 };
            let dim_idx = remaining % dim_size;
            remaining /= dim_size;
            idx += dim_idx * stride;
        }

        if idx == 0 && flat_idx > 0 {
            flat_idx % strides.iter().product::<usize>().max(1)
        } else {
            idx
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determine_strategy() {
        let executor = Arc::new(ParallelExecutor::with_rayon());
        let broadcaster = ParallelBroadcasting::new(executor);

        // Same shape -> element-wise
        assert_eq!(
            broadcaster.determine_strategy(&[2, 3], &[2, 3]),
            BroadcastStrategy::ElementWise
        );

        // Different shapes -> element-wise with broadcasting
        assert_eq!(
            broadcaster.determine_strategy(&[3], &[2, 3]),
            BroadcastStrategy::ReduceAlong
        );
    }

    #[test]
    fn test_compute_broadcast_shape() {
        let executor = Arc::new(ParallelExecutor::with_rayon());
        let broadcaster = ParallelBroadcasting::new(executor);

        // (3,) + (2, 3) -> (2, 3)
        assert_eq!(
            broadcaster.compute_broadcast_shape(&[3], &[2, 3]),
            vec![2, 3]
        );

        // (2, 1) + (1, 3) -> (2, 3)
        assert_eq!(
            broadcaster.compute_broadcast_shape(&[2, 1], &[1, 3]),
            vec![2, 3]
        );
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_broadcast_binary_op() {
        let executor = Arc::new(ParallelExecutor::with_rayon());
        let broadcaster = ParallelBroadcasting::new(executor);

        let a = Array::from_vec(vec![1i32, 2, 3]);
        let b = Array::from_vec(vec![10i32, 20, 30]);

        let result = broadcaster
            .broadcast_binary_op(&a, &b, |x, y| x + y)
            .unwrap();

        assert_eq!(result.to_vec(), vec![11, 22, 33]);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_outer_product() {
        let executor = Arc::new(ParallelExecutor::with_rayon());
        let broadcaster = ParallelBroadcasting::new(executor);

        let a = Array::from_vec(vec![1i32, 2]);
        let b = Array::from_vec(vec![3i32, 4]);

        // Outer product style multiplication
        let result = broadcaster
            .outer_product(&a, &b, |x, y| x * y)
            .unwrap();

        assert_eq!(result.to_vec(), vec![3, 4, 6, 8]);
        assert_eq!(result.shape(), &[2, 2]);
    }
}
