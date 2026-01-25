//! Memory Layout Optimization for Strided Operations
//!
//! This module provides optimization for different memory layouts (C-order vs F-order)
//! and improves performance for non-contiguous arrays through intelligent access patterns.

use crate::array::Array;

/// Memory layout type for arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Row-major (C-style) layout - last dimension changes fastest
    COrder,
    /// Column-major (Fortran-style) layout - first dimension changes fastest
    FOrder,
    /// Non-contiguous layout with custom strides
    NonContiguous,
}

/// Type of operation being performed
#[derive(Debug, Clone, Copy)]
pub enum OperationType {
    /// Element-wise operation
    ElementWise,
    /// Reduction operation (sum, mean, etc.)
    Reduction,
    /// Broadcasting operation
    Broadcast,
    /// Strided operation with custom access pattern
    Strided,
}

/// Optimal access pattern for a given operation
#[derive(Debug, Clone)]
pub struct AccessPattern {
    /// Whether to iterate forward or backward
    pub forward: bool,
    /// Suggested stride for optimal cache performance
    pub stride: isize,
    /// Whether transposition would improve performance
    pub should_transpose: bool,
    /// Cache line alignment recommendation
    pub alignment: usize,
}

/// Layout optimizer for determining optimal access patterns
pub struct LayoutOptimizer {
    preferred_layout: MemoryLayout,
    alignment: usize,
    cache_line_size: usize,
}

impl LayoutOptimizer {
    /// Create a new layout optimizer with default settings
    pub fn new() -> Self {
        Self {
            preferred_layout: MemoryLayout::COrder,
            alignment: 64, // Typical cache line size
            cache_line_size: 64,
        }
    }

    /// Create a layout optimizer with custom alignment
    pub fn with_alignment(alignment: usize) -> Self {
        Self {
            preferred_layout: MemoryLayout::COrder,
            alignment,
            cache_line_size: 64,
        }
    }

    /// Detect the memory layout of an array
    pub fn detect_layout<T>(&self, array: &Array<T>) -> MemoryLayout {
        if array.is_c_contiguous() {
            MemoryLayout::COrder
        } else if array.is_f_contiguous() {
            MemoryLayout::FOrder
        } else {
            MemoryLayout::NonContiguous
        }
    }

    /// Determine optimal access pattern for an operation
    pub fn optimize_access_pattern<T>(
        &self,
        array: &Array<T>,
        operation: OperationType,
    ) -> AccessPattern {
        let layout = self.detect_layout(array);

        match (layout, operation) {
            (MemoryLayout::COrder, OperationType::ElementWise) => AccessPattern {
                forward: true,
                stride: 1,
                should_transpose: false,
                alignment: self.alignment,
            },
            (MemoryLayout::FOrder, OperationType::ElementWise) => AccessPattern {
                forward: true,
                stride: 1,
                should_transpose: false,
                alignment: self.alignment,
            },
            (MemoryLayout::NonContiguous, _) => {
                // For non-contiguous arrays, analyze strides to determine optimal pattern
                self.optimize_strided_access(array)
            }
            (_, OperationType::Reduction) => {
                // For reductions, prefer contiguous memory access
                AccessPattern {
                    forward: true,
                    stride: 1,
                    should_transpose: false,
                    alignment: self.alignment,
                }
            }
            (_, OperationType::Broadcast) => {
                // For broadcasting, consider transposition for better cache usage
                self.optimize_broadcast_access(array)
            }
            (_, OperationType::Strided) => {
                self.optimize_strided_access(array)
            }
        }
    }

    /// Optimize access pattern for strided/non-contiguous arrays
    fn optimize_strided_access<T>(&self, array: &Array<T>) -> AccessPattern {
        let shape = array.shape();
        let strides = array.strides();

        if shape.len() < 2 {
            return AccessPattern {
                forward: true,
                stride: 1,
                should_transpose: false,
                alignment: self.alignment,
            };
        }

        // Check if last dimension is contiguous (better for cache)
        let last_dim_contiguous = strides[strides.len() - 1] == 1;

        AccessPattern {
            forward: last_dim_contiguous,
            stride: if last_dim_contiguous { 1 } else { strides[strides.len() - 1] },
            should_transpose: self.should_transpose_for_cache(shape),
            alignment: self.alignment,
        }
    }

    /// Optimize access pattern for broadcasting operations
    fn optimize_broadcast_access<T>(&self, array: &Array<T>) -> AccessPattern {
        let shape = array.shape();

        AccessPattern {
            forward: true,
            stride: 1,
            should_transpose: self.should_transpose_for_cache(shape),
            alignment: self.alignment,
        }
    }

    /// Determine if transposition would improve cache performance
    pub fn should_transpose_for_cache(&self, shape: &[usize]) -> bool {
        if shape.len() < 2 {
            return false;
        }

        // Simple heuristic: if innermost dimension is small, transposing might help
        // by making a larger dimension contiguous
        let innermost = shape[shape.len() - 1];
        let outer = shape[0];

        // Transpose if innermost dimension is much smaller than outer
        innermost < outer / 4 && innermost < 16
    }

    /// Calculate expected cache misses for a given access pattern
    pub fn estimate_cache_misses<T>(&self, array: &Array<T>, access_pattern: &AccessPattern) -> usize {
        let total_elements = array.size();
        let cache_line_capacity = self.cache_line_size / std::mem::size_of::<T>();

        // Estimate cache lines needed
        let cache_lines = (total_elements + cache_line_capacity - 1) / cache_line_capacity;

        // Adjust for stride (non-unit stride increases cache misses)
        if access_pattern.stride > 1 {
            cache_lines * access_pattern.stride as usize
        } else {
            cache_lines
        }
    }

    /// Get memory alignment requirement for SIMD operations
    pub fn simd_alignment_requirement<T>(&self) -> usize {
        // For SIMD, we typically need 32-byte alignment for AVX2
        #[cfg(target_arch = "x86_64")]
        {
            32
        }

        #[cfg(target_arch = "aarch64")]
        {
            16 // NEON typically uses 16-byte alignment
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            8 // Default to 8-byte alignment
        }
    }

    /// Check if array is aligned for SIMD operations
    pub fn is_simd_aligned<T>(&self, array: &Array<T>) -> bool {
        // This is a simplified check - in practice, we'd need to check the actual
        // memory address alignment
        let alignment = self.simd_alignment_requirement::<T>();
        array.size() >= alignment
    }
}

impl Default for LayoutOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator for cache-friendly array traversal
pub struct CacheFriendlyIterator<'a, T> {
    array: &'a Array<T>,
    index: usize,
    stride: isize,
    forward: bool,
}

impl<'a, T> CacheFriendlyIterator<'a, T> {
    /// Create a new cache-friendly iterator
    pub fn new(array: &'a Array<T>, access_pattern: &AccessPattern) -> Self {
        Self {
            array,
            index: 0,
            stride: access_pattern.stride,
            forward: access_pattern.forward,
        }
    }
}

impl<'a, T> Iterator for CacheFriendlyIterator<'a, T>
where
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.array.size() {
            return None;
        }

        let result = self.array.get(self.index).cloned();
        self.index += 1;
        result
    }
}

/// Memory prefetch hints for strided operations
pub mod prefetch {
    /// Prefetch distance in elements
    pub const DEFAULT_PREFETCH_DISTANCE: usize = 8;

    /// Suggest prefetch distance based on stride
    pub fn suggest_prefetch_distance(stride: isize) -> usize {
        if stride.abs() > 16 {
            DEFAULT_PREFETCH_DISTANCE * 2
        } else {
            DEFAULT_PREFETCH_DISTANCE
        }
    }

    /// Determine if prefetching is beneficial for a given access pattern
    pub fn should_prefetch(stride: isize, array_size: usize) -> bool {
        // Prefetching is beneficial for:
        // - Large arrays
        // - Non-unit strides
        // - Regular access patterns

        array_size > 1024 && stride.abs() > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_optimizer_creation() {
        let optimizer = LayoutOptimizer::new();
        assert_eq!(optimizer.alignment, 64);
        assert_eq!(optimizer.cache_line_size, 64);
    }

    #[test]
    fn test_detect_c_order_layout() {
        let optimizer = LayoutOptimizer::new();
        let array = Array::from_vec(vec![1i32, 2, 3, 4, 5, 6]);

        assert_eq!(optimizer.detect_layout(&array), MemoryLayout::COrder);
    }

    #[test]
    fn test_optimize_access_pattern_c_order() {
        let optimizer = LayoutOptimizer::new();
        let array = Array::from_vec(vec![1i32, 2, 3, 4]);

        let pattern = optimizer.optimize_access_pattern(&array, OperationType::ElementWise);

        assert!(pattern.forward);
        assert_eq!(pattern.stride, 1);
        assert!(!pattern.should_transpose);
    }

    #[test]
    fn test_should_transpose_for_cache() {
        let optimizer = LayoutOptimizer::new();

        // Small innermost dimension, large outer - should transpose
        assert!(optimizer.should_transpose_for_cache(&[1000, 8]));

        // Equal dimensions - no transpose needed
        assert!(!optimizer.should_transpose_for_cache(&[100, 100]));

        // Large innermost dimension - no transpose needed
        assert!(!optimizer.should_transpose_for_cache(&[8, 1000]));
    }

    #[test]
    fn test_estimate_cache_misses() {
        let optimizer = LayoutOptimizer::new();
        let array = Array::from_vec(vec![1i32; 256]);

        let pattern = AccessPattern {
            forward: true,
            stride: 1,
            should_transpose: false,
            alignment: 64,
        };

        let misses = optimizer.estimate_cache_misses(&array, &pattern);

        // Should have some cache misses for 256 elements
        assert!(misses > 0);
        assert!(misses <= 256);
    }

    #[test]
    fn test_cache_friendly_iterator() {
        let array = Array::from_vec(vec![1i32, 2, 3, 4, 5]);
        let pattern = AccessPattern {
            forward: true,
            stride: 1,
            should_transpose: false,
            alignment: 64,
        };

        let iter = CacheFriendlyIterator::new(&array, &pattern);
        let values: Vec<i32> = iter.collect();

        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_prefetch_suggestions() {
        assert_eq!(
            prefetch::suggest_prefetch_distance(1),
            prefetch::DEFAULT_PREFETCH_DISTANCE
        );
        assert_eq!(
            prefetch::suggest_prefetch_distance(32),
            prefetch::DEFAULT_PREFETCH_DISTANCE * 2
        );
    }

    #[test]
    fn test_should_prefetch() {
        assert!(prefetch::should_prefetch(2, 2000));
        assert!(!prefetch::should_prefetch(1, 100));
    }
}
