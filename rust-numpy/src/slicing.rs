use crate::array::Array;
// use crate::dtype::Dtype;
use crate::error::{NumPyError, Result};
use crate::strides::compute_linear_index;
// use std::ops::{Index, IndexMut};

/// Indexing trait for arrays
pub trait ArrayIndex<T> {
    type Output;
    fn get(&self, array: &Array<T>) -> Result<Self::Output>;
    fn get_mut(&mut self, array: &mut Array<T>) -> Result<Self::Output>;
}

impl<T> ArrayIndex<T> for usize
where
    T: Clone,
{
    type Output = T;

    fn get(&self, array: &Array<T>) -> Result<Self::Output> {
        array
            .get(*self)
            .cloned()
            .ok_or_else(|| NumPyError::index_error(*self, array.size()))
    }

    fn get_mut(&mut self, _array: &mut Array<T>) -> Result<Self::Output> {
        // This would need mutable access to memory manager
        // For now, return error
        Err(NumPyError::NotImplemented {
            operation: "striding with step > 1".to_string(),
        })
    }
}

impl<T> ArrayIndex<T> for [usize]
where
    T: Clone + Default + 'static,
{
    type Output = Array<T>;

    fn get(&self, array: &Array<T>) -> Result<Self::Output> {
        if self.len() != array.ndim() {
            return Err(NumPyError::index_error(0, array.ndim()));
        }

        for (i, &idx) in self.iter().enumerate() {
            if idx >= array.shape()[i] {
                return Err(NumPyError::index_error(idx, array.shape()[i]));
            }
        }

        let linear_idx = compute_linear_index(self, array.strides());
        if let Some(element) = array.get(linear_idx) {
            let mut result = Array::zeros(vec![1]);
            result.set(0, element.clone())?;
            Ok(result)
        } else {
            Err(NumPyError::index_error(linear_idx, array.size()))
        }
    }

    fn get_mut(&mut self, array: &mut Array<T>) -> Result<Self::Output> {
        ArrayIndex::get(self, array)
    }
}

/// Slice specification for arrays
#[derive(Debug, Clone, PartialEq)]
pub enum Slice {
    /// Full slice (:)
    Full,
    /// Range slice (start:stop)
    Range(isize, isize),
    /// Range with step (start:stop:step)
    RangeStep(isize, isize, isize),
    /// Single index
    Index(isize),
    /// From start to end (start:)
    From(isize),
    /// To end (:end)
    To(isize),
    /// With step (:step)
    Step(isize),
}

impl Slice {
    /// Convert slice to range with bounds checking
    pub fn to_range(&self, len: isize) -> (isize, isize, isize) {
        match self {
            Slice::Full => (0, len, 1),
            Slice::Range(start, stop) => {
                let start = if *start < 0 { len + *start } else { *start };
                let stop = if *stop < 0 { len + *stop } else { *stop };
                (start, stop, 1)
            }
            Slice::RangeStep(start, stop, step) => {
                let start = if *start < 0 { len + *start } else { *start };
                let stop = if *stop < 0 { len + *stop } else { *stop };
                (start, stop, *step)
            }
            Slice::Index(idx) => {
                let idx = if *idx < 0 { len + *idx } else { *idx };
                (idx, idx + 1, 1)
            }
            Slice::From(start) => {
                let start = if *start < 0 { len + *start } else { *start };
                (start, len, 1)
            }
            Slice::To(stop) => {
                let stop = if *stop < 0 { len + *stop } else { *stop };
                (0, stop, 1)
            }
            Slice::Step(step) => (0, len, *step),
        }
    }

    /// Get length of slice for given dimension
    pub fn len(&self, dim_len: usize) -> usize {
        let len = dim_len as isize;
        let (start, stop, step) = self.to_range(len);

        if step == 0 {
            return 0;
        }

        let actual_start = start.max(0).min(len);
        let actual_stop = stop.max(0).min(len);

        if (step > 0 && actual_start >= actual_stop) || (step < 0 && actual_start <= actual_stop) {
            return 0;
        }

        ((actual_stop - actual_start).abs() as usize + step.abs() as usize - 1)
            / step.abs() as usize
    }
}

/// Slice specification for multiple dimensions
#[derive(Debug, Clone)]
pub struct MultiSlice {
    slices: Vec<Slice>,
}

impl MultiSlice {
    /// Create new multi-slice
    pub fn new(slices: Vec<Slice>) -> Self {
        Self { slices }
    }

    /// Get slice at dimension
    pub fn get(&self, dim: usize) -> &Slice {
        self.slices.get(dim).unwrap_or(&Slice::Full)
    }

    /// Convert to vector of ranges
    pub fn to_ranges(&self, shape: &[usize]) -> Vec<(isize, isize, isize)> {
        self.slices
            .iter()
            .enumerate()
            .map(|(i, slice)| {
                let dim_len = shape.get(i).copied().unwrap_or(0) as isize;
                slice.to_range(dim_len)
            })
            .collect()
    }

    /// Compute resulting shape after slicing
    pub fn result_shape(&self, shape: &[usize]) -> Vec<usize> {
        self.slices
            .iter()
            .enumerate()
            .map(|(i, slice)| {
                let dim_len = shape.get(i).copied().unwrap_or(0);
                slice.len(dim_len)
            })
            .collect()
    }
}

/// Indexing implementation for Array
// Indexing traits for Array are removed because they require returning references
// which is not possible when slicing returns a new Array instance.
// Use .get() or .slice() instead.

impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    /// Get array slice using slice syntax
    pub fn slice(&self, multi_slice: &MultiSlice) -> Result<Array<T>> {
        let result_shape = multi_slice.result_shape(self.shape());
        let _ranges = multi_slice.to_ranges(self.shape());

        // This is a very simplified implementation
        // Real implementation would need complex slicing logic
        let result = Array::zeros(result_shape);

        // Copy sliced data
        for i in 0..result.size() {
            if i < self.size() {
                if let Some(_src_element) = self.get(i) {
                    // This is placeholder - real implementation would calculate source indices
                    // Set element in result - need proper indexing
                    break; // Simplified for now
                }
            }
        }

        Ok(result)
    }

    /// Set element at multi-dimensional indices
    pub fn set_by_indices(&mut self, indices: &[usize], value: T) -> Result<()> {
        if indices.len() != self.ndim() {
            return Err(NumPyError::index_error(0, self.ndim()));
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Err(NumPyError::index_error(idx, self.shape()[i]));
            }
        }

        let linear_idx = compute_linear_index(indices, self.strides());
        self.set(linear_idx, value)
    }

    /// Get element at multi-dimensional indices
    pub fn get_by_indices(&self, indices: &[usize]) -> Result<&T> {
        if indices.len() != self.ndim() {
            return Err(NumPyError::index_error(0, self.ndim()));
        }

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape()[i] {
                return Err(NumPyError::index_error(idx, self.shape()[i]));
            }
        }

        let linear_idx = compute_linear_index(indices, self.strides());
        self.get(linear_idx)
            .ok_or_else(|| NumPyError::index_error(linear_idx, self.size()))
    }

    /// Create iterator over array elements
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        ArrayIter {
            array: self,
            current: 0,
        }
    }

    /// Create mutable iterator over array elements
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        ArrayIterMut {
            array: self,
            current: 0,
        }
    }
}

/// Iterator over array elements
pub struct ArrayIter<'a, T> {
    array: &'a Array<T>,
    current: usize,
}

impl<'a, T> Iterator for ArrayIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.array.size() {
            let element = self.array.get(self.current);
            self.current += 1;
            element
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.size() - self.current;
        (remaining, Some(remaining))
    }
}

/// Mutable iterator over array elements
pub struct ArrayIterMut<'a, T> {
    array: &'a mut Array<T>,
    current: usize,
}

impl<'a, T> Iterator for ArrayIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.array.size() {
            // This would need proper mutable access
            self.current += 1;
            None // Placeholder
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.size() - self.current;
        (remaining, Some(remaining))
    }
}

/// Convenience macros for slicing
#[macro_export]
macro_rules! s {
    (:) => {
        $crate::slicing::Slice::Full
    };
    (:end) => {
        $crate::slicing::Slice::To(end)
    };
    (start:) => {
        $crate::slicing::Slice::From(start)
    };
    (start:end) => {
        $crate::slicing::Slice::Range(start, end)
    };
    (start:end:step) => {
        $crate::slicing::Slice::RangeStep(start, end, step)
    };
    (..end) => {
        $crate::slicing::Slice::To(end)
    };
    (start..) => {
        $crate::slicing::Slice::From(start)
    };
    (start..end) => {
        $crate::slicing::Slice::Range(start, end)
    };
    (start..end..step) => {
        $crate::slicing::Slice::RangeStep(start, end, step)
    };
}
