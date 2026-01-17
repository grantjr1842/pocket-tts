use std::fmt;
use std::sync::Arc;

use crate::dtype::Dtype;
use crate::error::{NumPyError, Result};
use crate::memory::MemoryManager;
use crate::strides::{compute_strides, is_c_contiguous, is_f_contiguous};

/// Multi-dimensional array structure mirroring NumPy's ndarray
pub struct Array<T> {
    data: Arc<MemoryManager<T>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,
    offset: usize,
}

/// Immutable view into an array
#[allow(dead_code)]
pub struct ArrayView<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,
    offset: usize,
}

/// Mutable view into an array
#[allow(dead_code)]
pub struct ArrayViewMut<'a, T> {
    data: &'a mut [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    dtype: Dtype,
    offset: usize,
}

impl<T> Array<T> {
    /// Get array shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get strides
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Get data offset
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get dtype
    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    /// Check if array is C-contiguous
    pub fn is_c_contiguous(&self) -> bool {
        is_c_contiguous(&self.shape, &self.strides)
    }

    /// Check if array is Fortran-contiguous
    pub fn is_f_contiguous(&self) -> bool {
        is_f_contiguous(&self.shape, &self.strides)
    }

    /// Get data as slice (only works for contiguous arrays)
    pub fn as_slice(&self) -> Result<&[T]> {
        if self.is_c_contiguous() && self.offset == 0 {
            Ok(self.data.as_slice())
        } else {
            Err(NumPyError::NotContiguous)
        }
    }

    /// Get element at linear index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.size() {
            // Calculate actual memory offset
            let linear_idx = self.offset + index;
            self.data.get(linear_idx)
        } else {
            None
        }
    }

    /// Create view of this array
    pub fn view(&self) -> ArrayView<'_, T> {
        ArrayView {
            data: self.data.as_slice(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype.clone(),
            offset: self.offset,
        }
    }

    /// Clone array (copy-on-write semantics)
    pub fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype.clone(),
            offset: self.offset,
        }
    }

    /// Reshape array (returns new view if possible)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        if self.size() != new_shape.iter().product::<usize>() {
            return Err(NumPyError::shape_mismatch(vec![self.size()], new_shape));
        }

        if self.is_c_contiguous() {
            // Can create view with new strides
            let new_strides = compute_strides(&new_shape);
            Ok(Self {
                data: self.data.clone(),
                shape: new_shape,
                strides: new_strides,
                dtype: self.dtype.clone(),
                offset: self.offset,
            })
        } else {
            // Need to copy and reorganize data.
            // This part requires T: Clone + Default, so it should be in the other impl block or handled differently.
            // Since this block doesn't have T bounds, we can't call clone_contiguous() if it requires bounds.
            // We can return an error or move reshape to the bounded block?
            // Actually, reshape on non-contiguous array DOES imply copying, so T MUST be Clone.
            // So reshape should probably be in the bounded block.
            Err(NumPyError::NotContiguous) // Placeholder, see note below
        }
    }

    /// Transpose array (swap axes)
    pub fn t(&self) -> Self {
        if self.ndim() <= 1 {
            return self.clone();
        }

        let mut new_shape = self.shape.clone();
        new_shape.reverse();

        let mut new_strides = self.strides.clone();
        new_strides.reverse();

        Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype.clone(),
            offset: self.offset,
        }
    }
}

impl<T> Array<T>
where
    T: Clone + Default + 'static,
{
    /// Create new array from vector data
    pub fn from_vec(data: Vec<T>) -> Self {
        let len = data.len();
        let shape = vec![len];
        let strides = vec![1];
        let dtype = Dtype::from_type::<T>();

        Self {
            data: Arc::new(MemoryManager::from_vec(data)),
            shape,
            strides,
            dtype,
            offset: 0,
        }
    }

    /// Create array from vector and shape
    pub fn from_shape_vec(shape: Vec<usize>, data: Vec<T>) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if expected_size != data.len() {
            return Err(NumPyError::shape_mismatch(
                vec![expected_size],
                vec![data.len()],
            ));
        }

        let strides = compute_strides(&shape);
        let dtype = Dtype::from_type::<T>();

        Ok(Self {
            data: Arc::new(MemoryManager::from_vec(data)),
            shape,
            strides,
            dtype,
            offset: 0,
        })
    }

    /// Create array filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        let data = vec![T::default(); size];
        Self::from_shape_vec(shape, data).unwrap()
    }

    /// Create array filled with ones
    pub fn ones(shape: Vec<usize>) -> Self
    where
        T: From<i32>,
    {
        let size = shape.iter().product();
        let data = vec![T::from(1); size];
        Self::from_shape_vec(shape, data).unwrap()
    }

    /// Create array filled with specified value
    pub fn full(shape: Vec<usize>, value: T) -> Self {
        let size = shape.iter().product();
        let data = vec![value; size];
        Self::from_shape_vec(shape, data).unwrap()
    }

    /// Reshape array (returns new view if possible) with data copy if needed
    pub fn reshape_clone(&self, new_shape: Vec<usize>) -> Result<Self> {
        if self.size() != new_shape.iter().product::<usize>() {
            return Err(NumPyError::shape_mismatch(vec![self.size()], new_shape));
        }

        if self.is_c_contiguous() {
            self.reshape(new_shape)
        } else {
            // Need to copy and reorganize data
            self.clone_contiguous().reshape(new_shape)
        }
    }

    /// Force contiguous copy of data
    pub fn clone_contiguous(&self) -> Self {
        let mut new_data = Vec::with_capacity(self.size());

        // Copy elements in C order
        for i in 0..self.size() {
            if let Some(element) = self.get(i) {
                new_data.push(element.clone());
            }
        }

        let new_strides = compute_strides(&self.shape);
        Self {
            data: Arc::new(MemoryManager::from_vec(new_data)),
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: self.dtype.clone(),
            offset: 0,
        }
    }

    /// Get mutable element at linear index (implements copy-on-write)
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.size() {
            let linear_idx = self.offset + index;
            // Ensure unique access
            Arc::make_mut(&mut self.data).get_mut(linear_idx)
        } else {
            None
        }
    }

    /// Set element at linear index
    pub fn set(&mut self, index: usize, value: T) -> Result<()> {
        if let Some(elem) = self.get_mut(index) {
            *elem = value;
            Ok(())
        } else {
            Err(NumPyError::index_error(index, self.size()))
        }
    }

    /// Compute dot product with another array
    pub fn dot(&self, other: &Array<T>) -> Result<Array<T>>
    where
        T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    {
        // Simplified matrix multiplication for 2D arrays
        if self.ndim() == 2 && other.ndim() == 2 {
            let (m, k1) = (self.shape[0], self.shape[1]);
            let (k2, n) = (other.shape[0], other.shape[1]);

            if k1 != k2 {
                return Err(NumPyError::shape_mismatch(vec![m, k1], vec![k2, n]));
            }

            let mut result = Array::<T>::zeros(vec![m, n]);

            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::default();
                    for k in 0..k1 {
                        let a = self.get(i * k1 + k).unwrap();
                        let b = other.get(k * n + j).unwrap();
                        sum = sum + a.clone() * b.clone();
                    }
                    result.set(i * n + j, sum)?;
                }
            }

            Ok(result)
        } else {
            Err(NumPyError::not_implemented(
                "dot product only supported for 2D arrays",
            ))
        }
    }
}

impl<T> fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array(shape={:?}, dtype={:?})", self.shape, self.dtype)
    }
}

impl<T> fmt::Display for Array<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Simplified display - real NumPy has sophisticated formatting
        if self.ndim() == 1 {
            write!(f, "[")?;
            for (i, _elem) in (0..self.size()).take(10).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                if let Some(val) = self.get(i) {
                    write!(f, "{}", val)?;
                }
            }
            if self.size() > 10 {
                write!(f, "...")?;
            }
            write!(f, "]")
        } else {
            write!(f, "Array({:?})", self.shape)
        }
    }
}
