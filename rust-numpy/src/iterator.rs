use crate::array::Array;

/// Iterator over array elements
pub struct ArrayIter<'a, T> {
    array: &'a Array<T>,
    indices: Vec<usize>,
    current_offset: isize,
    remaining: usize,
}

impl<'a, T> ArrayIter<'a, T> {
    pub fn new(array: &'a Array<T>) -> Self {
        let size = array.size();
        let ndim = array.shape().len();
        Self {
            array,
            indices: vec![0; ndim],
            current_offset: 0,
            remaining: size,
        }
    }
}

impl<'a, T> Iterator for ArrayIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // Get element at current offset
        // We trust our offset calculation is correct and within bounds of the storage
        // relative to array.offset
        let item = self.array.get_storage_at(self.current_offset);

        // Advance counters
        self.remaining -= 1;
        if self.remaining > 0 {
            let shape = self.array.shape();
            let strides = self.array.strides();
            let ndim = shape.len();

            for i in (0..ndim).rev() {
                self.indices[i] += 1;
                self.current_offset += strides[i];

                if self.indices[i] < shape[i] {
                    // No carry, we are done
                    break;
                }

                // Carry over: reset this dimension and continue loop to next dimension
                self.indices[i] = 0;
                // Backtrack offset: we added strides[i] * shape[i] total.
                // We want to return to 0-index offset for this dimension.
                // So we remove stride * shape.
                self.current_offset -= strides[i] * (shape[i] as isize);
            }
        }

        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T> ExactSizeIterator for ArrayIter<'a, T> {}

/// Broadcast-aware N-D iterator for binary operations
///
/// This iterator yields tuples of (offset_a, offset_b, offset_out) for
/// each element when operating on two arrays that may have different shapes
/// but have been broadcasted to a common shape.
///
/// Example:
/// - Array a with shape (3, 1) broadcasted to (3, 4)
/// - Array b with shape (1, 4) broadcasted to (3, 4)
/// - Output array with shape (3, 4)
///
/// The iterator correctly handles stride=0 for broadcast dimensions.
pub struct BroadcastIter<'a, 'b, 'c, T> {
    array_a: &'a Array<T>,
    array_b: &'b Array<T>,
    array_out: &'c Array<T>,

    // Current indices in the broadcasted shape
    indices_a: Vec<usize>,
    indices_b: Vec<usize>,

    // Current offsets in each array's storage
    offset_a: isize,
    offset_b: isize,
    offset_out: isize,

    // Remaining elements
    remaining: usize,
}

impl<'a, 'b, 'c, T> BroadcastIter<'a, 'b, 'c, T> {
    /// Create new broadcast iterator
    ///
    /// All three arrays must have the same shape (after broadcasting).
    pub fn new(array_a: &'a Array<T>, array_b: &'b Array<T>, array_out: &'c Array<T>) -> Self {
        let ndim = array_a.shape().len();
        assert_eq!(
            array_b.shape(),
            array_a.shape(),
            "All arrays must have same broadcasted shape"
        );
        assert_eq!(
            array_out.shape(),
            array_a.shape(),
            "Output must match broadcasted shape"
        );

        let size = array_a.size();

        Self {
            array_a,
            array_b,
            array_out,
            indices_a: vec![0; ndim],
            indices_b: vec![0; ndim],
            offset_a: 0,
            offset_b: 0,
            offset_out: 0,
            remaining: size,
        }
    }

    /// Get current element from array_a
    #[inline]
    pub fn current_a(&self) -> Option<&'a T> {
        self.array_a.get_storage_at(self.offset_a)
    }

    /// Get current element from array_b
    #[inline]
    pub fn current_b(&self) -> Option<&'b T> {
        self.array_b.get_storage_at(self.offset_b)
    }

    /// Get current output offset
    #[inline]
    pub fn current_out_offset(&self) -> isize {
        self.offset_out
    }
}

impl<'a, 'b, 'c, T> Iterator for BroadcastIter<'a, 'b, 'c, T> {
    type Item = (&'a T, &'b T, isize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // Get current elements
        let elem_a = self.array_a.get_storage_at(self.offset_a);
        let elem_b = self.array_b.get_storage_at(self.offset_b);
        let result = (elem_a?, elem_b?, self.offset_out);

        // Advance all arrays
        self.remaining -= 1;

        if self.remaining > 0 {
            let shape = self.array_a.shape();
            let strides_a = self.array_a.strides();
            let strides_b = self.array_b.strides();
            let strides_out = self.array_out.strides();
            let ndim = shape.len();

            for i in (0..ndim).rev() {
                // Advance indices
                self.indices_a[i] += 1;
                self.indices_b[i] += 1;

                self.offset_a += strides_a[i];
                self.offset_b += strides_b[i];
                self.offset_out += strides_out[i];

                if self.indices_a[i] < shape[i] {
                    // No carry, we're done
                    break;
                }

                // Carry over: reset this dimension and continue to next dimension
                self.indices_a[i] = 0;
                self.indices_b[i] = 0;

                // Backtrack offsets
                self.offset_a -= strides_a[i] * (shape[i] as isize);
                self.offset_b -= strides_b[i] * (shape[i] as isize);
                self.offset_out -= strides_out[i] * (shape[i] as isize);
            }
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, 'b, 'c, T> ExactSizeIterator for BroadcastIter<'a, 'b, 'c, T> {}
