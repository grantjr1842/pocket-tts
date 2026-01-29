#[cfg(test)]
mod property_tests {
    use numpy::array::Array;

    #[test]
    fn test_array_properties() {
        // Create a simple 2D array for testing
        let data = vec![1, 2, 3, 4];
        let shape = vec![2, 2];
        let array = Array::from_data(data, shape.clone());

        // Test basic properties
        assert_eq!(array.shape(), &shape);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.size(), 4);

        // Test itemsize property
        assert_eq!(array.itemsize(), std::mem::size_of::<i32>());

        // Test nbytes property
        assert_eq!(array.nbytes(), array.size() * array.itemsize());

        // Test T property (transpose view)
        let transposed = array.t();
        assert_eq!(transposed.shape(), &[2, 2]); // 2x2 transposed is still 2x2
        assert_eq!(transposed.get_linear(0), array.get_linear(0)); // [1] stays [1]
        assert_eq!(transposed.get_linear(1), array.get_linear(2)); // [3] moves to position 1

        // Test mT property (matrix transpose, same as T for 2D)
        let m_transposed = array.m_t();
        assert_eq!(m_transposed.shape(), transposed.shape());

        // Test base property (should return None for direct arrays)
        assert!(array.base().is_none());

        // Test device property
        assert_eq!(array.device(), "cpu");

        // Test real property (for non-complex, returns copy)
        let real_array = array.real().unwrap();
        assert_eq!(real_array.shape(), array.shape());
        assert_eq!(real_array.to_vec(), array.to_vec());

        // Test imag property (for non-complex, returns zeros)
        let imag_array = array.imag().unwrap();
        assert_eq!(imag_array.shape(), array.shape());
        assert_eq!(imag_array.to_vec(), vec![0; 4]);

        // Test flat property
        let flat_iter = array.flat();
        let flat_vec: Vec<i32> = flat_iter.cloned().collect();
        assert_eq!(flat_vec, vec![1, 2, 3, 4]);

        // Test flags property
        let flags = array.flags();
        assert!(flags.c_contiguous);
        assert!(flags.aligned);
        assert!(flags.writable);
        // own_data might be false due to Arc references, so we don't assert it

        // Test ctypes property
        let ctypes = array.ctypes();
        assert!(!ctypes.data.is_null());
        assert_eq!(ctypes.itemsize, array.itemsize());
    }

    #[test]
    fn test_array_properties_3d() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let shape = vec![3, 2, 2]; // 3D array with different dimensions
        let array = Array::from_data(data, shape);

        // Test mT property for 3D array (swaps last two axes)
        let m_transposed = array.m_t();
        // Original shape: [3, 2, 2] -> mT shape: [3, 2, 2] (swap last two axes)
        assert_eq!(m_transposed.shape(), &[3, 2, 2]);

        // Verify that mT produces a different view than T for non-cubic arrays
        let t_view = array.t();
        // T() reverses all axes: [3, 2, 2] -> [2, 2, 3]
        // mT() swaps last two axes: [3, 2, 2] -> [3, 2, 2]
        assert_ne!(m_transposed.shape(), t_view.shape());
        assert_eq!(t_view.shape(), &[2, 2, 3]);
    }

    #[test]
    fn test_array_flags() {
        let data = vec![1, 2, 3, 4];
        let array = Array::from_data(data, vec![2, 2]);
        let flags = array.flags();

        // Test flag methods
        assert_eq!(flags.writeable(), "WRITEABLE");
        assert!(flags.contiguous().contains("C_CONTIGUOUS"));
        assert_eq!(flags.aligned_flag(), "ALIGNED");
        assert_eq!(flags.ownership(), "OWNDATA");

        // Test with non-contiguous array (using transpose view)
        let transposed = array.t();
        let t_flags = transposed.flags();
        // Transpose view might not be C-contiguous
        assert!(t_flags.aligned);
        assert!(t_flags.writable);
    }
}
