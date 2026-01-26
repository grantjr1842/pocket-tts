//! Additional tests for the new MaskedArray methods

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;

    fn create_test_masked_array() -> MaskedArray<i32> {
        let data = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let mask = Array::from_vec(vec![false, true, false, false, true]);
        MaskedArray::new(data, mask).unwrap()
    }

    #[test]
    fn test_all_any() {
        let ma = create_test_masked_array();
        
        // Test all: should consider only unmasked elements [1, 3, 4]
        assert!(ma.all()); // All non-zero values evaluate to true
        
        let data_false = Array::from_vec(vec![0, 1, 0]);
        let mask_false = Array::from_vec(vec![false, true, false]);
        let ma_false = MaskedArray::new(data_false, mask_false).unwrap();
        assert!(!ma_false.all()); // 0 evaluates to false
        assert!(ma_false.any()); // 1 evaluates to true
        
        let data_none = Array::from_vec(vec![0, 0, 0]);
        let mask_none = Array::from_vec(vec![false, true, false]);
        let ma_none = MaskedArray::new(data_none, mask_none).unwrap();
        assert!(!ma_none.all());
        assert!(!ma_none.any());
    }

    #[test]
    fn test_max_min() {
        let ma = create_test_masked_array();
        
        // Unmasked elements are [1, 3, 4]
        assert_eq!(ma.max(), Some(&4));
        assert_eq!(ma.min(), Some(&1));
        
        // Test with all masked
        let data = Array::from_vec(vec![1, 2, 3]);
        let mask_all = Array::from_vec(vec![true, true, true]);
        let ma_all_masked = MaskedArray::new(data, mask_all).unwrap();
        assert_eq!(ma_all_masked.max(), None);
        assert_eq!(ma_all_masked.min(), None);
    }

    #[test]
    fn test_prod_product() {
        let ma = create_test_masked_array();
        
        // Unmasked elements are [1, 3, 4], product = 12
        assert_eq!(ma.prod(), 12);
        assert_eq!(ma.product(), 12);
        
        // Test with all masked
        let data = Array::from_vec(vec![1, 2, 3]);
        let mask_all = Array::from_vec(vec![true, true, true]);
        let ma_all_masked = MaskedArray::new(data, mask_all).unwrap();
        assert_eq!(ma_all_masked.prod(), 0); // Default value
    }

    #[test]
    fn test_ptp() {
        let ma = create_test_masked_array();
        
        // Unmasked elements are [1, 3, 4], ptp = 4 - 1 = 3
        assert_eq!(ma.ptp(), Some(3.0));
        
        // Test with all masked
        let data = Array::from_vec(vec![1, 2, 3]);
        let mask_all = Array::from_vec(vec![true, true, true]);
        let ma_all_masked = MaskedArray::new(data, mask_all).unwrap();
        assert_eq!(ma_all_masked.ptp(), None);
    }

    #[test]
    fn test_copy() {
        let ma = create_test_masked_array();
        let ma_copy = ma.copy();
        
        // Should be equal but independent
        assert_eq!(ma.data().data(), ma_copy.data().data());
        assert_eq!(ma.mask().data(), ma_copy.mask().data());
        assert_eq!(ma.fill_value(), ma_copy.fill_value());
    }

    #[test]
    fn test_flatten_ravel() {
        let data = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mask = Array::from_shape_vec(vec![2, 3], vec![false, true, false, false, true, false]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        
        let flattened = ma.flatten();
        assert_eq!(flattened.shape(), &[6]);
        assert_eq!(flattened.count(), 3); // Same number of unmasked elements
        
        let raveled = ma.ravel();
        assert_eq!(raveled.shape(), &[6]);
        assert_eq!(raveled.count(), 3);
    }

    #[test]
    fn test_reshape() {
        let ma = create_test_masked_array();
        
        let reshaped = ma.reshape(vec![5, 1]).unwrap();
        assert_eq!(reshaped.shape(), &[5, 1]);
        assert_eq!(reshaped.count(), 3); // Same number of unmasked elements
        
        // Test invalid reshape
        let invalid_reshape = ma.reshape(vec![2, 3]);
        assert!(invalid_reshape.is_err());
    }

    #[test]
    fn test_squeeze() {
        let data = Array::from_shape_vec(vec![1, 5, 1], vec![1, 2, 3, 4, 5]).unwrap();
        let mask = Array::from_shape_vec(vec![1, 5, 1], vec![false, true, false, false, true]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        
        let squeezed = ma.squeeze();
        assert_eq!(squeezed.shape(), &[5]);
        assert_eq!(squeezed.count(), 3);
    }

    #[test]
    fn test_transpose() {
        let data = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mask = Array::from_shape_vec(vec![2, 3], vec![false, true, false, false, true, false]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        
        let transposed = ma.transpose();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.count(), 3); // Same number of unmasked elements
    }

    #[test]
    fn test_take() {
        let ma = create_test_masked_array();
        
        let taken = ma.take(&[0, 2, 4], None).unwrap();
        assert_eq!(taken.shape(), &[3]);
        assert_eq!(taken.count(), 2); // Elements at indices 0 and 2 are unmasked
    }

    #[test]
    fn test_item() {
        let ma = create_test_masked_array();
        assert_eq!(ma.item(), None); // Not a 0D array
        
        // Create a 0D masked array
        let data_0d = Array::from_shape_vec(vec![], vec![42]).unwrap();
        let mask_0d = Array::from_shape_vec(vec![], vec![false]).unwrap();
        let ma_0d = MaskedArray::new(data_0d, mask_0d).unwrap();
        assert_eq!(ma_0d.item(), Some(&42));
        
        // 0D with masked element
        let mask_0d_masked = Array::from_shape_vec(vec![], vec![true]).unwrap();
        let ma_0d_masked = MaskedArray::new(data_0d, mask_0d_masked).unwrap();
        assert_eq!(ma_0d_masked.item(), Some(&42)); // Still returns the data value
    }

    #[test]
    fn test_itemset() {
        let mut ma = create_test_masked_array();
        
        // Not a 0D array
        assert!(ma.itemset(99).is_err());
        
        // 0D array (simplified implementation returns not_implemented)
        let data_0d = Array::from_shape_vec(vec![], vec![42]).unwrap();
        let mask_0d = Array::from_shape_vec(vec![], vec![false]).unwrap();
        let mut ma_0d = MaskedArray::new(data_0d, mask_0d).unwrap();
        assert!(ma_0d.itemset(99).is_err()); // Not implemented yet
    }

    #[test]
    fn test_mask_aware_operations() {
        // Test that operations properly respect masks
        let data = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let mask = Array::from_vec(vec![true, false, true, false, true]);
        let ma = MaskedArray::new(data, mask).unwrap();
        
        // Only unmasked elements are [2, 4]
        assert_eq!(ma.max(), Some(&4));
        assert_eq!(ma.min(), Some(&2));
        assert_eq!(ma.prod(), 8); // 2 * 4
        assert_eq!(ma.ptp(), Some(2.0)); // 4 - 2
        assert!(ma.all()); // Both 2 and 4 evaluate to true
        assert!(ma.any()); // Both evaluate to true
    }

    #[test]
    fn test_multidimensional_masked_arrays() {
        let data = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mask = Array::from_shape_vec(vec![2, 3], vec![false, true, false, false, true, false]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        
        // Unmasked elements are [1, 3, 4, 6]
        assert_eq!(ma.max(), Some(&6));
        assert_eq!(ma.min(), Some(&1));
        assert_eq!(ma.prod(), 72); // 1 * 3 * 4 * 6
        assert_eq!(ma.count(), 4);
        
        // Test operations that preserve shape
        let transposed = ma.transpose();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.count(), 4);
        
        let flattened = ma.flatten();
        assert_eq!(flattened.shape(), &[6]);
        assert_eq!(flattened.count(), 4);
    }

    #[test]
    fn test_edge_cases() {
        // Empty masked array
        let data_empty = Array::<i32>::from_vec(vec![]);
        let mask_empty = Array::from_vec(vec![]);
        let ma_empty = MaskedArray::new(data_empty, mask_empty).unwrap();
        
        assert!(ma_empty.all()); // Vacuously true
        assert!(!ma_empty.any()); // Vacuously false
        assert_eq!(ma_empty.max(), None);
        assert_eq!(ma_empty.min(), None);
        assert_eq!(ma_empty.ptp(), None);
        assert_eq!(ma_empty.count(), 0);
        
        // All elements masked
        let data = Array::from_vec(vec![1, 2, 3]);
        let mask_all = Array::from_vec(vec![true, true, true]);
        let ma_all_masked = MaskedArray::new(data, mask_all).unwrap();
        
        assert!(ma_all_masked.all()); // Vacuously true
        assert!(!ma_all_masked.any()); // Vacuously false
        assert_eq!(ma_all_masked.max(), None);
        assert_eq!(ma_all_masked.min(), None);
        assert_eq!(ma_all_masked.ptp(), None);
        assert_eq!(ma_all_masked.count(), 0);
    }

    #[test]
    fn test_type_compatibility() {
        // Test with different numeric types
        let data_f32 = Array::from_vec(vec![1.0f32, 2.0, 3.0]);
        let mask_f32 = Array::from_vec(vec![false, true, false]);
        let ma_f32 = MaskedArray::new(data_f32, mask_f32).unwrap();
        
        assert_eq!(ma_f32.max(), Some(&3.0f32));
        assert_eq!(ma_f32.min(), Some(&1.0f32));
        assert_eq!(ma_f32.prod(), 3.0f32); // 1.0 * 3.0
        
        // Test with boolean type
        let data_bool = Array::from_vec(vec![true, false, true]);
        let mask_bool = Array::from_vec(vec![false, true, false]);
        let ma_bool = MaskedArray::new(data_bool, mask_bool).unwrap();
        
        assert!(ma_bool.all()); // Only unmasked element is true
        assert!(ma_bool.any()); // Unmasked element is true
    }

    #[test]
    fn test_fill_value_preservation() {
        let mut ma = create_test_masked_array();
        ma.set_fill_value(Some(999));
        
        let copied = ma.copy();
        assert_eq!(copied.fill_value(), &Some(999));
        
        let flattened = ma.flatten();
        assert_eq!(flattened.fill_value(), &Some(999));
        
        let reshaped = ma.reshape(vec![5, 1]).unwrap();
        assert_eq!(reshaped.fill_value(), &Some(999));
    }
}
