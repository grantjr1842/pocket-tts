//! Additional tests for the new ndarray methods

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;

    #[test]
    fn test_max_min() {
        let arr = Array::from_vec(vec![1, 5, 3, 9, 2]);
        assert_eq!(arr.max(), Some(&9));
        assert_eq!(arr.min(), Some(&1));
        
        let empty_arr: Array<i32> = Array::from_vec(vec![]);
        assert_eq!(empty_arr.max(), None);
        assert_eq!(empty_arr.min(), None);
    }

    #[test]
    fn test_sum_prod() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(arr.sum(), 10);
        assert_eq!(arr.prod(), 24);
        
        let arr_f = Array::from_vec(vec![1.5, 2.5, 3.0]);
        assert!((arr_f.sum() - 7.0).abs() < 1e-10);
        assert!((arr_f.prod() - 11.25).abs() < 1e-10);
    }

    #[test]
    fn test_mean_std_var() {
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let mean = arr.mean();
        assert!((mean - 3.0).abs() < 1e-10);
        
        let std = arr.std();
        let expected_std = ((4.0 + 1.0 + 0.0 + 1.0 + 4.0) / 4.0).sqrt();
        assert!((std - expected_std).abs() < 1e-10);
        
        let var = arr.var();
        assert!((var - expected_std.powi(2)).abs() < 1e-10);
        
        // Test with single element
        let single = Array::from_vec(vec![42]);
        assert!(single.mean().is_nan());
        assert!(single.std().is_nan());
        assert!(single.var().is_nan());
    }

    #[test]
    fn test_all_any() {
        let arr_true = Array::from_vec(vec![true, true, true]);
        assert!(arr_true.all());
        assert!(arr_true.any());
        
        let arr_mixed = Array::from_vec(vec![true, false, true]);
        assert!(!arr_mixed.all());
        assert!(arr_mixed.any());
        
        let arr_false = Array::from_vec(vec![false, false]);
        assert!(!arr_false.all());
        assert!(!arr_false.any());
        
        // Test with numeric types (0 = false, non-zero = true)
        let arr_nums = Array::from_vec(vec![1, 2, 0]);
        assert!(!arr_nums.all());
        assert!(arr_nums.any());
    }

    #[test]
    fn test_ptp() {
        let arr = Array::from_vec(vec![1, 5, 3, 9, 2]);
        assert_eq!(arr.ptp(), Some(8.0)); // 9 - 1
        
        let empty_arr: Array<i32> = Array::from_vec(vec![]);
        assert_eq!(empty_arr.ptp(), None);
    }

    #[test]
    fn test_round() {
        let arr = Array::from_vec(vec![1.234, 2.567, 3.891]);
        let rounded = arr.round(2).unwrap();
        
        assert_eq!(rounded.get(0), Some(&1.23));
        assert_eq!(rounded.get(1), Some(&2.57));
        assert_eq!(rounded.get(2), Some(&3.89));
        
        let rounded_zero = arr.round(0).unwrap();
        assert_eq!(rounded_zero.get(0), Some(&1.0));
        assert_eq!(rounded_zero.get(1), Some(&3.0));
        assert_eq!(rounded_zero.get(2), Some(&4.0));
    }

    #[test]
    fn test_flatten_ravel() {
        let arr = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        let flattened = arr.flatten();
        
        assert_eq!(flattened.shape(), &[6]);
        assert_eq!(flattened.get(0), Some(&1));
        assert_eq!(flattened.get(5), Some(&6));
        
        let raveled = arr.ravel();
        assert_eq!(raveled.shape(), &[6]);
        assert_eq!(raveled.get(0), Some(&1));
        assert_eq!(raveled.get(5), Some(&6));
    }

    #[test]
    fn test_squeeze() {
        // Test removing single dimensions
        let arr = Array::from_shape_vec(vec![1, 3, 1, 4], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
        let squeezed = arr.squeeze();
        
        assert_eq!(squeezed.shape(), &[3, 4]);
        assert_eq!(squeezed.size(), 12);
        
        // Test array with all single dimensions
        let single_dim = Array::from_shape_vec(vec![1, 1, 1], vec![42]).unwrap();
        let squeezed_single = single_dim.squeeze();
        assert_eq!(squeezed_single.shape(), &[1]);
    }

    #[test]
    fn test_argsort_sort() {
        let mut arr = Array::from_vec(vec![3, 1, 4, 1, 5, 9]);
        
        let argsorted = arr.argsort().unwrap();
        let expected_indices = vec![1, 3, 0, 2, 4, 5]; // 1,1,3,4,5,9
        assert_eq!(argsorted.shape(), &[6]);
        for (i, &expected_idx) in expected_indices.iter().enumerate() {
            assert_eq!(argsorted.get(i), Some(&expected_idx));
        }
        
        // Test sorting
        arr.sort().unwrap();
        let expected_sorted = vec![1, 1, 3, 4, 5, 9];
        for (i, &expected_val) in expected_sorted.iter().enumerate() {
            assert_eq!(arr.get(i), Some(&expected_val));
        }
    }

    #[test]
    fn test_tolist() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]);
        let list = arr.tolist();
        
        assert_eq!(list, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_argmax_argmin() {
        let arr = Array::from_vec(vec![3, 1, 4, 1, 5, 9, 2]);
        
        assert_eq!(arr.argmax(), Some(5)); // 9 at index 5
        assert_eq!(arr.argmin(), Some(1)); // 1 at index 1 (first occurrence)
        
        let empty_arr: Array<i32> = Array::from_vec(vec![]);
        assert_eq!(empty_arr.argmax(), None);
        assert_eq!(empty_arr.argmin(), None);
    }

    #[test]
    fn test_trace() {
        // Test 2D array
        let arr_2d = Array::from_shape_vec(vec![3, 3], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let trace = arr_2d.trace();
        assert_eq!(trace, 15); // 1 + 5 + 9
        
        // Test 1D array (should return default)
        let arr_1d = Array::from_vec(vec![1, 2, 3]);
        let trace_1d = arr_1d.trace();
        assert_eq!(trace_1d, 0);
    }

    #[test]
    fn test_cumsum_cumprod() {
        let arr = Array::from_vec(vec![1, 2, 3, 4]);
        
        let cumsum = arr.cumsum();
        let expected_cumsum = vec![1, 3, 6, 10];
        for (i, &expected) in expected_cumsum.iter().enumerate() {
            assert_eq!(cumsum.get(i), Some(&expected));
        }
        
        let cumprod = arr.cumprod();
        let expected_cumprod = vec![1, 2, 6, 24];
        for (i, &expected) in expected_cumprod.iter().enumerate() {
            assert_eq!(cumprod.get(i), Some(&expected));
        }
    }

    #[test]
    fn test_multidimensional_operations() {
        // Test operations on 2D arrays
        let arr_2d = Array::from_shape_vec(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        
        assert_eq!(arr_2d.max(), Some(&6));
        assert_eq!(arr_2d.min(), Some(&1));
        assert_eq!(arr_2d.sum(), 21);
        assert_eq!(arr_2d.prod(), 720);
        
        let mean = arr_2d.mean();
        assert!((mean - 3.5).abs() < 1e-10);
        
        // Test all/any on 2D array
        let bool_2d = Array::from_shape_vec(vec![2, 2], vec![true, false, true, true]).unwrap();
        assert!(!bool_2d.all());
        assert!(bool_2d.any());
    }

    #[test]
    fn test_edge_cases() {
        // Test with negative numbers
        let arr_neg = Array::from_vec(vec![-1, -5, -3, -2]);
        assert_eq!(arr_neg.max(), Some(&-1));
        assert_eq!(arr_neg.min(), Some(&-5));
        assert_eq!(arr_neg.ptp(), Some(4.0)); // -1 - (-5) = 4
        
        // Test with floating point edge cases
        let arr_float = Array::from_vec(vec![0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY]);
        assert_eq!(arr_float.max(), Some(&f64::INFINITY));
        assert_eq!(arr_float.min(), Some(&f64::NEG_INFINITY));
        
        // Test with mixed types
        let arr_mixed = Array::from_vec(vec![0, 1, 2, 3]);
        assert!(!arr_mixed.all()); // 0 evaluates to false
        assert!(arr_mixed.any());  // non-zero values evaluate to true
    }

    #[test]
    fn test_type_compatibility() {
        // Test with different numeric types
        let arr_i32 = Array::from_vec(vec![1i32, 2, 3]);
        assert_eq!(arr_i32.sum(), 6i32);
        assert_eq!(arr_i32.prod(), 6i32);
        
        let arr_f32 = Array::from_vec(vec![1.0f32, 2.0, 3.0]);
        let mean_f32 = arr_f32.mean();
        assert!((mean_f32 - 2.0).abs() < 1e-6);
        
        let arr_u64 = Array::from_vec(vec![1u64, 2, 3, 4]);
        assert_eq!(arr_u64.sum(), 10u64);
        assert_eq!(arr_u64.prod(), 24u64);
    }

    #[test]
    fn test_statistical_properties() {
        // Test statistical properties with known values
        let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        
        let mean = arr.mean();
        assert!((mean - 3.0).abs() < 1e-10);
        
        let variance = arr.var();
        let expected_var = 2.5; // Sample variance of [1,2,3,4,5]
        assert!((variance - expected_var).abs() < 1e-10);
        
        let std = arr.std();
        assert!((std - expected_var.sqrt()).abs() < 1e-10);
        
        // Test with identical values (zero variance)
        let arr_identical = Array::from_vec(vec![3.0, 3.0, 3.0]);
        assert_eq!(arr_identical.mean(), 3.0);
        assert!((arr_identical.var() - 0.0).abs() < 1e-10);
        assert!((arr_identical.std() - 0.0).abs() < 1e-10);
    }
}
