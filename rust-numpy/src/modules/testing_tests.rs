/// Tests for numpy.testing module
///
/// This module tests all the assertion functions provided by the
/// testing module to ensure they work correctly and match NumPy's testing API.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    fn test_assert_array_equal() {
        let arr1 = Array::from_data(vec![1, 2, 3], vec![3]);
        let arr2 = Array::from_data(vec![1, 2, 3], vec![3]);
        
        assert!(assert_array_equal(&arr1, &arr2).is_ok());
        
        let arr3 = Array::from_data(vec![1, 2, 4], vec![3]);
        assert!(assert_array_equal(&arr1, &arr3).is_err());
        
        let arr4 = Array::from_data(vec![1, 2, 3, 4], vec![4]);
        assert!(assert_array_equal(&arr1, &arr4).is_err());
    }

    #[test]
    fn test_assert_equal() {
        assert!(assert_equal(5, 5).is_ok());
        assert!(assert_equal("hello", "hello").is_ok());
        assert!(assert_equal(5, 6).is_err());
        assert!(assert_equal("hello", "world").is_err());
    }

    #[test]
    fn test_assert_array_almost_equal() {
        let arr1 = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let arr2 = Array::from_data(vec![1.0001, 2.0001, 3.0001], vec![3]);
        
        assert!(assert_array_almost_equal(&arr1, &arr2, 3).is_ok());
        assert!(assert_array_almost_equal(&arr1, &arr2, 5).is_err());
    }

    #[test]
    fn test_assert_almost_equal() {
        assert!(assert_almost_equal(1.0, 1.0001, 3).is_ok());
        assert!(assert_almost_equal(1.0, 1.0001, 5).is_err());
    }

    #[test]
    fn test_assert_approx_equal() {
        assert!(assert_approx_equal(1.0, 1.001, 2).is_ok());
        assert!(assert_approx_equal(1.0, 1.001, 4).is_err());
    }

    #[test]
    fn test_assert_allclose() {
        let arr1 = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let arr2 = Array::from_data(vec![1.001, 2.001, 3.001], vec![3]);
        
        assert!(assert_allclose(&arr1, &arr2, 1e-2, 1e-3).is_ok());
        assert!(assert_allclose(&arr1, &arr2, 1e-4, 1e-5).is_err());
    }

    #[test]
    fn test_assert_array_almost_nulp() {
        let arr1 = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let arr2 = Array::from_data(vec![1.0000001, 2.0000001, 3.0000001], vec![3]);
        
        // Should pass with reasonable ULP tolerance
        assert!(assert_array_almost_nulp(&arr1, &arr2, 10).is_ok());
        // Should fail with very strict ULP tolerance
        assert!(assert_array_almost_nulp(&arr1, &arr2, 0).is_err());
    }

    #[test]
    fn test_assert_array_almost_equal_nulp() {
        let arr1 = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let arr2 = Array::from_data(vec![1.0000001, 2.0000001, 3.0000001], vec![3]);
        
        assert!(assert_array_almost_equal_nulp(&arr1, &arr2, 10).is_ok());
        assert!(assert_array_almost_equal_nulp(&arr1, &arr2, 0).is_err());
    }

    #[test]
    fn test_assert_array_max_ulp() {
        let arr1 = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let arr2 = Array::from_data(vec![1.0000001, 2.0000001, 3.0000001], vec![3]);
        
        assert!(assert_array_max_ulp(&arr1, &arr2, 10).is_ok());
        assert!(assert_array_max_ulp(&arr1, &arr2, 0).is_err());
    }

    #[test]
    fn test_assert_array_less() {
        let arr1 = Array::from_data(vec![1, 2, 3], vec![3]);
        let arr2 = Array::from_data(vec![2, 3, 4], vec![3]);
        
        assert!(assert_array_less(&arr1, &arr2).is_ok());
        
        let arr3 = Array::from_data(vec![1, 3, 3], vec![3]);
        assert!(assert_array_less(&arr1, &arr3).is_err());
    }

    #[test]
    fn test_assert_array_compare() {
        let arr1 = Array::from_data(vec![1, 2, 3], vec![3]);
        let arr2 = Array::from_data(vec![1, 2, 3], vec![3]);
        
        assert!(assert_array_compare(&arr1, &arr2).is_ok());
        
        let arr3 = Array::from_data(vec![1, 2, 4], vec![3]);
        assert!(assert_array_compare(&arr1, &arr3).is_err());
    }

    #[test]
    fn test_assert_array_shape_equal() {
        let arr1 = Array::from_data(vec![1, 2, 3], vec![3]);
        let arr2 = Array::from_data(vec![4, 5, 6], vec![3]);
        let arr3 = Array::from_data(vec![1, 2, 3, 4], vec![4]);
        
        assert!(assert_array_shape_equal(&arr1, &arr2).is_ok());
        assert!(assert_array_shape_equal(&arr1, &arr3).is_err());
    }

    #[test]
    fn test_assert_string_equal() {
        assert!(assert_string_equal("hello", "hello").is_ok());
        assert!(assert_string_equal("hello", "world").is_err());
    }

    #[test]
    fn test_assert_raises() {
        // Test function that panics
        let panic_func = || {
            panic!("This should panic");
        };
        
        assert!(assert_raises(panic_func, "panic").is_ok());
        
        // Test function that doesn't panic
        let normal_func = || {
            42
        };
        
        assert!(assert_raises(normal_func, "panic").is_err());
    }

    #[test]
    fn test_assert_raises_regex() {
        let panic_func = || {
            panic!("This should panic");
        };
        
        assert!(assert_raises_regex(panic_func, "panic", ".*panic.*").is_ok());
        
        let normal_func = || {
            42
        };
        
        assert!(assert_raises_regex(normal_func, "panic", ".*panic.*").is_err());
    }

    #[test]
    fn test_assert_warns() {
        let test_func = || {
            42
        };
        
        // Should return the result
        assert_eq!(assert_warns(test_func).unwrap(), 42);
    }

    #[test]
    fn test_assert_no_warnings() {
        let test_func = || {
            42
        };
        
        // Should return the result
        assert_eq!(assert_no_warnings(test_func).unwrap(), 42);
    }

    #[test]
    fn test_assert_no_gc_cycles() {
        // This is a no-op in Rust, should always succeed
        assert!(assert_no_gc_cycles().is_ok());
    }

    #[test]
    fn test_check_allclose() {
        let arr1 = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let arr2 = Array::from_data(vec![1.001, 2.001, 3.001], vec![3]);
        
        assert!(check_allclose(&arr1, &arr2, 1e-2, 1e-3));
        assert!(!check_allclose(&arr1, &arr2, 1e-4, 1e-5));
    }

    #[test]
    fn test_all_exports_available() {
        // Test that all functions are available through exports
        use super::exports::*;
        
        // This is just a compilation test - if it compiles, all exports are available
        let _ = assert_allclose as fn(_, _, _, _) -> _;
        let _ = assert_almost_equal as fn(_, _, _) -> _;
        let _ = assert_approx_equal as fn(_, _, _) -> _;
        let _ = assert_array_almost_equal as fn(_, _, _) -> _;
        let _ = assert_array_almost_equal_nulp as fn(_, _, _) -> _;
        let _ = assert_array_almost_nulp as fn(_, _, _) -> _;
        let _ = assert_array_compare as fn(_, _) -> _;
        let _ = assert_array_equal as fn(_, _) -> _;
        let _ = assert_array_less as fn(_, _) -> _;
        let _ = assert_array_max_ulp as fn(_, _, _) -> _;
        let _ = assert_array_shape_equal as fn(_, _) -> _;
        let _ = assert_equal as fn(_, _) -> _;
        let _ = assert_no_gc_cycles as fn() -> _;
        let _ = assert_no_warnings as fn(_) -> _;
        let _ = assert_raises as fn(_, _) -> _;
        let _ = assert_raises_regex as fn(_, _, _) -> _;
        let _ = assert_string_equal as fn(_, _) -> _;
        let _ = assert_warns as fn(_) -> _;
    }

    #[test]
    fn test_comprehensive_testing_workflow() {
        // Test a comprehensive workflow using multiple assertion functions
        
        // Create test arrays
        let actual = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let desired = Array::from_data(vec![1.001, 2.001, 3.001], vec![3]);
        
        // Test shape equality first
        assert!(assert_array_shape_equal(&actual, &desired).is_ok());
        
        // Test that actual is less than desired (for this specific case)
        assert!(assert_array_less(&actual, &desired).is_ok());
        
        // Test approximate equality
        assert!(assert_allclose(&actual, &desired, 1e-2, 1e-3).is_ok());
        
        // Test ULP-based comparison
        assert!(assert_array_almost_nulp(&actual, &desired, 1000).is_ok());
        
        // Test string assertions
        assert!(assert_string_equal("test", "test").is_ok());
        
        // Test value equality
        assert!(assert_equal(42, 42).is_ok());
        
        // Test floating point approximations
        assert!(assert_almost_equal(3.14159, 3.14160, 4).is_ok());
        assert!(assert_approx_equal(2.71828, 2.71830, 4).is_ok());
    }
}
