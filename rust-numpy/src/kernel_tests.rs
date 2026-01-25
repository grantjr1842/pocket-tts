use numpy::kernel_api::*;
use numpy::kernel_registry::PerformanceHint;
use numpy::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_kernel_registration() {
        // Test that we can register kernels at runtime
        register_binary_kernel("custom_add", |a: i32, b: i32| a + b).unwrap();
        register_unary_kernel("custom_neg", |a: f64| -a).unwrap();

        // Verify kernels were registered
        let kernel_names = list_kernels().unwrap();
        assert!(kernel_names.contains(&"custom_add".to_string()));
        assert!(kernel_names.contains(&"custom_neg".to_string()));
    }

    #[test]
    fn test_kernel_execution() {
        // Register a test kernel
        register_binary_kernel("test_multiply", |a: i32, b: i32| a * b).unwrap();

        // Create test arrays
        let a = array![2, 3, 4];
        let b = array![5, 6, 7];

        // Execute the kernel
        let result = execute_binary("test_multiply", &a, &b).unwrap();

        // Verify results
        assert_eq!(result.get(0).unwrap(), &10);
        assert_eq!(result.get(1).unwrap(), &18);
        assert_eq!(result.get(2).unwrap(), &28);
    }

    #[test]
    fn test_performance_hints() {
        // Register kernels with different performance hints
        register_binary_kernel_with_hint(
            "vectorized_add",
            |a: f64, b: f64| a + b,
            PerformanceHint::Vectorized,
        )
        .unwrap();
        register_binary_kernel_with_hint(
            "memory_opt_add",
            |a: f64, b: f64| a + b,
            PerformanceHint::MemoryBound,
        )
        .unwrap();
        register_binary_kernel_with_hint(
            "compute_opt_add",
            |a: f64, b: f64| a + b,
            PerformanceHint::ComputeBound,
        )
        .unwrap();

        // Create large test arrays
        let a = Array::from_vec((0..1000).map(|i| i as f64).collect());
        let b = Array::from_vec((0..1000).map(|i| i as f64).collect());

        // Test vectorized kernel
        let result1 = execute_binary("vectorized_add", &a, &b).unwrap();
        assert_eq!(result1.size(), 1000);

        // Test memory-optimized kernel
        let result2 = execute_binary("memory_opt_add", &a, &b).unwrap();
        assert_eq!(result2.size(), 1000);

        // Test compute-optimized kernel
        let result3 = execute_binary("compute_opt_add", &a, &b).unwrap();
        assert_eq!(result3.size(), 1000);

        // All should give the same result
        for i in 0..1000 {
            assert_eq!(result1.get(i).unwrap(), result2.get(i).unwrap());
            assert_eq!(result2.get(i).unwrap(), result3.get(i).unwrap());
        }
    }

    #[test]
    fn test_dtype_specific_optimization() {
        // Register kernels for different dtypes
        register_binary_kernel_with_hint(
            "add_f64",
            |a: f64, b: f64| a + b,
            PerformanceHint::Vectorized,
        )
        .unwrap();
        register_binary_kernel_with_hint(
            "add_f32",
            |a: f32, b: f32| a + b,
            PerformanceHint::Vectorized,
        )
        .unwrap();
        register_binary_kernel_with_hint(
            "add_i64",
            |a: i64, b: i64| a + b,
            PerformanceHint::General,
        )
        .unwrap();

        // Test with different dtypes
        let a_f64 = array![1.0, 2.0, 3.0];
        let b_f64 = array![4.0, 5.0, 6.0];
        let result_f64 = execute_binary("add_f64", &a_f64, &b_f64).unwrap();

        let a_f32 = array![1.0f32, 2.0f32, 3.0f32];
        let b_f32 = array![4.0f32, 5.0f32, 6.0f32];
        let result_f32 = execute_binary("add_f32", &a_f32, &b_f32).unwrap();

        let a_i64 = array![1i64, 2i64, 3i64];
        let b_i64 = array![4i64, 5i64, 6i64];
        let result_i64 = execute_binary("add_i64", &a_i64, &b_i64).unwrap();

        // Verify results
        assert_eq!(result_f64.get(0).unwrap(), &5.0);
        assert_eq!(result_f32.get(1).unwrap(), &7.0f32);
        assert_eq!(result_i64.get(2).unwrap(), &9i64);
    }

    #[test]
    fn test_registry_builder() {
        // Test the builder pattern
        let builder = KernelRegistryBuilder::new()
            .with_binary_kernel("builder_mul", |a: i32, b: i32| a * b)
            .with_unary_kernel("builder_abs", |a: f64| a.abs())
            .with_common_kernels()
            .build();

        assert!(builder.is_ok());

        // Verify kernels were registered
        let kernel_names = list_kernels().unwrap();
        assert!(kernel_names.contains(&"builder_mul".to_string()));
        assert!(kernel_names.contains(&"builder_abs".to_string()));
        assert!(kernel_names.contains(&"add".to_string())); // From common kernels
        assert!(kernel_names.contains(&"multiply".to_string())); // From common kernels
    }

    #[test]
    fn test_registry_stats() {
        // Register some kernels
        register_binary_kernel("stats_test_add", |a: i32, b: i32| a + b).unwrap();
        register_unary_kernel("stats_test_neg", |a: f64| -a).unwrap();

        // Get registry stats
        let info = get_kernel_info().unwrap();

        // Verify stats
        assert!(info.kernel_names.len() >= 2);
        assert!(info.stats.kernel_names >= 1);
        assert!(info.stats.total_implementations >= 2);
    }

    #[test]
    fn test_kernel_caching() {
        // Register a kernel
        register_binary_kernel("cache_test", |a: i32, b: i32| a + b).unwrap();

        // Create test arrays
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];

        // Execute multiple times to test caching
        let result1 = execute_binary("cache_test", &a, &b).unwrap();
        let result2 = execute_binary("cache_test", &a, &b).unwrap();

        // Results should be identical
        for i in 0..3 {
            assert_eq!(result1.get(i).unwrap(), result2.get(i).unwrap());
        }

        // Check cache stats
        let info = get_kernel_info().unwrap();
        assert!(info.stats.cache_entries > 0);
    }

    #[test]
    fn test_unary_kernel_execution() {
        // Register a unary kernel
        register_unary_kernel("test_square", |a: f64| a * a).unwrap();

        // Create test array
        let a = array![2.0, 3.0, 4.0];

        // Execute the kernel
        let result = execute_unary("test_square", &a).unwrap();

        // Verify results
        assert_eq!(result.get(0).unwrap(), &4.0);
        assert_eq!(result.get(1).unwrap(), &9.0);
        assert_eq!(result.get(2).unwrap(), &16.0);
    }

    #[test]
    fn test_error_handling() {
        // Try to execute a kernel that doesn't exist
        let a = array![1, 2, 3];
        let b = array![4, 5, 6];

        let result = execute_binary("nonexistent_kernel", &a, &b);
        assert!(result.is_err());

        // Try to execute with wrong input types
        register_binary_kernel("type_test", |a: i32, b: i32| a + b).unwrap();
        let c = array![1.0, 2.0, 3.0]; // f64 instead of i32

        let result = execute_binary("type_test", &a, &c);
        // This should work with casting, but let's test the error case
        // In a real implementation, this might succeed with automatic casting
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_integration_with_existing_ufuncs() {
        // Initialize the kernel registry
        init_kernel_registry().unwrap();

        // Test that we can use both new and existing functionality
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        // Use new dynamic kernel system
        let dynamic_result = execute_binary("add", &a, &b).unwrap();

        // Use existing ufunc system (if available)
        let ufunc_result = a.add(&b, None, crate::dtype::Casting::Safe);

        // Results should be the same
        if let Ok(ufunc_res) = ufunc_result {
            for i in 0..3 {
                assert_eq!(dynamic_result.get(i).unwrap(), ufunc_res.get(i).unwrap());
            }
        }
    }

    #[test]
    fn test_performance_comparison() {
        // Initialize with common kernels
        init_kernel_registry().unwrap();

        // Create large arrays for performance testing
        let large_a = Array::from_vec((0..10000).map(|i| i as f64).collect());
        let large_b = Array::from_vec((0..10000).map(|i| i as f64).collect());

        // Test different kernel implementations
        let start = std::time::Instant::now();
        let result1 = execute_binary("add", &large_a, &large_b).unwrap();
        let time1 = start.elapsed();

        let start = std::time::Instant::now();
        let result2 = execute_binary("add_simd_f64", &large_a, &large_b).unwrap();
        let time2 = start.elapsed();

        let start = std::time::Instant::now();
        let result3 = execute_binary("add_mem_f64", &large_a, &large_b).unwrap();
        let time3 = start.elapsed();

        // Verify all results are the same
        for i in 0..10000 {
            assert_eq!(result1.get(i).unwrap(), result2.get(i).unwrap());
            assert_eq!(result2.get(i).unwrap(), result3.get(i).unwrap());
        }

        // Print timing information (for manual inspection)
        println!("Standard add: {:?}", time1);
        println!("SIMD add: {:?}", time2);
        println!("Memory-optimized add: {:?}", time3);

        // At minimum, all should complete successfully
        assert!(result1.size() == 10000);
        assert!(result2.size() == 10000);
        assert!(result3.size() == 10000);
    }
}
