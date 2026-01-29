use numpy::dtype::Casting;
use numpy::kernels::{
    dtype_kernels::{
        register_dtype_kernels, F32AddKernel, F64AddKernel, F64MulKernel, I32AddKernel,
    },
    UfuncKernel, UfuncKernelRegistry, UfuncType,
};
use numpy::*;
use std::sync::Arc;

#[cfg(test)]
mod kernel_registry_tests {
    use super::*;

    // #[test]
    // fn test_kernel_registry_creation() {
    //     let registry = UfuncKernelRegistry::new();
    //     let stats = registry.stats().unwrap();
    //     assert_eq!(stats.total_kernels, 0);
    //     assert_eq!(stats.vectorized_kernels, 0);
    // }

    #[test]
    fn test_f64_add_kernel() {
        let kernel = F64AddKernel;
        assert_eq!(kernel.name(), "f64_add_simd");
        assert!(!kernel.is_vectorized());

        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut output = [0.0; 3];

        kernel.execute(&[&a[..], &b[..]], &mut output).unwrap();
        assert_eq!(output, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_f64_mul_kernel() {
        let kernel = F64MulKernel;
        assert_eq!(kernel.name(), "f64_mul_simd");
        assert!(!kernel.is_vectorized());

        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut output = [0.0; 3];

        kernel.execute(&[&a[..], &b[..]], &mut output).unwrap();
        assert_eq!(output, [4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_f32_add_kernel() {
        let kernel = F32AddKernel;
        assert_eq!(kernel.name(), "f32_add");

        let a = [1.0f32, 2.0f32, 3.0f32];
        let b = [4.0f32, 5.0f32, 6.0f32];
        let mut output = [0.0f32; 3];

        kernel.execute(&[&a[..], &b[..]], &mut output).unwrap();
        assert_eq!(output, [5.0f32, 7.0f32, 9.0f32]);
    }

    #[test]
    fn test_i32_add_kernel() {
        let kernel = I32AddKernel;
        assert_eq!(kernel.name(), "i32_add");

        let a = [1i32, 2i32, 3i32];
        let b = [4i32, 5i32, 6i32];
        let mut output = [0i32; 3];

        kernel.execute(&[&a[..], &b[..]], &mut output).unwrap();
        assert_eq!(output, [5i32, 7i32, 9i32]);
    }

    #[test]
    fn test_kernel_registration() {
        let mut registry = UfuncKernelRegistry::new();

        registry.register(UfuncType::Add, F64AddKernel).unwrap();

        let stats = registry.stats().unwrap();
        assert_eq!(stats.total_kernels, 1);
        assert_eq!(stats.ufunc_counts.get(&UfuncType::Add), Some(&1));

        let name = registry.with_kernel::<f64, _, _>(UfuncType::Add, |k| k.name().to_string());
        assert!(name.is_some());
        assert_eq!(name.unwrap(), "f64_add_simd");
    }

    #[test]
    fn test_multiple_kernel_registration() {
        let mut registry = UfuncKernelRegistry::new();

        registry.register(UfuncType::Add, F64AddKernel).unwrap();
        registry
            .register(UfuncType::Multiply, F64MulKernel)
            .unwrap();
        registry.register(UfuncType::Add, F32AddKernel).unwrap();
        registry.register(UfuncType::Add, I32AddKernel).unwrap();

        let stats = registry.stats().unwrap();
        assert_eq!(stats.total_kernels, 4);
        assert_eq!(stats.ufunc_counts.get(&UfuncType::Add), Some(&3));
        assert_eq!(stats.ufunc_counts.get(&UfuncType::Multiply), Some(&1));

        let add_kernels = registry.list_kernels(UfuncType::Add).unwrap();
        assert_eq!(add_kernels.len(), 3);
        assert!(add_kernels.contains(&"f64_add_simd".to_string()));
        assert!(add_kernels.contains(&"f32_add".to_string()));
        assert!(add_kernels.contains(&"i32_add".to_string()));
    }

    #[test]
    fn test_kernel_find_best() {
        let mut registry = UfuncKernelRegistry::new();

        registry.register(UfuncType::Add, F64AddKernel).unwrap();
        registry.register(UfuncType::Add, F32AddKernel).unwrap();
        registry
            .register(UfuncType::Multiply, F64MulKernel)
            .unwrap();

        let best_f64 = registry.find_best_kernel::<f64>(UfuncType::Add);
        assert!(best_f64.is_some());
        let (kernel_name, needs_cast) = best_f64.unwrap();
        assert_eq!(kernel_name, "f64_add_simd");
        assert!(!needs_cast);

        let best_f32 = registry.find_best_kernel::<f32>(UfuncType::Add);
        assert!(best_f32.is_some());
        let (kernel_name, needs_cast) = best_f32.unwrap();
        assert_eq!(kernel_name, "f32_add");
        assert!(!needs_cast);

        let best_i64 = registry.find_best_kernel::<i64>(UfuncType::Add);
        assert!(best_i64.is_none());
    }

    #[test]
    fn test_kernel_error_handling() {
        let kernel = F64AddKernel;

        let a = [1.0, 2.0];
        let mut output = [0.0; 2];
        let result = kernel.execute(&[&a[..]], &mut output);
        assert!(result.is_err());

        let a = [1.0, 2.0];
        let b = [3.0];
        let mut output = [0.0; 2];
        let result = kernel.execute(&[&a, &b], &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtype_kernels_registration() {
        let mut registry = UfuncKernelRegistry::new();

        register_dtype_kernels(&mut registry).unwrap();

        let stats = registry.stats().unwrap();
        assert!(stats.total_kernels >= 7);

        assert!(registry
            .with_kernel::<f64, _, _>(UfuncType::Add, |_| {})
            .is_some());
        assert!(registry
            .with_kernel::<f64, _, _>(UfuncType::Multiply, |_| {})
            .is_some());
        assert!(registry
            .with_kernel::<f32, _, _>(UfuncType::Add, |_| {})
            .is_some());
        assert!(registry
            .with_kernel::<i32, _, _>(UfuncType::Add, |_| {})
            .is_some());
        assert!(registry
            .with_kernel::<i32, _, _>(UfuncType::Multiply, |_| {})
            .is_some());
    }
}

#[cfg(test)]
mod kernel_integration_tests {
    use super::*;

    #[test]
    fn test_array_add_with_kernel_optimization() {
        let a = array![1.0f64, 2.0f64, 3.0f64];
        let b = array![4.0f64, 5.0f64, 6.0f64];

        let result = a.add(&b, None, Casting::SameKind).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.get(0).unwrap(), &5.0f64);
        assert_eq!(result.get(1).unwrap(), &7.0f64);
        assert_eq!(result.get(2).unwrap(), &9.0f64);
    }

    #[test]
    fn test_array_multiply_with_kernel_optimization() {
        let a = array![1.0f64, 2.0f64, 3.0f64];
        let b = array![4.0f64, 5.0f64, 6.0f64];

        let result = a.multiply(&b, None, Casting::SameKind).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.get(0).unwrap(), &4.0f64);
        assert_eq!(result.get(1).unwrap(), &10.0f64);
        assert_eq!(result.get(2).unwrap(), &18.0f64);
    }

    #[test]
    fn test_array_operations_different_dtypes() {
        let a_f32 = array![1.0f32, 2.0f32];
        let b_f32 = array![3.0f32, 4.0f32];
        let result_f32 = a_f32.add(&b_f32, None, Casting::SameKind).unwrap();
        let val = result_f32.get(0).unwrap();
        assert!((val - 4.0f32).abs() < 1e-2, "Expected 4.0, got {}", val);

        let a_i32 = array![1i32, 2i32];
        let b_i32 = array![3i32, 4i32];
        let result_i32 = a_i32.add(&b_i32, None, Casting::SameKind).unwrap();
        assert_eq!(result_i32.len(), 2);
        assert_eq!(result_i32.get(0).unwrap(), &4i32);
    }

    #[test]
    fn test_broadcast_operations_with_kernels() {
        let a = Array::from_shape_vec(vec![2, 2], vec![1.0f64, 2.0f64, 3.0f64, 4.0f64]);
        let b = Array::from_shape_vec(vec![2], vec![10.0f64, 20.0f64]);

        let result = a.add(&b, None, Casting::SameKind).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.get_multi(&[0, 0]).unwrap(), 11.0f64);
        assert_eq!(result.get_multi(&[0, 1]).unwrap(), 22.0f64);
        assert_eq!(result.get_multi(&[1, 0]).unwrap(), 13.0f64);
        assert_eq!(result.get_multi(&[1, 1]).unwrap(), 24.0f64);
    }

    #[test]
    fn test_large_array_operations() {
        let size = 1000;
        let a = Array::from_vec((0..size).map(|i| i as f64).collect());
        let b = Array::from_vec((0..size).map(|i| (i * 2) as f64).collect());

        let result = a.add(&b, None, Casting::SameKind).unwrap();

        assert_eq!(result.len(), size);
        for i in 0..size.min(10) {
            assert_eq!(result.get(i).unwrap(), &(i as f64 + (i * 2) as f64));
        }
    }
}

#[cfg(test)]
mod kernel_performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_kernel_execution_performance() {
        let kernel = F64AddKernel;
        let size = 10000;
        let a: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..size).map(|i| (i * 2) as f64).collect();
        let mut output: Vec<f64> = vec![0.0; size];

        let start = Instant::now();
        kernel.execute(&[&a[..], &b[..]], &mut output).unwrap();
        let duration = start.elapsed();

        assert_eq!(output.len(), size);

        for i in 0..size.min(100) {
            assert_eq!(output[i], a[i] + b[i]);
        }

        println!(
            "Kernel execution time for {} elements: {:?}",
            size, duration
        );
    }

    #[test]
    fn test_multiple_kernel_types_performance() {
        let size = 5000;

        let f64_kernel = F64AddKernel;
        let a_f64: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let b_f64: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let mut output_f64: Vec<f64> = vec![0.0; size];

        let start_f64 = Instant::now();
        f64_kernel
            .execute(&[&a_f64[..], &b_f64[..]], &mut output_f64)
            .unwrap();
        let duration_f64 = start_f64.elapsed();

        let f32_kernel = F32AddKernel;
        let a_f32: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b_f32: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let mut output_f32: Vec<f32> = vec![0.0; size];

        let start_f32 = Instant::now();
        f32_kernel
            .execute(&[&a_f32[..], &b_f32[..]], &mut output_f32)
            .unwrap();
        let duration_f32 = start_f32.elapsed();

        let i32_kernel = I32AddKernel;
        let a_i32: Vec<i32> = (0..size).map(|i| i as i32).collect();
        let b_i32: Vec<i32> = (0..size).map(|i| i as i32).collect();
        let mut output_i32: Vec<i32> = vec![0; size];

        let start_i32 = Instant::now();
        i32_kernel
            .execute(&[&a_i32[..], &b_i32[..]], &mut output_i32)
            .unwrap();
        let duration_i32 = start_i32.elapsed();

        println!("Performance comparison for {} elements:", size);
        println!("  f64 kernel: {:?}", duration_f64);
        println!("  f32 kernel: {:?}", duration_f32);
        println!("  i32 kernel: {:?}", duration_i32);

        assert!(duration_f64.as_millis() < 1000);
        assert!(duration_f32.as_millis() < 1000);
        assert!(duration_i32.as_millis() < 1000);
    }
}
