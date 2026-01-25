use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{NumPyError, Result};
use crate::kernel_impls::{BinaryKernel, MemoryOptimizedKernel, SimdBinaryKernel, UnaryKernel};
use crate::kernel_registry::{
    find_kernel, get_registry_stats, list_kernels, register_kernel, PerformanceHint,
};
use std::sync::Arc;

/// Runtime API for dynamic kernel registration and execution

/// Register a binary operation kernel
pub fn register_binary_kernel<T, F>(name: &str, operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    let kernel = Arc::new(BinaryKernel::new(name, operation));
    register_kernel(kernel)
}

/// Register a binary operation kernel with performance hint
pub fn register_binary_kernel_with_hint<T, F>(
    name: &str,
    operation: F,
    hint: PerformanceHint,
) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    let kernel = Arc::new(BinaryKernel::new(name, operation).with_performance_hint(hint));
    register_kernel(kernel)
}

/// Register a unary operation kernel
pub fn register_unary_kernel<T, F>(name: &str, operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T) -> T + Send + Sync + 'static,
{
    let kernel = Arc::new(UnaryKernel::new(name, operation));
    register_kernel(kernel)
}

/// Register a unary operation kernel with performance hint
pub fn register_unary_kernel_with_hint<T, F>(
    name: &str,
    operation: F,
    hint: PerformanceHint,
) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T) -> T + Send + Sync + 'static,
{
    let kernel = Arc::new(UnaryKernel::new(name, operation).with_performance_hint(hint));
    register_kernel(kernel)
}

/// Register a SIMD-optimized binary kernel
pub fn register_simd_binary_kernel<T, F>(name: &str, operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    let kernel = Arc::new(SimdBinaryKernel::new(name, operation));
    register_kernel(kernel)
}

/// Register a memory-optimized binary kernel
pub fn register_memory_optimized_kernel<T, F>(name: &str, operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    let kernel = Arc::new(MemoryOptimizedKernel::new(name, operation));
    register_kernel(kernel)
}

/// Execute a binary operation using the dynamic kernel registry
pub fn execute_binary<T>(name: &str, a: &Array<T>, b: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let input_dtypes = vec![a.dtype().clone(), b.dtype().clone()];
    let kernel = find_kernel(name, &input_dtypes)?
        .ok_or_else(|| NumPyError::ufunc_error(name, "No kernel found for input types"))?;

    // Create output array
    let output_shape = crate::broadcasting::compute_broadcast_shape(a.shape(), b.shape());
    let mut output = Array::zeros(output_shape);

    // Broadcast inputs if needed
    let broadcasted = crate::broadcasting::broadcast_arrays(&[a, b])?;
    let input_views: Vec<&dyn crate::ufunc::ArrayView> = broadcasted
        .iter()
        .map(|arr| arr as &dyn crate::ufunc::ArrayView)
        .collect();

    let mut output_views: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

    kernel.execute(&input_views, &mut output_views)?;

    Ok(output)
}

/// Execute a unary operation using the dynamic kernel registry
pub fn execute_unary<T>(name: &str, a: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let input_dtypes = vec![a.dtype().clone()];
    let kernel = find_kernel(name, &input_dtypes)?
        .ok_or_else(|| NumPyError::ufunc_error(name, "No kernel found for input type"))?;

    // Create output array
    let mut output = a.clone();

    let input_views: Vec<&dyn crate::ufunc::ArrayView> = vec![a];
    let mut output_views: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

    kernel.execute(&input_views, &mut output_views)?;

    Ok(output)
}

/// Register common mathematical kernels
pub fn register_common_kernels() -> Result<()> {
    // Addition kernels
    register_binary_kernel_with_hint("add", |a: f64, b: f64| a + b, PerformanceHint::Vectorized)?;
    register_binary_kernel_with_hint("add", |a: f32, b: f32| a + b, PerformanceHint::Vectorized)?;
    register_binary_kernel_with_hint("add", |a: i64, b: i64| a + b, PerformanceHint::General)?;
    register_binary_kernel_with_hint("add", |a: i32, b: i32| a + b, PerformanceHint::General)?;

    // SIMD-optimized addition for large arrays
    register_simd_binary_kernel("add_simd_f64", |a: f64, b: f64| a + b)?;
    register_simd_binary_kernel("add_simd_f32", |a: f32, b: f32| a + b)?;

    // Memory-optimized addition for very large arrays
    register_memory_optimized_kernel("add_mem_f64", |a: f64, b: f64| a + b)?;
    register_memory_optimized_kernel("add_mem_f32", |a: f32, b: f32| a + b)?;

    // Subtraction kernels
    register_binary_kernel_with_hint(
        "subtract",
        |a: f64, b: f64| a - b,
        PerformanceHint::Vectorized,
    )?;
    register_binary_kernel_with_hint(
        "subtract",
        |a: f32, b: f32| a - b,
        PerformanceHint::Vectorized,
    )?;
    register_binary_kernel_with_hint("subtract", |a: i64, b: i64| a - b, PerformanceHint::General)?;
    register_binary_kernel_with_hint("subtract", |a: i32, b: i32| a - b, PerformanceHint::General)?;

    // Multiplication kernels
    register_binary_kernel_with_hint(
        "multiply",
        |a: f64, b: f64| a * b,
        PerformanceHint::Vectorized,
    )?;
    register_binary_kernel_with_hint(
        "multiply",
        |a: f32, b: f32| a * b,
        PerformanceHint::Vectorized,
    )?;
    register_binary_kernel_with_hint("multiply", |a: i64, b: i64| a * b, PerformanceHint::General)?;
    register_binary_kernel_with_hint("multiply", |a: i32, b: i32| a * b, PerformanceHint::General)?;

    // Division kernels
    register_binary_kernel_with_hint(
        "divide",
        |a: f64, b: f64| a / b,
        PerformanceHint::ComputeBound,
    )?;
    register_binary_kernel_with_hint(
        "divide",
        |a: f32, b: f32| a / b,
        PerformanceHint::ComputeBound,
    )?;
    register_binary_kernel_with_hint("divide", |a: i64, b: i64| a / b, PerformanceHint::General)?;
    register_binary_kernel_with_hint("divide", |a: i32, b: i32| a / b, PerformanceHint::General)?;

    // Unary kernels
    register_unary_kernel_with_hint("negative", |a: f64| -a, PerformanceHint::Vectorized)?;
    register_unary_kernel_with_hint("negative", |a: f32| -a, PerformanceHint::Vectorized)?;
    register_unary_kernel_with_hint("negative", |a: i64| -a, PerformanceHint::General)?;
    register_unary_kernel_with_hint("negative", |a: i32| -a, PerformanceHint::General)?;

    register_unary_kernel_with_hint("absolute", |a: f64| a.abs(), PerformanceHint::Vectorized)?;
    register_unary_kernel_with_hint("absolute", |a: f32| a.abs(), PerformanceHint::Vectorized)?;
    register_unary_kernel_with_hint("absolute", |a: i64| a.abs(), PerformanceHint::General)?;
    register_unary_kernel_with_hint("absolute", |a: i32| a.abs(), PerformanceHint::General)?;

    Ok(())
}

/// Kernel registry builder for convenient registration
pub struct KernelRegistryBuilder {
    _private: (),
}

impl KernelRegistryBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Register common kernels
    pub fn with_common_kernels(self) -> Result<Self> {
        register_common_kernels()?;
        Ok(self)
    }

    /// Register a custom binary kernel
    pub fn with_binary_kernel<T, F>(self, name: &str, operation: F) -> Result<Self>
    where
        T: Clone + Default + 'static + Send + Sync,
        F: Fn(T, T) -> T + Send + Sync,
    {
        register_binary_kernel(name, operation)?;
        Ok(self)
    }

    /// Register a custom unary kernel
    pub fn with_unary_kernel<T, F>(self, name: &str, operation: F) -> Result<Self>
    where
        T: Clone + Default + 'static + Send + Sync,
        F: Fn(T) -> T + Send + Sync,
    {
        register_unary_kernel(name, operation)?;
        Ok(self)
    }

    /// Build (finalizes registration)
    pub fn build(self) -> Result<()> {
        // In the future, this could perform validation or optimization
        Ok(())
    }
}

impl Default for KernelRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to initialize the kernel registry
pub fn init_kernel_registry() -> Result<()> {
    KernelRegistryBuilder::new().with_common_kernels()?.build()
}

/// Get information about registered kernels
pub fn get_kernel_info() -> Result<KernelInfo> {
    let kernel_names = list_kernels()?;
    let stats = get_registry_stats()?;

    Ok(KernelInfo {
        kernel_names,
        stats,
    })
}

/// Information about registered kernels
#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub kernel_names: Vec<String>,
    pub stats: crate::kernel_registry::RegistryStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;

    #[test]
    fn test_kernel_registration() {
        // Register a simple kernel
        register_binary_kernel("test_add", |a: i32, b: i32| a + b).unwrap();

        // Verify it was registered
        let kernel_names = list_kernels().unwrap();
        assert!(kernel_names.contains(&"test_add".to_string()));
    }

    #[test]
    fn test_kernel_execution() {
        // Register a kernel
        register_binary_kernel("test_mul", |a: i32, b: i32| a * b).unwrap();

        // Create test arrays
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![4, 5, 6]);

        // Execute the kernel
        let result = execute_binary("test_mul", &a, &b).unwrap();

        // Verify result
        assert_eq!(result.get(0).unwrap(), &4);
        assert_eq!(result.get(1).unwrap(), &10);
        assert_eq!(result.get(2).unwrap(), &18);
    }

    #[test]
    fn test_registry_builder() {
        let builder = KernelRegistryBuilder::new()
            .with_binary_kernel("builder_add", |a: f64, b: f64| a + b)
            .with_unary_kernel("builder_neg", |a: f64| -a)
            .build();

        assert!(builder.is_ok());

        let kernel_names = list_kernels().unwrap();
        assert!(kernel_names.contains(&"builder_add".to_string()));
        assert!(kernel_names.contains(&"builder_neg".to_string()));
    }
}
