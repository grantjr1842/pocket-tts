use crate::array::Array;
use crate::error::{NumPyError, Result};
use crate::kernel_impls::{BinaryKernel, MemoryOptimizedKernel, SimdBinaryKernel, UnaryKernel};
use crate::kernel_registry::{
    get_registry_stats, list_kernels, register_kernel, with_kernel, PerformanceHint,
};
use crate::kernels::UfuncType;

use crate::array::Array;
use crate::error::{NumPyError, Result};
use crate::kernels::UfuncPerformanceHint as PerformanceHint;

/// Register a binary operation kernel
///
/// TODO: Not yet implemented - needs wrapper to adapt BinaryKernel to KernelFunction
pub fn register_binary_kernel<T, F>(_name: &str, _operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    let ufunc: UfuncType = name.parse()?;
    let kernel = BinaryKernel::new(name, operation);
    register_kernel::<T>(kernel, ufunc)
}

/// Register a binary operation kernel with performance hint
///
/// TODO: Not yet implemented
pub fn register_binary_kernel_with_hint<T, F>(
    _name: &str,
    _operation: F,
    _hint: PerformanceHint,
) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    let ufunc: UfuncType = name.parse()?;
    let kernel = BinaryKernel::new(name, operation).with_performance_hint(hint);
    register_kernel::<T>(kernel, ufunc)
}

/// Register a unary operation kernel
///
/// TODO: Not yet implemented
pub fn register_unary_kernel<T, F>(_name: &str, _operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T) -> T + Send + Sync + 'static,
{
    let ufunc: UfuncType = name.parse()?;
    let kernel = UnaryKernel::new(name, operation);
    register_kernel::<T>(kernel, ufunc)
}

/// Register a unary operation kernel with performance hint
///
/// TODO: Not yet implemented
pub fn register_unary_kernel_with_hint<T, F>(
    _name: &str,
    _operation: F,
    _hint: PerformanceHint,
) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T) -> T + Send + Sync + 'static,
{
    let ufunc: UfuncType = name.parse()?;
    let kernel = UnaryKernel::new(name, operation).with_performance_hint(hint);
    register_kernel::<T>(kernel, ufunc)
}

/// Register a SIMD-optimized binary kernel
///
/// TODO: Not yet implemented
pub fn register_simd_binary_kernel<T, F>(_name: &str, _operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    let ufunc: UfuncType = name.parse()?;
    let kernel = SimdBinaryKernel::new(name, operation);
    register_kernel::<T>(kernel, ufunc)
}

/// Register a memory-optimized binary kernel
///
/// TODO: Not yet implemented
pub fn register_memory_optimized_kernel<T, F>(_name: &str, _operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    let ufunc: UfuncType = name.parse()?;
    let kernel = MemoryOptimizedKernel::new(name, operation);
    register_kernel::<T>(kernel, ufunc)
}

/// Execute a binary operation using the dynamic kernel registry
///
/// TODO: Not yet implemented - needs kernel lookup and execution
pub fn execute_binary<T>(_name: &str, _a: &Array<T>, _b: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    // Parse ufunc name
    let ufunc: UfuncType = name.parse()?;

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

    with_kernel::<T, _, _>(ufunc, |kernel| {
        kernel.execute(&input_views, &mut output_views)
    })
    .ok_or_else(|| NumPyError::ufunc_error(name, "No kernel found for input types"))??;

    Ok(output)
}

/// Execute a unary operation using the dynamic kernel registry
///
/// TODO: Not yet implemented
pub fn execute_unary<T>(_name: &str, _a: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let ufunc: UfuncType = name.parse()?;

    // Create output array
    let mut output = a.clone();

    let input_views: Vec<&dyn crate::ufunc::ArrayView> = vec![a];
    let mut output_views: Vec<&mut dyn crate::ufunc::ArrayViewMut> = vec![&mut output];

    with_kernel::<T, _, _>(ufunc, |kernel| {
        kernel.execute(&input_views, &mut output_views)
    })
    .ok_or_else(|| NumPyError::ufunc_error(name, "No kernel found for input type"))??;

    Ok(output)
}

/// Register common mathematical kernels
///
/// TODO: Not yet implemented
pub fn register_common_kernels() -> Result<()> {
    // Addition kernels
    register_binary_kernel_with_hint("add", |a: f64, b: f64| a + b, PerformanceHint::Vectorized)?;
    register_binary_kernel_with_hint("add", |a: f32, b: f32| a + b, PerformanceHint::Vectorized)?;
    register_binary_kernel_with_hint("add", |a: i64, b: i64| a + b, PerformanceHint::General)?;
    register_binary_kernel_with_hint("add", |a: i32, b: i32| a + b, PerformanceHint::General)?;

    // SIMD-optimized addition for large arrays
    register_simd_binary_kernel("add", |a: f64, b: f64| a + b)?;
    register_simd_binary_kernel("add", |a: f32, b: f32| a + b)?;

    // Memory-optimized addition for very large arrays
    register_memory_optimized_kernel("add", |a: f64, b: f64| a + b)?;
    register_memory_optimized_kernel("add", |a: f32, b: f32| a + b)?;

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
///
/// TODO: Not yet implemented
pub struct KernelRegistryBuilder {
    _private: (),
}

impl KernelRegistryBuilder {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn with_binary_kernel<T, F>(self, _name: &str, _operation: F) -> Result<Self>
    where
        T: Clone + Default + 'static + Send + Sync,
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        Err(NumPyError::not_implemented(
            "KernelRegistryBuilder: Not yet implemented",
        ))
    }

    pub fn with_unary_kernel<T, F>(self, _name: &str, _operation: F) -> Result<Self>
    where
        T: Clone + Default + 'static + Send + Sync,
        F: Fn(T) -> T + Send + Sync + 'static,
    {
        Err(NumPyError::not_implemented(
            "KernelRegistryBuilder: Not yet implemented",
        ))
    }

    pub fn with_common_kernels(self) -> Result<Self> {
        Err(NumPyError::not_implemented(
            "KernelRegistryBuilder: Not yet implemented",
        ))
    }

    /// Register a custom binary kernel
    pub fn with_binary_kernel<T, F>(self, name: &str, operation: F) -> Result<Self>
    where
        T: Clone + Default + 'static + Send + Sync,
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        register_binary_kernel(name, operation)?;
        Ok(self)
    }

    /// Register a custom unary kernel
    pub fn with_unary_kernel<T, F>(self, name: &str, operation: F) -> Result<Self>
    where
        T: Clone + Default + 'static + Send + Sync,
        F: Fn(T) -> T + Send + Sync + 'static,
    {
        register_unary_kernel(name, operation)?;
        Ok(self)
    }

    /// Build (finalizes registration)
    pub fn build(self) -> Result<()> {
        Err(NumPyError::not_implemented(
            "KernelRegistryBuilder: Not yet implemented",
        ))
    }
}

impl Default for KernelRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize the kernel registry
///
/// TODO: Not yet implemented
pub fn init_kernel_registry() -> Result<()> {
    Err(NumPyError::not_implemented(
        "init_kernel_registry: Not yet implemented",
    ))
}

/// Get information about registered kernels
///
/// TODO: Not yet implemented
pub fn get_kernel_info() -> Result<KernelInfo> {
    let kernel_names = list_kernels().into_iter().map(|(_, name)| name).collect();
    let stats = get_registry_stats();

    Ok(KernelInfo {
        kernel_names,
        stats,
    })
}

/// Information about registered kernels
#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub kernel_names: Vec<String>,
    pub stats: crate::dynamic_kernel_registry::RegistryStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_registration() {
        // Register a simple kernel
        register_binary_kernel("add", |a: i32, b: i32| a + b).unwrap();

        // Verify it was registered
        let kernel_names = list_kernels();
        assert!(kernel_names.iter().any(|(_, name)| name == "add"));
    }

    #[test]
    fn test_kernel_execution() {
        // Register a kernel
        register_binary_kernel("multiply", |a: i32, b: i32| a * b).unwrap();

        // Create test arrays
        let a = Array::from_vec(vec![1, 2, 3]);
        let b = Array::from_vec(vec![4, 5, 6]);

        // Execute the kernel
        let result = execute_binary("multiply", &a, &b).unwrap();

        // Verify result
        assert_eq!(result.get(0).unwrap(), &4);
        assert_eq!(result.get(1).unwrap(), &10);
        assert_eq!(result.get(2).unwrap(), &18);
    }

    #[test]
    fn test_registry_builder() -> std::result::Result<(), Box<dyn std::error::Error>> {
        KernelRegistryBuilder::new()
            .with_binary_kernel("add", |a: f64, b: f64| a + b)?
            .with_unary_kernel("negative", |a: f64| -a)?
            .build()?;

        let kernel_names = list_kernels();
        assert!(kernel_names.iter().any(|(_, name)| name == "add"));
        assert!(kernel_names.iter().any(|(_, name)| name == "negative"));
        Ok(())
    }
}
