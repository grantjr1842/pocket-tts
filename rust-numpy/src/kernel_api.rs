//! Kernel API for dynamic registration and execution
//!
//! NOTE: This module is partially implemented. The dynamic kernel registry
//! system exists but the wrapper functions to bridge kernel_impls to it
//! are not yet complete. These are stubbed out for now.

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
    Err(NumPyError::not_implemented(
        "register_binary_kernel: Kernel wrapper adapter not yet implemented",
    ))
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
    Err(NumPyError::not_implemented(
        "register_binary_kernel_with_hint: Kernel wrapper adapter not yet implemented",
    ))
}

/// Register a unary operation kernel
///
/// TODO: Not yet implemented
pub fn register_unary_kernel<T, F>(_name: &str, _operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T) -> T + Send + Sync + 'static,
{
    Err(NumPyError::not_implemented(
        "register_unary_kernel: Kernel wrapper adapter not yet implemented",
    ))
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
    Err(NumPyError::not_implemented(
        "register_unary_kernel_with_hint: Kernel wrapper adapter not yet implemented",
    ))
}

/// Register a SIMD-optimized binary kernel
///
/// TODO: Not yet implemented
pub fn register_simd_binary_kernel<T, F>(_name: &str, _operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    Err(NumPyError::not_implemented(
        "register_simd_binary_kernel: Kernel wrapper adapter not yet implemented",
    ))
}

/// Register a memory-optimized binary kernel
///
/// TODO: Not yet implemented
pub fn register_memory_optimized_kernel<T, F>(_name: &str, _operation: F) -> Result<()>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    Err(NumPyError::not_implemented(
        "register_memory_optimized_kernel: Kernel wrapper adapter not yet implemented",
    ))
}

/// Execute a binary operation using the dynamic kernel registry
///
/// TODO: Not yet implemented - needs kernel lookup and execution
pub fn execute_binary<T>(_name: &str, _a: &Array<T>, _b: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    Err(NumPyError::not_implemented(
        "execute_binary: Dynamic kernel execution not yet implemented",
    ))
}

/// Execute a unary operation using the dynamic kernel registry
///
/// TODO: Not yet implemented
pub fn execute_unary<T>(_name: &str, _a: &Array<T>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    Err(NumPyError::not_implemented(
        "execute_unary: Dynamic kernel execution not yet implemented",
    ))
}

/// Register common mathematical kernels
///
/// TODO: Not yet implemented
pub fn register_common_kernels() -> Result<()> {
    Err(NumPyError::not_implemented(
        "register_common_kernels: Auto-registration not yet implemented",
    ))
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
    Err(NumPyError::not_implemented(
        "get_kernel_info: Not yet implemented",
    ))
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
    fn test_stub_returns_error() {
        let result = register_binary_kernel("test", |a: i32, b: i32| a + b);
        assert!(result.is_err());

        let result = execute_binary("test", &Array::from_vec(vec![1]), &Array::from_vec(vec![2]));
        assert!(result.is_err());
    }
}
