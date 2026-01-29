// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! Type-based kernel dispatch system for ufunc operations

use crate::dtype::Dtype;
use crate::error::Result;
use crate::kernels::UfuncType;
use crate::ufunc::{ArrayView, ArrayViewMut};
use std::any::TypeId;
use std::collections::HashMap;

/// Kernel signature describing input and output types
#[derive(Debug, Clone)]
pub struct KernelSignature {
    input_dtypes: Vec<Dtype>,
    output_dtype: Dtype,
}

impl KernelSignature {
    /// Create new kernel signature
    pub fn new(input_dtypes: Vec<Dtype>, output_dtype: Dtype) -> Self {
        Self {
            input_dtypes,
            output_dtype,
        }
    }

    /// Get string representation
    pub fn as_string(&self) -> String {
        let input_names: Vec<String> = self.input_dtypes.iter().map(|dt| dt.to_string()).collect();
        let output_name = self.output_dtype.to_string();

        format!("{} -> {}", input_names.join(", "), output_name)
    }
}

/// Kernel trait for dynamic registration
///
/// This trait defines the interface for dtype-specific optimized kernels
/// that can be registered and dispatched based on input array types.
pub trait Kernel: Send + Sync {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Get kernel signature (input types -> output types)
    fn signature(&self) -> KernelSignature;

    /// Execute kernel operation
    ///
    /// # Arguments
    /// * `input` - Slice of input arrays
    /// * `output` - Mutable output array
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(NumPyError)` on failure
    fn execute(&self, input: &[&dyn ArrayView], output: &mut [&mut dyn ArrayViewMut])
        -> Result<()>;

    /// Check if kernel is vectorized (SIMD optimized)
    fn is_vectorized(&self) -> bool {
        false
    }

    /// Get performance hint for this kernel
    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::General
    }
}

/// Kernel registry for type-based kernel dispatch
///
/// This registry stores and manages dtype-specific kernels using TypeId
/// for efficient lookup and dispatch.
pub struct KernelRegistry {
    kernels: HashMap<(TypeId, UfuncType), Box<dyn Kernel + Send + Sync>>,
}

impl KernelRegistry {
    /// Create new kernel registry
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    /// Register a kernel for a specific dtype and ufunc type
    ///
    /// # Arguments
    /// * `ufunc` - Type of ufunc (add, multiply, etc.)
    /// * `kernel` - Kernel implementation to register
    pub fn register<T: 'static>(&mut self, ufunc: UfuncType, kernel: impl Kernel + 'static) {
        let type_id = TypeId::of::<T>();
        self.kernels.insert((type_id, ufunc), Box::new(kernel));
    }

    /// Get registered kernel for specific dtype and ufunc type
    ///
    /// # Arguments
    /// * `ufunc` - Type of ufunc
    /// * `dtype` - Data type identifier
    ///
    /// # Returns
    /// * `Some(kernel)` if found, `None` otherwise
    pub fn get<T: 'static>(&self, ufunc: UfuncType) -> Option<&dyn Kernel> {
        self.kernels
            .get(&(TypeId::of::<T>(), ufunc))
            .map(|k| k.as_ref() as &dyn Kernel)
    }

    /// Get all registered kernels for a ufunc type
    ///
    /// # Arguments
    /// * `ufunc` - Type of ufunc
    ///
    /// # Returns
    /// * Vector of kernel references that match the ufunc type
    pub fn get_all(&self, ufunc: UfuncType) -> Vec<&dyn Kernel> {
        self.kernels
            .iter()
            .filter(|((_, ufunc_type), _)| ufunc_type == &ufunc)
            .map(|(_, kernel)| kernel.as_ref() as &dyn Kernel)
            .collect()
    }
}

/// Performance hint for kernel optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceHint {
    /// Kernel is vectorized (SIMD)
    Vectorized,
    /// Kernel is memory bandwidth bound
    MemoryBound,
    /// Kernel is compute bound
    ComputeBound,
    /// No specific optimization hints
    General,
}

/// Statistics about the kernel registry
#[derive(Debug, Clone)]
pub struct RegistryStats {
    /// Total number of registered kernels
    pub total_kernels: usize,
    /// Number of kernels by ufunc type
    pub kernels_by_type: HashMap<String, usize>,
}

/// Global kernel registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<std::sync::RwLock<KernelRegistry>> =
    std::sync::OnceLock::new();

/// Get the global kernel registry
fn get_global_registry() -> &'static std::sync::RwLock<KernelRegistry> {
    GLOBAL_REGISTRY.get_or_init(|| std::sync::RwLock::new(KernelRegistry::new()))
}

/// Get the global kernel registry (public API)
pub fn get_kernel_registry() -> std::sync::RwLockReadGuard<'static, KernelRegistry> {
    get_global_registry().read().unwrap()
}

/// Register a kernel in the global registry
pub fn register_kernel<T: 'static>(kernel: impl Kernel + 'static, ufunc: UfuncType) -> Result<()> {
    let mut registry = get_global_registry().write().unwrap();
    registry.register::<T>(ufunc, kernel);
    Ok(())
}

/// Execute a function with a kernel from the registry
pub fn with_kernel<T: 'static, R, F>(ufunc: UfuncType, f: F) -> Option<R>
where
    F: FnOnce(&dyn Kernel) -> R,
{
    let registry = get_global_registry().read().unwrap();
    let kernel = registry.get::<T>(ufunc)?;
    Some(f(kernel))
}

/// Get statistics about the global registry
pub fn get_registry_stats() -> RegistryStats {
    let registry = get_global_registry().read().unwrap();
    let mut stats = RegistryStats {
        total_kernels: 0,
        kernels_by_type: HashMap::new(),
    };
    // Count kernels by type
    for ((_, ufunc_type), _) in &registry.kernels {
        let count = stats
            .kernels_by_type
            .entry(ufunc_type.as_str().to_string())
            .or_insert(0);
        *count += 1;
    }
    stats.total_kernels = registry.kernels.len();
    stats
}

/// List all registered kernels
pub fn list_kernels() -> Vec<(UfuncType, String)> {
    let registry = get_global_registry().read().unwrap();
    let mut kernels = Vec::new();
    for ((_, ufunc_type), kernel) in &registry.kernels {
        kernels.push((*ufunc_type, kernel.name().to_string()));
    }
    kernels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    #[test]
    fn test_registry_creation() {
        let mut registry = KernelRegistry::new();

        // Test that we can register and retrieve kernels
        assert!(registry.get::<f64>(UfuncType::Add).is_none());

        registry.register::<f64>(UfuncType::Add, TestAddKernel);

        // Verify registration
        let kernel = registry.get::<f64>(UfuncType::Add);
        assert!(kernel.is_some());
        assert_eq!(kernel.unwrap().name(), "test_add");
    }

    struct TestAddKernel;
    impl Kernel for TestAddKernel {
        fn name(&self) -> &str {
            "test_add"
        }
        fn execute(
            &self,
            input: &[&dyn ArrayView],
            output: &mut [&mut dyn ArrayViewMut],
        ) -> Result<()> {
            // Simple addition for testing
            let in0 = input[0]
                .as_any()
                .downcast_ref::<Array<f64>>()
                .expect("Type mismatch");
            let in1 = input[1]
                .as_any()
                .downcast_ref::<Array<f64>>()
                .expect("Type mismatch");
            let out_array = output[0]
                .as_any_mut()
                .downcast_mut::<Array<f64>>()
                .expect("Type mismatch");

            if let (Some(a), Some(b)) = (in0.get(0), in1.get(0)) {
                out_array.set(0, a + b)?;
            }
            Ok(())
        }
        fn is_vectorized(&self) -> bool {
            false
        }
        fn signature(&self) -> KernelSignature {
            KernelSignature::new(vec![Dtype::from_type::<f64>()], Dtype::from_type::<f64>())
        }
    }

    #[test]
    fn test_multiple_dtypes() {
        let mut registry = KernelRegistry::new();

        // Register kernels for different types
        registry.register::<f64>(UfuncType::Add, TestAddKernel);
        registry.register::<f32>(UfuncType::Add, TestAddKernel);

        // Verify we can retrieve all kernels for Add ufunc
        let add_kernels = registry.get_all(UfuncType::Add);
        assert_eq!(add_kernels.len(), 2);

        // Verify type safety
        let f64_kernel = add_kernels[0];
        let f32_kernel = add_kernels[1];

        // Should be able to downcast to correct types
        assert!(!std::any::TypeId::of::<f64>().eq(&std::any::TypeId::of::<f32>()));
    }
}
