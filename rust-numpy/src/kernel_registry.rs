// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! Type-based kernel dispatch system for ufunc operations

use crate::array::Array;
use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
use std::any::{Any, TypeId};
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
}

/// Kernel registry for type-based kernel dispatch
///
/// This registry stores and manages dtype-specific kernels using TypeId
/// for efficient lookup and dispatch.
pub struct KernelRegistry {
    kernels: HashMap<(TypeId, crate::ufunc::UfuncType), Box<dyn Any>>,
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
    pub fn register<T, K>(&mut self, ufunc: crate::ufunc::UfuncType, kernel: K)
    where
        T: 'static,
        K: Kernel<T> + 'static,
    {
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
    pub fn get<T>(&self, ufunc: crate::ufunc::UfuncType, dtype: TypeId) -> Option<&dyn Kernel<T>>
    where
        T: 'static,
    {
        self.kernels
            .get(&(TypeId::of::<T>(), ufunc))
            .and_then(|k| k.downcast_ref())
    }

    /// Get all registered kernels for a ufunc type
    ///
    /// # Arguments
    /// * `ufunc` - Type of ufunc
    ///
    /// # Returns
    /// * Vector of kernel references that match the ufunc type
    pub fn get_all(&self, ufunc: crate::ufunc::UfuncType) -> Vec<&dyn Kernel<T>>
    where
        T: 'static,
    {
        self.kernels
            .iter()
            .filter(|((type_id, ufunc_type), _)| ufunc_type == *ufunc_type)
            .map(|(_, kernel)| kernel.downcast_ref())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let mut registry = KernelRegistry::new();

        // Test that we can register and retrieve kernels
        assert!(registry.get::<f64>(crate::ufunc::UfuncType::Add).is_none());

        // Register a test kernel
        struct TestAddKernel;
        impl Kernel<f64> for TestAddKernel {
            fn name(&self) -> &str {
                "test_add"
            }
            fn execute(
                &self,
                input: &[&dyn ArrayView],
                output: &mut [&mut dyn ArrayViewMut],
            ) -> Result<()> {
                // Simple addition for testing
                let in0 = unsafe { &*(input[0] as *const Array<f64>) };
                let in1 = unsafe { &*(input[1] as *const Array<f64>) };
                let out = unsafe { &mut *(output[0] as *mut Array<f64>) };

                if let (Some(a), Some(b)) = (in0.get(0), in1.get(0)) {
                    out.set(0, a + b)?;
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

        registry.register::<f64, TestAddKernel>(crate::ufunc::UfuncType::Add, TestAddKernel);

        // Verify registration
        let kernel = registry.get::<f64>(crate::ufunc::UfuncType::Add);
        assert!(kernel.is_some());
        assert_eq!(kernel.unwrap().name(), "test_add");
    }

    #[test]
    fn test_multiple_dtypes() {
        let mut registry = KernelRegistry::new();

        // Register kernels for different types
        registry.register::<f64, TestAddKernel>(crate::ufunc::UfuncType::Add, TestAddKernel);
        registry.register::<f32, TestAddKernel>(crate::ufunc::UfuncType::Add, TestAddKernel);

        // Verify we can retrieve all kernels for Add ufunc
        let add_kernels = registry.get_all(crate::ufunc::UfuncType::Add);
        assert_eq!(add_kernels.len(), 2);

        // Verify type safety
        let f64_kernel = add_kernels[0];
        let f32_kernel = add_kernels[1];

        // Should be able to downcast to correct types
        assert!(!std::any::TypeId::of::<f64>().eq(std::any::TypeId::of::<f32>()));
        assert!(!std::any::TypeId::of::<f64>().eq(std::any::TypeId::of::<f32>()));
    }
}
