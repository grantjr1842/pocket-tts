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
use crate::ufunc::{ArrayView, ArrayViewMut};
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
pub trait Kernel<T: 'static>: Send + Sync {
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
    kernels: HashMap<(TypeId, crate::kernels::UfuncType), Box<dyn Any>>,
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
    pub fn register<T, K>(&mut self, ufunc: crate::kernels::UfuncType, kernel: K)
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
    ///
    /// TODO: This method needs architectural redesign. We cannot downcast from
    /// `dyn Any` to `dyn Kernel<T>` directly because `dyn Kernel<T>` is unsized.
    /// The registry needs to store kernels differently to support retrieval.
    pub fn get<T>(&self, _ufunc: crate::kernels::UfuncType, _dtype: TypeId) -> Option<&dyn Kernel<T>>
    where
        T: 'static,
    {
        // Cannot downcast to unsized trait object
        // This requires redesign of the kernel storage approach
        None
    }

    /// Get all registered kernels for a ufunc type
    ///
    /// # Arguments
    /// * `ufunc` - Type of ufunc
    ///
    /// # Returns
    /// * Vector of kernel references that match the ufunc type
    ///
    /// TODO: This method needs architectural redesign. We cannot downcast from
    /// `dyn Any` to `dyn Kernel<T>` directly because `dyn Kernel<T>` is unsized.
    pub fn get_all<T: 'static>(&self, _ufunc: crate::kernels::UfuncType) -> Vec<&dyn Kernel<T>> {
        // Cannot downcast to unsized trait object
        // This requires redesign of the kernel storage approach
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let mut registry = KernelRegistry::new();

        // Test that registry can be created
        assert_eq!(registry.kernels.len(), 0);

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

        registry.register::<f64, TestAddKernel>(crate::kernels::UfuncType::Add, TestAddKernel);

        // Verify kernel was stored (internal check)
        assert_eq!(registry.kernels.len(), 1);
    }

    // TODO: Re-enable retrieval tests once the downcasting issue is resolved
    // The current architecture doesn't support retrieving kernels as trait objects
    // because dyn Kernel<T> is unsized and cannot be used with downcast_ref
}
