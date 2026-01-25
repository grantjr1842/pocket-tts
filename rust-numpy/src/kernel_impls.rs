use crate::array::Array;
use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
use crate::kernel_registry::{Kernel, KernelSignature, PerformanceHint};
use std::marker::PhantomData;

/// Generic binary kernel implementation
pub struct BinaryKernel<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    name: String,
    operation: F,
    phantom: PhantomData<T>,
    performance_hint: PerformanceHint,
}

impl<T, F> BinaryKernel<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    /// Create new binary kernel
    pub fn new(name: &str, operation: F) -> Self {
        Self {
            name: name.to_string(),
            operation,
            phantom: PhantomData,
            performance_hint: PerformanceHint::General,
        }
    }

    /// Set performance hint
    pub fn with_performance_hint(mut self, hint: PerformanceHint) -> Self {
        self.performance_hint = hint;
        self
    }
}

impl<T, F> Kernel for BinaryKernel<T, F>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> KernelSignature {
        KernelSignature::new(
            vec![Dtype::from_type::<T>(), Dtype::from_type::<T>()],
            vec![Dtype::from_type::<T>()],
        )
    }

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                &self.name,
                format!(
                    "Expected 2 inputs, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let input1 = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let input2 = unsafe { &*(inputs[1] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };

        // Fast path for contiguous arrays
        if input1.is_c_contiguous()
            && input2.is_c_contiguous()
            && output.is_c_contiguous()
            && input1.shape() == input2.shape()
            && input1.shape() == output.shape()
        {
            let data1 = input1.data.as_slice();
            let data2 = input2.data.as_slice();
            let out_slice = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, output.size())
            };

            for i in 0..output.size() {
                out_slice[i] = (self.operation)(
                    data1[input1.offset + i].clone(),
                    data2[input2.offset + i].clone(),
                );
            }
        } else {
            // Generic path using broadcasting
            let broadcasted = crate::broadcasting::broadcast_arrays(&[input1, input2])?;
            let arr1 = &broadcasted[0];
            let arr2 = &broadcasted[1];

            for i in 0..output.size() {
                if let (Some(a), Some(b)) = (arr1.get(i), arr2.get(i)) {
                    output.set(i, (self.operation)(a.clone(), b.clone()))?;
                }
            }
        }

        Ok(())
    }

    fn performance_hint(&self) -> PerformanceHint {
        self.performance_hint
    }
}

/// Generic unary kernel implementation
pub struct UnaryKernel<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T) -> T + Send + Sync,
{
    name: String,
    operation: F,
    phantom: PhantomData<T>,
    performance_hint: PerformanceHint,
}

impl<T, F> UnaryKernel<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T) -> T + Send + Sync,
{
    /// Create new unary kernel
    pub fn new(name: &str, operation: F) -> Self {
        Self {
            name: name.to_string(),
            operation,
            phantom: PhantomData,
            performance_hint: PerformanceHint::General,
        }
    }

    /// Set performance hint
    pub fn with_performance_hint(mut self, hint: PerformanceHint) -> Self {
        self.performance_hint = hint;
        self
    }
}

impl<T, F> Kernel for UnaryKernel<T, F>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T) -> T + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> KernelSignature {
        KernelSignature::new(vec![Dtype::from_type::<T>()], vec![Dtype::from_type::<T>()])
    }

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                &self.name,
                format!(
                    "Expected 1 input, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let input = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };

        // Fast path for contiguous arrays
        if input.is_c_contiguous() && output.is_c_contiguous() && input.shape() == output.shape() {
            let data = input.data.as_slice();
            let out_slice = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, output.size())
            };

            for i in 0..output.size() {
                out_slice[i] = (self.operation)(data[input.offset + i].clone());
            }
        } else {
            for i in 0..input.size() {
                if let Some(a) = input.get(i) {
                    output.set(i, (self.operation)(a.clone()))?;
                }
            }
        }

        Ok(())
    }

    fn performance_hint(&self) -> PerformanceHint {
        self.performance_hint
    }
}

/// SIMD-optimized binary kernel for floating point operations
pub struct SimdBinaryKernel<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    name: String,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> SimdBinaryKernel<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    /// Create new SIMD-optimized binary kernel
    pub fn new(name: &str, operation: F) -> Self {
        Self {
            name: name.to_string(),
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Kernel for SimdBinaryKernel<T, F>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> KernelSignature {
        KernelSignature::new(
            vec![Dtype::from_type::<T>(), Dtype::from_type::<T>()],
            vec![Dtype::from_type::<T>()],
        )
    }

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
    ) -> Result<()> {
        // For now, fall back to regular implementation
        // In a real implementation, this would use SIMD intrinsics
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                &self.name,
                format!(
                    "Expected 2 inputs, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let input1 = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let input2 = unsafe { &*(inputs[1] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };

        // Use regular implementation for now
        for i in 0..output.size() {
            if let (Some(a), Some(b)) = (input1.get(i), input2.get(i)) {
                output.set(i, (self.operation)(a.clone(), b.clone()))?;
            }
        }

        Ok(())
    }

    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::Vectorized
    }
}

/// Memory-optimized kernel for large arrays
pub struct MemoryOptimizedKernel<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    name: String,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> MemoryOptimizedKernel<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    /// Create new memory-optimized kernel
    pub fn new(name: &str, operation: F) -> Self {
        Self {
            name: name.to_string(),
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Kernel for MemoryOptimizedKernel<T, F>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> KernelSignature {
        KernelSignature::new(
            vec![Dtype::from_type::<T>(), Dtype::from_type::<T>()],
            vec![Dtype::from_type::<T>()],
        )
    }

    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
    ) -> Result<()> {
        // Memory-optimized implementation with chunked processing
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                &self.name,
                format!(
                    "Expected 2 inputs, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let input1 = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let input2 = unsafe { &*(inputs[1] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };

        // Process in chunks to improve cache locality
        const CHUNK_SIZE: usize = 1024;
        let total_size = output.size();

        for chunk_start in (0..total_size).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(total_size);

            for i in chunk_start..chunk_end {
                if let (Some(a), Some(b)) = (input1.get(i), input2.get(i)) {
                    output.set(i, (self.operation)(a.clone(), b.clone()))?;
                }
            }
        }

        Ok(())
    }

    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::MemoryBound
    }
}

/// Convenience macros for creating common kernels

/// Create a binary kernel with automatic type inference
#[macro_export]
macro_rules! binary_kernel {
    ($name:expr, $op:expr) => {
        std::sync::Arc::new(crate::kernel_impls::BinaryKernel::new($name, $op))
    };
    ($name:expr, $op:expr, $hint:expr) => {
        std::sync::Arc::new(
            crate::kernel_impls::BinaryKernel::new($name, $op).with_performance_hint($hint),
        )
    };
}

/// Create a unary kernel with automatic type inference
#[macro_export]
macro_rules! unary_kernel {
    ($name:expr, $op:expr) => {
        std::sync::Arc::new(crate::kernel_impls::UnaryKernel::new($name, $op))
    };
    ($name:expr, $op:expr, $hint:expr) => {
        std::sync::Arc::new(
            crate::kernel_impls::UnaryKernel::new($name, $op).with_performance_hint($hint),
        )
    };
}

/// Create a SIMD-optimized binary kernel
#[macro_export]
macro_rules! simd_binary_kernel {
    ($name:expr, $op:expr) => {
        std::sync::Arc::new(crate::kernel_impls::SimdBinaryKernel::new($name, $op))
    };
}

/// Create a memory-optimized binary kernel
#[macro_export]
macro_rules! memory_optimized_kernel {
    ($name:expr, $op:expr) => {
        std::sync::Arc::new(crate::kernel_impls::MemoryOptimizedKernel::new($name, $op))
    };
}
