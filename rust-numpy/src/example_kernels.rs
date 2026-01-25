//! Example kernel implementations for dynamic registry
//!
//! This module provides concrete examples of kernels that can be registered
//! dynamically to demonstrate the dynamic kernel registry system.

use crate::array::Array;
use crate::dtype::{Dtype, DtypeKind};
use crate::dynamic_kernel_registry::{KernelFunction, PerformanceTier};
use crate::error::Result;
use crate::ufunc::{ArrayView, ArrayViewMut};
use std::sync::Arc;

/// Optimized addition kernel for f64
pub struct OptimizedAddF64;

impl KernelFunction for OptimizedAddF64 {
    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        _where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(crate::error::NumPyError::ufunc_error(
                "add",
                format!(
                    "Expected 2 inputs, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let in0 = unsafe { &*(inputs[0] as *const _ as *const Array<f64>) };
        let in1 = unsafe { &*(inputs[1] as *const _ as *const Array<f64>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<f64>) };

        if in0.is_c_contiguous()
            && in1.is_c_contiguous()
            && output.is_c_contiguous()
            && in0.shape() == in1.shape()
            && in0.shape() == output.shape()
        {
            let d0 = in0.data.as_slice();
            let d1 = in1.data.as_slice();
            let out_slice = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, output.size())
            };

            for i in 0..output.size() {
                out_slice[i] = d0[in0.offset + i] + d1[in1.offset + i];
            }
        } else {
            let broadcasted = crate::broadcasting::broadcast_arrays(&[in0, in1])?;
            for i in 0..output.size() {
                if let (Some(a), Some(b)) = (broadcasted[0].get(i), broadcasted[1].get(i)) {
                    output.set(i, a + b)?;
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "add"
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[DtypeKind::Float]
    }

    fn performance_tier(&self) -> PerformanceTier {
        PerformanceTier::Vectorized
    }
}

/// Generic addition kernel for fallback
pub struct GenericAdd<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> GenericAdd<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> KernelFunction for GenericAdd<T>
where
    T: Clone + Default + std::ops::Add<Output = T> + Send + Sync + 'static,
{
    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        _where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(crate::error::NumPyError::ufunc_error(
                "add",
                format!(
                    "Expected 2 inputs, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let in0 = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let in1 = unsafe { &*(inputs[1] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };

        let broadcasted = crate::broadcasting::broadcast_arrays(&[in0, in1])?;
        for i in 0..output.size() {
            if let (Some(a), Some(b)) = (broadcasted[0].get(i), broadcasted[1].get(i)) {
                output.set(i, a.clone() + b.clone())?;
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "add"
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[
            DtypeKind::Integer,
            DtypeKind::Unsigned,
            DtypeKind::Float,
            DtypeKind::Complex,
        ]
    }

    fn performance_tier(&self) -> PerformanceTier {
        PerformanceTier::Generic
    }
}

/// Hardware-accelerated multiplication kernel (placeholder)
pub struct HardwareMultiply;

impl KernelFunction for HardwareMultiply {
    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        _where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(crate::error::NumPyError::ufunc_error(
                "multiply",
                format!(
                    "Expected 2 inputs, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let in0 = unsafe { &*(inputs[0] as *const _ as *const Array<f64>) };
        let in1 = unsafe { &*(inputs[1] as *const _ as *const Array<f64>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<f64>) };

        let d0 = in0.data.as_slice();
        let d1 = in1.data.as_slice();
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, output.size())
        };

        for i in 0..output.size() {
            out_slice[i] = d0[in0.offset + i] * d1[in1.offset + i];
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "multiply"
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[DtypeKind::Float]
    }

    fn performance_tier(&self) -> PerformanceTier {
        PerformanceTier::HardwareAccelerated
    }
}

/// Factory function to create example kernels
pub fn create_example_kernels() -> Vec<(Box<dyn KernelFunction>, Vec<Dtype>)> {
    let mut kernels = Vec::new();

    kernels.push((
        Box::new(OptimizedAddF64),
        vec![Dtype::from_type::<f64>(), Dtype::from_type::<f64>()],
    ));

    kernels.push((
        Box::new(GenericAdd::<f32>::new()),
        vec![Dtype::from_type::<f32>(), Dtype::from_type::<f32>()],
    ));

    kernels.push((
        Box::new(GenericAdd::<i64>::new()),
        vec![Dtype::from_type::<i64>(), Dtype::from_type::<i64>()],
    ));

    kernels.push((
        Box::new(GenericAdd::<i32>::new()),
        vec![Dtype::from_type::<i32>(), Dtype::from_type::<i32>()],
    ));

    kernels.push((
        Box::new(HardwareMultiply),
        vec![Dtype::from_type::<f64>(), Dtype::from_type::<f64>()],
    ));

    kernels
}

/// Register all example kernels with the dynamic registry
pub fn register_example_kernels() -> Result<()> {
    use crate::dynamic_kernel_registry::register_kernel;

    let kernels = create_example_kernels();

    for (kernel, dtypes) in kernels {
        register_kernel(kernel.name(), Box::new(kernel), dtypes)?;
    }

    Ok(())
}
