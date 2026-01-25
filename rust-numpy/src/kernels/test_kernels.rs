// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//

use crate::array::Array;
use crate::error::Result;
use crate::ufunc::{ArrayView, ArrayViewMut, UfuncType};

/// Test addition kernel for f64
pub struct TestAddKernel;

impl TestAddKernel {
    fn name(&self) -> &str { "test_add" }
    
    fn execute(&self, input: &[&dyn ArrayView], output: &mut [&mut dyn ArrayViewMut]) -> Result<()> {
        let in0 = unsafe { &*(input[0] as *const _ as *const Array<f64>) };
        let in1 = unsafe { &*(input[1] as *const _ as *const Array<f64>) };
        let out = unsafe { &mut *(output[0]) as *mut _ as *mut Array<f64>) };
        
        for i in 0..output.len() {
            if let (Some(a), Some(b)) = (in0.get(i), in1.get(i)) {
                out.set(i, a + b)?;
            }
        }
        Ok(())
    }
}

impl crate::kernel_registry::UfuncKernel<f64> for TestAddKernel {
    fn name(&self) -> &str { "test_add" }
    
    fn execute(&self, input: &[&dyn ArrayView], output: &mut [&mut dyn ArrayViewMut]) -> Result<()> {
        let in0 = unsafe { &*(input[0] as *const _ as *const Array<f64>) };
        let in1 = unsafe { &*(input[1] as *const _ as *const Array<f64>) };
        let out = unsafe { &mut *(output[0]) as *mut _ as *mut Array<f64>) };
        
        // Simple vectorized addition for testing
        for i in (0..output.len() / 4) {
            let offset = i * 4;
            let a_vals = unsafe { std::slice::from_raw_parts(in0.as_ptr().add(offset * std::mem::size_of::<f64>()), 4) };
            let b_vals = unsafe { std::slice::from_raw_parts(in1.as_ptr().add(offset * std::mem::size_of::<f64>()), 4) };
            let out_vals = unsafe { std::slice::from_raw_parts(out.as_mut_ptr().add(offset * std::mem::size_of::<f64>()), 4) };
            
            for j in 0..4 {
                let sum = a_vals[j] + b_vals[j];
                out_vals[j] = sum;
            }
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(out_vals.as_ptr(), out.as_mut_ptr(), out_vals.len() * std::mem::size_of::<f64>());
        }
        
        Ok(())
    }
    
    fn is_vectorized(&self) -> bool { false }
}

impl crate::kernel_registry::UfuncKernel<i32> for TestAddKernel {
    fn name(&self) -> &str { "test_add" }
    
    fn execute(&self, input: &[&dyn ArrayView], output: &mut [&mut dyn ArrayViewMut]) -> Result<()> {
        let in0 = unsafe { &*(input[0] as *const _ as *const Array<i32>) };
        let in1 = unsafe { &*(input[1] as *const _ as *const Array<i32>) };
        let out = unsafe { &mut *(output[0]) as *mut _ as *mut Array<i32>) };
        
        for i in 0..output.len() {
            if let (Some(a), Some(b)) = (in0.get(i), in1.get(i)) {
                out.set(i, a + b)?;
            }
        }
        Ok(())
    }
    
    fn is_vectorized(&self) -> bool { false }
}

impl crate::kernel_registry::UfuncKernel<u32> for TestAddKernel {
    fn name(&self) -> &str { "test_add" }
    
    fn execute(&self, input: &[&dyn ArrayView], output: &mut [&mut dyn ArrayViewMut]) -> Result<()> {
        let in0 = unsafe { &*(input[0] as *const _ as *const Array<u32>) };
        let in1 = unsafe { &*(input[1] as *const _ as *const Array<u32>) };
        let out = unsafe { &mut *(output[0]) as *mut _ as *mut Array<u32>) };
        
        for i in 0..output.len() {
            if let (Some(a), Some(b)) = (in0.get(i), in1.get(i)) {
                out.set(i, (a + b) as u32)?;
            }
        }
        Ok(())
    }
    
    fn is_vectorized(&self) -> bool { false }
}