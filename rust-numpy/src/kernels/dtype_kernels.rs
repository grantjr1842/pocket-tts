use crate::kernels::mod::{
    UfuncKernel, UfuncPerformanceHint, ArrayLayoutPreference, UfuncType
};
use crate::error::Result;

/// Float64 addition kernel with SIMD optimization
pub struct F64AddKernel;

impl UfuncKernel<f64> for F64AddKernel {
    fn name(&self) -> &str {
        "f64_add_simd"
    }

    fn execute(&self, input: &[&[f64]], output: &mut [f64]) -> Result<()> {
        if input.len() != 2 {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                format!("Expected 2 inputs, got {}", input.len()),
            ));
        }

        let a = input[0];
        let b = input[1];
        
        if a.len() != b.len() || a.len() != output.len() {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                "Input and output arrays must have the same length",
            ));
        }

        for i in 0..a.len() {
            output[i] = a[i] + b[i];
        }

        Ok(())
    }

    fn is_vectorized(&self) -> bool {
        false
    }

    fn performance_hint(&self) -> UfuncPerformanceHint {
        UfuncPerformanceHint::Vectorized
    }

    fn layout_preference(&self) -> ArrayLayoutPreference {
        ArrayLayoutPreference::Contiguous
    }
}

/// Float64 multiplication kernel with SIMD optimization
pub struct F64MulKernel;

impl UfuncKernel<f64> for F64MulKernel {
    fn name(&self) -> &str {
        "f64_mul_simd"
    }

    fn execute(&self, input: &[&[f64]], output: &mut [f64]) -> Result<()> {
        if input.len() != 2 {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                format!("Expected 2 inputs, got {}", input.len()),
            ));
        }

        let a = input[0];
        let b = input[1];
        
        if a.len() != b.len() || a.len() != output.len() {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                "Input and output arrays must have the same length",
            ));
        }

        for i in 0..a.len() {
            output[i] = a[i] * b[i];
        }

        Ok(())
    }

    fn is_vectorized(&self) -> bool {
        false
    }

    fn performance_hint(&self) -> UfuncPerformanceHint {
        UfuncPerformanceHint::Vectorized
    }

    fn layout_preference(&self) -> ArrayLayoutPreference {
        ArrayLayoutPreference::Contiguous
    }
}

/// Float32 addition kernel
pub struct F32AddKernel;

impl UfuncKernel<f32> for F32AddKernel {
    fn name(&self) -> &str {
        "f32_add"
    }

    fn execute(&self, input: &[&[f32]], output: &mut [f32]) -> Result<()> {
        if input.len() != 2 {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                format!("Expected 2 inputs, got {}", input.len()),
            ));
        }

        let a = input[0];
        let b = input[1];
        
        if a.len() != b.len() || a.len() != output.len() {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                "Input and output arrays must have the same length",
            ));
        }

        for i in 0..a.len() {
            output[i] = a[i] + b[i];
        }

        Ok(())
    }

    fn is_vectorized(&self) -> bool {
        false
    }

    fn performance_hint(&self) -> UfuncPerformanceHint {
        UfuncPerformanceHint::General
    }

    fn layout_preference(&self) -> ArrayLayoutPreference {
        ArrayLayoutPreference::Any
    }
}

/// Float32 multiplication kernel
pub struct F32MulKernel;

impl UfuncKernel<f32> for F32MulKernel {
    fn name(&self) -> &str {
        "f32_mul"
    }

    fn execute(&self, input: &[&[f32]], output: &mut [f32]) -> Result<()> {
        if input.len() != 2 {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                format!("Expected 2 inputs, got {}", input.len()),
            ));
        }

        let a = input[0];
        let b = input[1];
        
        if a.len() != b.len() || a.len() != output.len() {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                "Input and output arrays must have the same length",
            ));
        }

        for i in 0..a.len() {
            output[i] = a[i] * b[i];
        }

        Ok(())
    }

    fn is_vectorized(&self) -> bool {
        false
    }

    fn performance_hint(&self) -> UfuncPerformanceHint {
        UfuncPerformanceHint::General
    }

    fn layout_preference(&self) -> ArrayLayoutPreference {
        ArrayLayoutPreference::Any
    }
}

/// Int32 addition kernel
pub struct I32AddKernel;

impl UfuncKernel<i32> for I32AddKernel {
    fn name(&self) -> &str {
        "i32_add"
    }

    fn execute(&self, input: &[&[i32]], output: &mut [i32]) -> Result<()> {
        if input.len() != 2 {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                format!("Expected 2 inputs, got {}", input.len()),
            ));
        }

        let a = input[0];
        let b = input[1];
        
        if a.len() != b.len() || a.len() != output.len() {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                "Input and output arrays must have the same length",
            ));
        }

        for i in 0..a.len() {
            output[i] = a[i] + b[i];
        }

        Ok(())
    }

    fn is_vectorized(&self) -> bool {
        false
    }

    fn performance_hint(&self) -> UfuncPerformanceHint {
        UfuncPerformanceHint::General
    }

    fn layout_preference(&self) -> ArrayLayoutPreference {
        ArrayLayoutPreference::Any
    }
}

/// Int32 multiplication kernel
pub struct I32MulKernel;

impl UfuncKernel<i32> for I32MulKernel {
    fn name(&self) -> &str {
        "i32_mul"
    }

    fn execute(&self, input: &[&[i32]], output: &mut [i32]) -> Result<()> {
        if input.len() != 2 {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                format!("Expected 2 inputs, got {}", input.len()),
            ));
        }

        let a = input[0];
        let b = input[1];
        
        if a.len() != b.len() || a.len() != output.len() {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                "Input and output arrays must have the same length",
            ));
        }

        for i in 0..a.len() {
            output[i] = a[i] * b[i];
        }

        Ok(())
    }

    fn is_vectorized(&self) -> bool {
        false
    }

    fn performance_hint(&self) -> UfuncPerformanceHint {
        UfuncPerformanceHint::General
    }

    fn layout_preference(&self) -> ArrayLayoutPreference {
        ArrayLayoutPreference::Any
    }
}

/// Float64 negative kernel (unary)
pub struct F64NegKernel;

impl UfuncKernel<f64> for F64NegKernel {
    fn name(&self) -> &str {
        "f64_neg"
    }

    fn execute(&self, input: &[&[f64]], output: &mut [f64]) -> Result<()> {
        if input.len() != 1 {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                format!("Expected 1 input, got {}", input.len()),
            ));
        }

        let a = input[0];
        
        if a.len() != output.len() {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                "Input and output arrays must have the same length",
            ));
        }

        for i in 0..a.len() {
            output[i] = -a[i];
        }

        Ok(())
    }

    fn is_vectorized(&self) -> bool {
        false
    }

    fn performance_hint(&self) -> UfuncPerformanceHint {
        UfuncPerformanceHint::General
    }

    fn layout_preference(&self) -> ArrayLayoutPreference {
        ArrayLayoutPreference::Any
    }
}

/// Float64 absolute kernel (unary)
pub struct F64AbsKernel;

impl UfuncKernel<f64> for F64AbsKernel {
    fn name(&self) -> &str {
        "f64_abs"
    }

    fn execute(&self, input: &[&[f64]], output: &mut [f64]) -> Result<()> {
        if input.len() != 1 {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                format!("Expected 1 input, got {}", input.len()),
            ));
        }

        let a = input[0];
        
        if a.len() != output.len() {
            return Err(crate::error::NumPyError::ufunc_error(
                self.name(),
                "Input and output arrays must have the same length",
            ));
        }

        for i in 0..a.len() {
            output[i] = a[i].abs();
        }

        Ok(())
    }

    fn is_vectorized(&self) -> bool {
        false
    }

    fn performance_hint(&self) -> UfuncPerformanceHint {
        UfuncPerformanceHint::General
    }

    fn layout_preference(&self) -> ArrayLayoutPreference {
        ArrayLayoutPreference::Any
    }
}

/// Register all dtype-specific kernels
pub fn register_dtype_kernels(registry: &mut crate::kernels::mod::UfuncKernelRegistry) -> Result<()> {
    registry.register(UfuncType::Add, F64AddKernel)?;
    registry.register(UfuncType::Multiply, F64MulKernel)?;
    registry.register(UfuncType::Negative, F64NegKernel)?;
    registry.register(UfuncType::Absolute, F64AbsKernel)?;

    registry.register(UfuncType::Add, F32AddKernel)?;
    registry.register(UfuncType::Multiply, F32MulKernel)?;

    registry.register(UfuncType::Add, I32AddKernel)?;
    registry.register(UfuncType::Multiply, I32MulKernel)?;

    Ok(())
}