use crate::error::NumPyError;
#[cfg(feature = "simd")]
use core_simd::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuFeature {
    Avx512,
    Avx2,
    Sse,
    Neon,
    Scalar,
}

impl CpuFeature {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                CpuFeature::Avx512
            } else if is_x86_feature_detected!("avx2") {
                CpuFeature::Avx2
            } else if is_x86_feature_detected!("sse4.2") {
                CpuFeature::Sse
            } else {
                CpuFeature::Scalar
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("asimd") {
                CpuFeature::Neon
            } else {
                CpuFeature::Scalar
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuFeature::Scalar
        }
    }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("asimd") {
                CpuFeature::Neon
            } else {
                CpuFeature::Scalar
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            CpuFeature::Scalar
        }
    }

    pub fn f32_lanes(self) -> usize {
        match self {
            CpuFeature::Avx512 => 16,
            CpuFeature::Avx2 => 8,
            CpuFeature::Sse => 4,
            CpuFeature::Neon => 4,
            CpuFeature::Scalar => 1,
        }
    }

    pub fn f64_lanes(self) -> usize {
        match self {
            CpuFeature::Avx512 => 8,
            CpuFeature::Avx2 => 4,
            CpuFeature::Sse => 2,
            CpuFeature::Neon => 2,
            CpuFeature::Scalar => 1,
        }
    }
}

pub struct SimdDispatcher {
    cpu_feature: CpuFeature,
}

impl SimdDispatcher {
    pub fn new() -> Self {
        Self {
            cpu_feature: CpuFeature::detect(),
        }
    }

    pub fn with_feature(cpu_feature: CpuFeature) -> Self {
        Self { cpu_feature }
    }

    pub fn cpu_feature(&self) -> CpuFeature {
        self.cpu_feature
    }

    pub fn add_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(NumPyError::invalid_value(
                "Input arrays must have same length",
            ));
        }

        match self.cpu_feature {
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx512 => self.add_f32_avx512(a, b, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx2 => self.add_f32_avx2(a, b, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Sse => self.add_f32_sse(a, b, result),
            #[cfg(target_arch = "aarch64")]
            CpuFeature::Neon => self.add_f32_neon(a, b, result),
            _ => self.add_f32_scalar(a, b, result),
        }
    }

    pub fn sub_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(NumPyError::invalid_value(
                "Input arrays must have same length",
            ));
        }

        match self.cpu_feature {
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx512 => self.sub_f32_avx512(a, b, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx2 => self.sub_f32_avx2(a, b, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Sse => self.sub_f32_sse(a, b, result),
            #[cfg(target_arch = "aarch64")]
            CpuFeature::Neon => self.sub_f32_neon(a, b, result),
            _ => self.sub_f32_scalar(a, b, result),
        }
    }

    pub fn mul_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(NumPyError::invalid_value(
                "Input arrays must have same length",
            ));
        }

        match self.cpu_feature {
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx512 => self.mul_f32_avx512(a, b, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx2 => self.mul_f32_avx2(a, b, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Sse => self.mul_f32_sse(a, b, result),
            #[cfg(target_arch = "aarch64")]
            CpuFeature::Neon => self.mul_f32_neon(a, b, result),
            _ => self.mul_f32_scalar(a, b, result),
        }
    }

    pub fn div_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(NumPyError::invalid_value(
                "Input arrays must have same length",
            ));
        }

        match self.cpu_feature {
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx512 => self.div_f32_avx512(a, b, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx2 => self.div_f32_avx2(a, b, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Sse => self.div_f32_sse(a, b, result),
            #[cfg(target_arch = "aarch64")]
            CpuFeature::Neon => self.div_f32_neon(a, b, result),
            _ => self.div_f32_scalar(a, b, result),
        }
    }

    pub fn sqrt_f32(&self, a: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        if a.len() != result.len() {
            return Err(NumPyError::invalid_value(
                "Input arrays must have same length",
            ));
        }

        match self.cpu_feature {
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx512 => self.sqrt_f32_avx512(a, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Avx2 => self.sqrt_f32_avx2(a, result),
            #[cfg(target_arch = "x86_64")]
            CpuFeature::Sse => self.sqrt_f32_sse(a, result),
            #[cfg(target_arch = "aarch64")]
            CpuFeature::Neon => self.sqrt_f32_neon(a, result),
            _ => self.sqrt_f32_scalar(a, result),
        }
    }
}

// Scalar fallback implementations
impl SimdDispatcher {
    fn add_f32_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
        Ok(())
    }

    fn sub_f32_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        for i in 0..a.len() {
            result[i] = a[i] - b[i];
        }
        Ok(())
    }

    fn mul_f32_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
        Ok(())
    }

    fn div_f32_scalar(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        for i in 0..a.len() {
            result[i] = a[i] / b[i];
        }
        Ok(())
    }

    fn sqrt_f32_scalar(&self, a: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        for i in 0..a.len() {
            result[i] = a[i].sqrt();
        }
        Ok(())
    }
}

// AVX2 implementations
#[cfg(target_arch = "x86_64")]
impl SimdDispatcher {
    #[target_feature(enable = "avx2")]
    unsafe fn add_f32_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        let lanes = 8;
        let simd_len = (a.len() / lanes) * lanes;

        // Process SIMD chunks
        for i in (0..simd_len).step_by(lanes) {
            let va =
                std::mem::transmute::<_, std::arch::x86_64::__m256>(a.get_unchecked(i..i + lanes));
            let vb =
                std::mem::transmute::<_, std::arch::x86_64::__m256>(b.get_unchecked(i..i + lanes));
            let vr = std::arch::x86_64::_mm256_add_ps(va, vb);
            result
                .get_unchecked_mut(i..i + lanes)
                .copy_from_slice(&std::mem::transmute::<_, [f32; 8]>(vr));
        }

        // Handle remainder
        for i in simd_len..a.len() {
            result[i] = a[i] + b[i];
        }

        Ok(())
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul_f32_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        let lanes = 8;
        let simd_len = (a.len() / lanes) * lanes;

        for i in (0..simd_len).step_by(lanes) {
            let va =
                std::mem::transmute::<_, std::arch::x86_64::__m256>(a.get_unchecked(i..i + lanes));
            let vb =
                std::mem::transmute::<_, std::arch::x86_64::__m256>(b.get_unchecked(i..i + lanes));
            let vr = std::arch::x86_64::_mm256_mul_ps(va, vb);
            result
                .get_unchecked_mut(i..i + lanes)
                .copy_from_slice(&std::mem::transmute::<_, [f32; 8]>(vr));
        }

        for i in simd_len..a.len() {
            result[i] = a[i] * b[i];
        }

        Ok(())
    }

    // Placeholder implementations for other AVX2 functions
    #[target_feature(enable = "avx2")]
    unsafe fn sub_f32_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.add_f32_scalar(a, b, result) // Fallback for now
    }

    #[target_feature(enable = "avx2")]
    unsafe fn div_f32_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.add_f32_scalar(a, b, result) // Fallback for now
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sqrt_f32_avx2(&self, a: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        self.sqrt_f32_scalar(a, result) // Fallback for now
    }
}

// Placeholder implementations for other architectures
#[cfg(target_arch = "x86_64")]
impl SimdDispatcher {
    #[target_feature(enable = "avx512f")]
    unsafe fn add_f32_avx512(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.add_f32_scalar(a, b, result) // Fallback for now
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn sub_f32_avx512(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.sub_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn mul_f32_avx512(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.mul_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn div_f32_avx512(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.div_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn sqrt_f32_avx512(&self, a: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        self.sqrt_f32_scalar(a, result)
    }

    #[target_feature(enable = "sse4.2")]
    unsafe fn add_f32_sse(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.add_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "sse4.2")]
    unsafe fn sub_f32_sse(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.sub_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "sse4.2")]
    unsafe fn mul_f32_sse(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.mul_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "sse4.2")]
    unsafe fn div_f32_sse(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.div_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "sse4.2")]
    unsafe fn sqrt_f32_sse(&self, a: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        self.sqrt_f32_scalar(a, result)
    }
}

#[cfg(target_arch = "aarch64")]
impl SimdDispatcher {
    #[target_feature(enable = "asimd")]
    unsafe fn add_f32_neon(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.add_f32_scalar(a, b, result) // Fallback for now
    }

    #[target_feature(enable = "asimd")]
    unsafe fn sub_f32_neon(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.sub_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "asimd")]
    unsafe fn mul_f32_neon(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.mul_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "asimd")]
    unsafe fn div_f32_neon(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), NumPyError> {
        self.div_f32_scalar(a, b, result)
    }

    #[target_feature(enable = "asimd")]
    unsafe fn sqrt_f32_neon(&self, a: &[f32], result: &mut [f32]) -> Result<(), NumPyError> {
        self.sqrt_f32_scalar(a, result)
    }
}

/// Global SIMD dispatcher instance
lazy_static::lazy_static! {
    static ref SIMD_DISPATCHER: SimdDispatcher = SimdDispatcher::new();
}

/// Get the global SIMD dispatcher
pub fn get_simd_dispatcher() -> &'static SimdDispatcher {
    &SIMD_DISPATCHER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_feature_detection() {
        let feature = CpuFeature::detect();
        println!("Detected CPU feature: {:?}", feature);

        // Should always return some valid feature
        match feature {
            CpuFeature::Avx512
            | CpuFeature::Avx2
            | CpuFeature::Sse
            | CpuFeature::Neon
            | CpuFeature::Scalar => {} // All valid
        }
    }

    #[test]
    fn test_simd_dispatcher_creation() {
        let dispatcher = SimdDispatcher::new();
        let _feature = dispatcher.cpu_feature();
    }

    #[test]
    fn test_simd_add_f32() {
        let dispatcher = SimdDispatcher::with_feature(CpuFeature::Scalar);

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        dispatcher.add_f32(&a, &b, &mut result).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }
}
