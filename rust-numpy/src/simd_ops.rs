// SIMD-optimized operations for mathematical ufuncs
//
// This module provides SIMD-optimized implementations of common mathematical
// operations using runtime CPU feature detection for optimal performance.

#[cfg(feature = "simd")]
use crate::cpu_features::{get_cpu_features, CpuFeatures};

/// SIMD chunk size for different architectures
#[cfg(feature = "simd")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdChunkSize {
    /// 256-bit vectors (AVX2): 4x f64, 8x f32
    Avx2,
    /// 512-bit vectors (AVX-512): 8x f64, 16x f32
    Avx512,
    /// 128-bit vectors (NEON/SSE): 2x f64, 4x f32
    Sse128,
    /// Scalar fallback
    Scalar,
}

#[cfg(feature = "simd")]
impl SimdChunkSize {
    /// Create new chunk size based on runtime CPU feature detection
    #[inline]
    pub fn new() -> Self {
        let features = crate::simd_intrinsics::CpuFeatures::detect();
        if features.has_avx512f {
            SimdChunkSize::Avx512
        } else if features.has_avx2 {
            SimdChunkSize::Avx2
        } else if features.has_sse41 {
            SimdChunkSize::Sse128
        } else {
            SimdChunkSize::Scalar
        }
    }

    #[inline]
    pub fn chunk_size_f64(&self) -> usize {
        match self {
            SimdChunkSize::Avx2 => 4,
            SimdChunkSize::Avx512 => 8,
            SimdChunkSize::Sse128 => 2,
            SimdChunkSize::Scalar => 1,
        }
    }

    #[inline]
    pub fn chunk_size_f32(&self) -> usize {
        match self {
            SimdChunkSize::Avx2 => 8,
            SimdChunkSize::Avx512 => 16,
            SimdChunkSize::Sse128 => 4,
            SimdChunkSize::Scalar => 1,
        }
    }
}

/// Process array using SIMD-optimized sin function
#[cfg(feature = "simd")]
pub fn simd_sin_f64(values: &[f64]) -> Vec<f64> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; values.len()];
        crate::simd_intrinsics::simd_sin_f64_approx(values, &mut result);
        result
    } else {
        // Fallback to scalar
        values.iter().copied().map(|x| x.sin()).collect()
    }
}

/// Process array using SIMD-optimized cos function
#[cfg(feature = "simd")]
pub fn simd_cos_f64(values: &[f64]) -> Vec<f64> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; values.len()];
        crate::simd_intrinsics::simd_cos_f64_approx(values, &mut result);
        result
    } else {
        // Fallback to scalar
        values.iter().copied().map(|x| x.cos()).collect()
    }
}

/// Process array using SIMD-optimized exp function
#[cfg(feature = "simd")]
pub fn simd_exp_f64(values: &[f64]) -> Vec<f64> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; values.len()];
        crate::simd_intrinsics::simd_exp_f64_approx(values, &mut result);
        result
    } else {
        // Fallback to scalar
        values.iter().copied().map(|x| x.exp()).collect()
    }
}

/// Process array using SIMD-optimized log function
#[cfg(feature = "simd")]
pub fn simd_log_f64(values: &[f64]) -> Vec<f64> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; values.len()];
        crate::simd_intrinsics::simd_log_f64_approx(values, &mut result);
        result
    } else {
        // Fallback to scalar
        values.iter().copied().map(|x| x.ln()).collect()
    }
}

/// Process array using SIMD-optimized sqrt function
#[cfg(feature = "simd")]
pub fn simd_sqrt_f64(values: &[f64]) -> Vec<f64> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; values.len()];
        crate::simd_intrinsics::simd_sqrt_f64_avx2(values, &mut result);
        result
    } else if features.has_sse41 {
        let mut result = vec![0.0; values.len()];
        for i in 0..values.len() {
            result[i] = values[i].sqrt();
        }
        result
    } else {
        // Fallback to scalar
        values.iter().copied().map(|x| x.sqrt()).collect()
    }
}

/// Process array using SIMD-optimized f64 addition
#[cfg(feature = "simd")]
pub fn simd_add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; a.len()];
        crate::simd_intrinsics::simd_add_f64_avx2(a, b, &mut result);
        result
    } else {
        // Fallback to scalar
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }
}

/// Process array using SIMD-optimized f32 addition
#[cfg(feature = "simd")]
pub fn simd_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; a.len()];
        crate::simd_intrinsics::simd_add_f32_avx2(a, b, &mut result);
        result
    } else {
        // Fallback to scalar
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }
}

/// Process array using SIMD-optimized f64 subtraction
#[cfg(feature = "simd")]
pub fn simd_sub_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; a.len()];
        crate::simd_intrinsics::simd_sub_f64_avx2(a, b, &mut result);
        result
    } else {
        // Fallback to scalar
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }
}

/// Process array using SIMD-optimized f64 multiplication
#[cfg(feature = "simd")]
pub fn simd_mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; a.len()];
        crate::simd_intrinsics::simd_mul_f64_avx2(a, b, &mut result);
        result
    } else {
        // Fallback to scalar
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }
}

/// Process array using SIMD-optimized f64 division
#[cfg(feature = "simd")]
pub fn simd_div_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    let features = crate::simd_intrinsics::CpuFeatures::detect();
    if features.has_avx2 {
        let mut result = vec![0.0; a.len()];
        crate::simd_intrinsics::simd_div_f64_avx2(a, b, &mut result);
        result
    } else {
        // Fallback to scalar
        a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
    }
}

/// Scalar fallback for architectures without SIMD support
#[cfg(not(feature = "simd"))]
pub fn simd_sin_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.sin()).collect()
}

#[cfg(not(feature = "simd"))]
pub fn simd_cos_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.cos()).collect()
}

#[cfg(not(feature = "simd"))]
pub fn simd_exp_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.exp()).collect()
}

#[cfg(not(feature = "simd"))]
pub fn simd_log_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.ln()).collect()
}

#[cfg(not(feature = "simd"))]
pub fn simd_sqrt_f64(values: &[f64]) -> Vec<f64> {
    values.iter().copied().map(|x| x.sqrt()).collect()
}

#[cfg(not(feature = "simd"))]
pub fn simd_add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[cfg(not(feature = "simd"))]
pub fn simd_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[cfg(not(feature = "simd"))]
pub fn simd_sub_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

#[cfg(not(feature = "simd"))]
pub fn simd_mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

#[cfg(not(feature = "simd"))]
pub fn simd_div_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x / y).collect()
}

/// Benchmark helper to compare SIMD vs scalar performance
#[cfg(feature = "simd")]
pub fn benchmark_simd_vs_scalar_f64(
    name: &str,
    scalar_fn: impl Fn(&[f64]) -> Vec<f64>,
    simd_fn: impl Fn(&[f64]) -> Vec<f64>,
) {
    use std::time::Instant;

    let test_data: Vec<f64> = (0..10000).map(|i| i as f64).collect();

    let start = Instant::now();
    let _scalar_result = scalar_fn(&test_data);
    let scalar_time = start.elapsed();

    let start = Instant::now();
    let _simd_result = simd_fn(&test_data);
    let simd_time = start.elapsed();

    println!("Benchmark: {}", name);
    println!("  Scalar: {:?}", scalar_time);
    println!("  SIMD:   {:?}", simd_time);
    println!(
        "  Speedup: {:.2}x",
        scalar_time.as_secs_f64() / simd_time.as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_sin() {
        let input = vec![
            0.0f64,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 2.0,
        ];
        let result = simd_sin_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 0.0_f64.sin()).abs() < 1e-10);
        assert!((result[1] - (std::f64::consts::PI / 4.0).sin()).abs() < 1e-10);
        assert!((result[2] - (std::f64::consts::PI / 2.0).sin()).abs() < 1e-10);
    }

    #[test]
    fn test_simd_cos() {
        let input = vec![0.0f64, std::f64::consts::PI / 3.0, std::f64::consts::PI];
        let result = simd_cos_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 0.0_f64.cos()).abs() < 1e-10);
        assert!((result[1] - (std::f64::consts::PI / 3.0).cos()).abs() < 1e-10);
    }

    #[test]
    fn test_simd_exp() {
        let input = vec![0.0f64, 1.0, 2.0];
        let result = simd_exp_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 0.0_f64.exp()).abs() < 1e-10);
        assert!((result[1] - 1.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_simd_log() {
        let input = vec![1.0f64, std::f64::consts::E, 10.0];
        let result = simd_log_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 1.0_f64.ln()).abs() < 1e-10);
        assert!((result[1] - std::f64::consts::E.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_simd_sqrt() {
        let input = vec![0.0f64, 1.0, 4.0, 9.0, 16.0];
        let result = simd_sqrt_f64(&input);

        assert_eq!(result.len(), input.len());
        assert!((result[0] - 0.0_f64.sqrt()).abs() < 1e-10);
        assert!((result[2] - 4.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_simd_add() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![5.0f64, 6.0, 7.0, 8.0];
        let result = simd_add_f64(&a, &b);

        assert_eq!(result.len(), a.len());
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_simd_sub() {
        let a = vec![5.0f64, 6.0, 7.0, 8.0];
        let b = vec![1.0f64, 2.0, 3.0, 4.0];
        let result = simd_sub_f64(&a, &b);

        assert_eq!(result.len(), a.len());
        assert_eq!(result, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_cpu_features() {
        let features = crate::cpu_features::CpuFeatures::detect();
        // Should not panic
        let _width_f64 = features.best_vector_width_f64();
        let _width_f32 = features.best_vector_width_f32();
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_chunk_size() {
        let chunk_size = SimdChunkSize::new();
        let _size_f64 = chunk_size.chunk_size_f64();
        let _size_f32 = chunk_size.chunk_size_f32();
    }
}
