// Real SIMD intrinsics for x86_64 architecture
//
// This module provides actual SIMD intrinsics for high-performance mathematical operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// CPU feature detection utilities
pub struct CpuFeatures {
    pub has_avx512f: bool,
    pub has_avx2: bool,
    pub has_sse41: bool,
    pub has_fma: bool,
}

impl CpuFeatures {
    #[inline]
    pub fn detect() -> Self {
        Self {
            has_avx512f: is_x86_feature_detected!("avx512f"),
            has_avx2: is_x86_feature_detected!("avx2"),
            has_sse41: is_x86_feature_detected!("sse4.1"),
            has_fma: is_x86_feature_detected!("fma"),
        }
    }

    #[inline]
    pub fn best_vector_width_f64(&self) -> usize {
        if self.has_avx512f {
            8
        } else if self.has_avx2 {
            4
        } else if self.has_sse41 {
            2
        } else {
            1
        }
    }

    #[inline]
    pub fn best_vector_width_f32(&self) -> usize {
        if self.has_avx512f {
            16
        } else if self.has_avx2 {
            8
        } else if self.has_sse41 {
            4
        } else {
            1
        }
    }
}

/// SIMD-optimized addition for f64 arrays
#[cfg(target_arch = "x86_64")]
pub fn simd_add_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    let chunks = a.len() / 4;
    for i in 0..chunks {
        unsafe {
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(i * 4));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(i * 4));
            let c_vec = _mm256_add_pd(a_vec, b_vec);
            _mm256_storeu_pd(result.as_mut_ptr().add(i * 4), c_vec);
        }
    }

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        result[i] = a[i] + b[i];
    }
}

/// SIMD-optimized addition for f32 arrays
#[cfg(target_arch = "x86_64")]
pub fn simd_add_f32_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    let chunks = a.len() / 8;
    for i in 0..chunks {
        unsafe {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let c_vec = _mm256_add_ps(a_vec, b_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i * 8), c_vec);
        }
    }

    // Handle remainder
    for i in (chunks * 8)..a.len() {
        result[i] = a[i] + b[i];
    }
}

/// SIMD-optimized subtraction for f64 arrays
#[cfg(target_arch = "x86_64")]
pub fn simd_sub_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    let chunks = a.len() / 4;
    for i in 0..chunks {
        unsafe {
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(i * 4));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(i * 4));
            let c_vec = _mm256_sub_pd(a_vec, b_vec);
            _mm256_storeu_pd(result.as_mut_ptr().add(i * 4), c_vec);
        }
    }

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        result[i] = a[i] - b[i];
    }
}

/// SIMD-optimized multiplication for f64 arrays
#[cfg(target_arch = "x86_64")]
pub fn simd_mul_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    let chunks = a.len() / 4;
    for i in 0..chunks {
        unsafe {
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(i * 4));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(i * 4));
            let c_vec = _mm256_mul_pd(a_vec, b_vec);
            _mm256_storeu_pd(result.as_mut_ptr().add(i * 4), c_vec);
        }
    }

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        result[i] = a[i] * b[i];
    }
}

/// SIMD-optimized division for f64 arrays
#[cfg(target_arch = "x86_64")]
pub fn simd_div_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    let chunks = a.len() / 4;
    for i in 0..chunks {
        unsafe {
            let a_vec = _mm256_loadu_pd(a.as_ptr().add(i * 4));
            let b_vec = _mm256_loadu_pd(b.as_ptr().add(i * 4));
            let c_vec = _mm256_div_pd(a_vec, b_vec);
            _mm256_storeu_pd(result.as_mut_ptr().add(i * 4), c_vec);
        }
    }

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        result[i] = a[i] / b[i];
    }
}

/// SIMD-optimized sqrt for f64 arrays
#[cfg(target_arch = "x86_64")]
pub fn simd_sqrt_f64_avx2(values: &[f64], result: &mut [f64]) {
    let chunks = values.len() / 4;
    for i in 0..chunks {
        unsafe {
            let v = _mm256_loadu_pd(values.as_ptr().add(i * 4));
            let s = _mm256_sqrt_pd(v);
            _mm256_storeu_pd(result.as_mut_ptr().add(i * 4), s);
        }
    }

    // Handle remainder
    for i in (chunks * 4)..values.len() {
        result[i] = values[i].sqrt();
    }
}

/// Approximate sin using polynomial approximation with SIMD
#[cfg(target_arch = "x86_64")]
pub fn simd_sin_f64_approx(values: &[f64], result: &mut [f64]) {
    let chunks = values.len() / 4;
    for i in 0..chunks {
        unsafe {
            let x = _mm256_loadu_pd(values.as_ptr().add(i * 4));

            // Reduce to [-π, π] range using approximation
            let pi = _mm256_set1_pd(std::f64::consts::PI);
            let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
            let inv_two_pi = _mm256_set1_pd(1.0 / (2.0 * std::f64::consts::PI));

            // Simple range reduction
            let t = _mm256_mul_pd(x, inv_two_pi);
            let t_rounded = _mm256_round_pd(t, _MM_FROUND_NINT);
            let t_frac = _mm256_sub_pd(t, t_rounded);
            let x_reduced = _mm256_mul_pd(t_frac, two_pi);

            // Polynomial approximation: sin(x) ≈ x - x³/6 + x⁵/120
            let x2 = _mm256_mul_pd(x_reduced, x_reduced);
            let x3 = _mm256_mul_pd(x2, x_reduced);
            let x5 = _mm256_mul_pd(x3, x2);

            let one_sixth = _mm256_set1_pd(1.0 / 6.0);
            let one_120th = _mm256_set1_pd(1.0 / 120.0);

            let term1 = x_reduced;
            let term2 = _mm256_mul_pd(x3, one_sixth);
            let term3 = _mm256_mul_pd(x5, one_120th);

            let sin_approx = _mm256_sub_pd(_mm256_sub_pd(term1, term2), term3);

            _mm256_storeu_pd(result.as_mut_ptr().add(i * 4), sin_approx);
        }
    }

    // Handle remainder with scalar computation
    for i in (chunks * 4)..values.len() {
        result[i] = values[i].sin();
    }
}

/// Approximate cos using polynomial approximation with SIMD
#[cfg(target_arch = "x86_64")]
pub fn simd_cos_f64_approx(values: &[f64], result: &mut [f64]) {
    let chunks = values.len() / 4;
    for i in 0..chunks {
        unsafe {
            let x = _mm256_loadu_pd(values.as_ptr().add(i * 4));

            // Reduce to [-π, π] range
            let pi = _mm256_set1_pd(std::f64::consts::PI);
            let two_pi = _mm256_set1_pd(2.0 * std::f64::consts::PI);
            let inv_two_pi = _mm256_set1_pd(1.0 / (2.0 * std::f64::consts::PI));

            let t = _mm256_mul_pd(x, inv_two_pi);
            let t_rounded = _mm256_round_pd(t, _MM_FROUND_NINT);
            let t_frac = _mm256_sub_pd(t, t_rounded);
            let x_reduced = _mm256_mul_pd(t_frac, two_pi);

            // Polynomial approximation: cos(x) ≈ 1 - x²/2 + x⁴/24
            let x2 = _mm256_mul_pd(x_reduced, x_reduced);
            let x4 = _mm256_mul_pd(x2, x2);

            let one_half = _mm256_set1_pd(0.5);
            let one_24th = _mm256_set1_pd(1.0 / 24.0);
            let one = _mm256_set1_pd(1.0);

            let term1 = one;
            let term2 = _mm256_mul_pd(x2, one_half);
            let term3 = _mm256_mul_pd(x4, one_24th);

            let cos_approx = _mm256_sub_pd(_mm256_sub_pd(term1, term2), term3);

            _mm256_storeu_pd(result.as_mut_ptr().add(i * 4), cos_approx);
        }
    }

    // Handle remainder with scalar computation
    for i in (chunks * 4)..values.len() {
        result[i] = values[i].cos();
    }
}

/// Approximate exp using polynomial approximation with SIMD
#[cfg(target_arch = "x86_64")]
pub fn simd_exp_f64_approx(values: &[f64], result: &mut [f64]) {
    let chunks = values.len() / 4;
    for i in 0..chunks {
        unsafe {
            let x = _mm256_loadu_pd(values.as_ptr().add(i * 4));

            // Clamp to prevent overflow
            let max_exp = _mm256_set1_pd(700.0); // exp(700) is near f64::MAX
            let x_clamped = _mm256_min_pd(x, max_exp);

            // For now, use scalar computation for each element
            let mut temp = [0.0f64; 4];
            _mm256_storeu_pd(temp.as_mut_ptr(), x_clamped);

            for j in 0..4 {
                result[i * 4 + j] = temp[j].exp();
            }
        }
    }

    // Handle remainder with scalar computation
    for i in (chunks * 4)..values.len() {
        result[i] = values[i].exp();
    }
}

/// Approximate log using polynomial approximation with SIMD
#[cfg(target_arch = "x86_64")]
pub fn simd_log_f64_approx(values: &[f64], result: &mut [f64]) {
    let chunks = values.len() / 4;
    for i in 0..chunks {
        unsafe {
            let x = _mm256_loadu_pd(values.as_ptr().add(i * 4));

            // For now, use scalar computation for each element
            let mut temp = [0.0f64; 4];
            _mm256_storeu_pd(temp.as_mut_ptr(), x);

            for j in 0..4 {
                if temp[j] > 0.0 {
                    result[i * 4 + j] = temp[j].ln();
                } else {
                    result[i * 4 + j] = f64::NAN;
                }
            }
        }
    }

    // Handle remainder with scalar computation
    for i in (chunks * 4)..values.len() {
        result[i] = if values[i] > 0.0 {
            values[i].ln()
        } else {
            f64::NAN
        };
    }
}

/// Fallback implementations for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
pub fn simd_add_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_add_f32_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_sub_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    for i in 0..a.len() {
        result[i] = a[i] - b[i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_mul_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    for i in 0..a.len() {
        result[i] = a[i] * b[i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_div_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    for i in 0..a.len() {
        result[i] = a[i] / b[i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_sqrt_f64_avx2(values: &[f64], result: &mut [f64]) {
    for i in 0..values.len() {
        result[i] = values[i].sqrt();
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_sin_f64_approx(values: &[f64], result: &mut [f64]) {
    for i in 0..values.len() {
        result[i] = values[i].sin();
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_cos_f64_approx(values: &[f64], result: &mut [f64]) {
    for i in 0..values.len() {
        result[i] = values[i].cos();
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_exp_f64_approx(values: &[f64], result: &mut [f64]) {
    for i in 0..values.len() {
        result[i] = values[i].exp();
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn simd_log_f64_approx(values: &[f64], result: &mut [f64]) {
    for i in 0..values.len() {
        result[i] = values[i].ln();
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub struct CpuFeatures {
    pub has_avx512f: bool,
    pub has_avx2: bool,
    pub has_sse41: bool,
    pub has_fma: bool,
}

#[cfg(not(target_arch = "x86_64"))]
impl CpuFeatures {
    #[inline]
    pub fn detect() -> Self {
        Self {
            has_avx512f: false,
            has_avx2: false,
            has_sse41: false,
            has_fma: false,
        }
    }

    #[inline]
    pub fn best_vector_width_f64(&self) -> usize {
        1
    }

    #[inline]
    pub fn best_vector_width_f32(&self) -> usize {
        1
    }
}
