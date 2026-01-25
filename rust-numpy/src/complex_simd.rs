//! SIMD-optimized operations for complex numbers
//!
//! This module provides high-performance SIMD operations for complex number arithmetic,
//! including complex64 (f32 pairs) and complex128 (f64 pairs) using architecture-specific
//! intrinsics.

use num_complex::{Complex32, Complex64};
use std::arch::x86_64::*;

/// SIMD-optimized addition for complex64 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn complex_add_avx2_f32(a: &[Complex32], b: &[Complex32]) -> Vec<Complex32> {
    assert_eq!(a.len(), b.len());

    // Process 8 complex numbers (32 floats) at a time with AVX2
    let chunk_size = 8;
    let result_len = a.len();
    let mut result: Vec<Complex32> = Vec::with_capacity(result_len);

    let mut i = 0;
    unsafe {
        while i + chunk_size <= result_len {
            // Load complex numbers as interleaved real/imaginary
            let a_real = _mm256_loadu_ps(a.as_ptr().add(i) as *const f32);
            let a_imag = _mm256_loadu_ps(a.as_ptr().add(i).add(chunk_size) as *const f32);
            let b_real = _mm256_loadu_ps(b.as_ptr().add(i) as *const f32);
            let b_imag = _mm256_loadu_ps(b.as_ptr().add(i).add(chunk_size) as *const f32);

            // Add real and imaginary parts
            let res_real = _mm256_add_ps(a_real, b_real);
            let res_imag = _mm256_add_ps(a_imag, b_imag);

            // Store results
            let result_ptr = result.as_mut_ptr().add(i) as *mut f32;
            _mm256_storeu_ps(result_ptr, res_real);
            _mm256_storeu_ps(result_ptr.add(chunk_size), res_imag);

            i += chunk_size;
        }
    }

    // Handle remaining elements
    while i < result_len {
        result.push(a[i] + b[i]);
        i += 1;
    }

    result
}

/// SIMD-optimized multiplication for complex64 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn complex_mul_avx2_f32(a: &[Complex32], b: &[Complex32]) -> Vec<Complex32> {
    assert_eq!(a.len(), b.len());

    let chunk_size = 8;
    let result_len = a.len();
    let mut result: Vec<Complex32> = Vec::with_capacity(result_len);

    let mut i = 0;
    unsafe {
        while i + chunk_size <= result_len {
            // Load complex numbers
            let a_real = _mm256_loadu_ps(a.as_ptr().add(i) as *const f32);
            let a_imag = _mm256_loadu_ps(a.as_ptr().add(i).add(chunk_size) as *const f32);
            let b_real = _mm256_loadu_ps(b.as_ptr().add(i) as *const f32);
            let b_imag = _mm256_loadu_ps(b.as_ptr().add(i).add(chunk_size) as *const f32);

            // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
            let ac = _mm256_mul_ps(a_real, b_real);
            let bd = _mm256_mul_ps(a_imag, b_imag);
            let ad = _mm256_mul_ps(a_real, b_imag);
            let bc = _mm256_mul_ps(a_imag, b_real);

            let res_real = _mm256_sub_ps(ac, bd);
            let res_imag = _mm256_add_ps(ad, bc);

            // Store results
            let result_ptr = result.as_mut_ptr().add(i) as *mut f32;
            _mm256_storeu_ps(result_ptr, res_real);
            _mm256_storeu_ps(result_ptr.add(chunk_size), res_imag);

            i += chunk_size;
        }
    }

    // Handle remaining elements
    while i < result_len {
        result.push(a[i] * b[i]);
        i += 1;
    }

    result
}

/// SIMD-optimized addition for complex128 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn complex_add_avx2_f64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(a.len(), b.len());

    // Process 4 complex numbers (8 doubles) at a time with AVX2
    let chunk_size = 4;
    let result_len = a.len();
    let mut result: Vec<Complex64> = Vec::with_capacity(result_len);

    let mut i = 0;
    unsafe {
        while i + chunk_size <= result_len {
            // Load complex numbers as interleaved real/imaginary
            let a_real = _mm256_loadu_pd(a.as_ptr().add(i) as *const f64);
            let a_imag = _mm256_loadu_pd(a.as_ptr().add(i).add(chunk_size) as *const f64);
            let b_real = _mm256_loadu_pd(b.as_ptr().add(i) as *const f64);
            let b_imag = _mm256_loadu_pd(b.as_ptr().add(i).add(chunk_size) as *const f64);

            // Add real and imaginary parts
            let res_real = _mm256_add_pd(a_real, b_real);
            let res_imag = _mm256_add_pd(a_imag, b_imag);

            // Store results
            let result_ptr = result.as_mut_ptr().add(i) as *mut f64;
            _mm256_storeu_pd(result_ptr, res_real);
            _mm256_storeu_pd(result_ptr.add(chunk_size), res_imag);

            i += chunk_size;
        }
    }

    // Handle remaining elements
    while i < result_len {
        result.push(a[i] + b[i]);
        i += 1;
    }

    result
}

/// SIMD-optimized multiplication for complex128 arrays
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn complex_mul_avx2_f64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(a.len(), b.len());

    let chunk_size = 4;
    let result_len = a.len();
    let mut result: Vec<Complex64> = Vec::with_capacity(result_len);

    let mut i = 0;
    unsafe {
        while i + chunk_size <= result_len {
            // Load complex numbers
            let a_real = _mm256_loadu_pd(a.as_ptr().add(i) as *const f64);
            let a_imag = _mm256_loadu_pd(a.as_ptr().add(i).add(chunk_size) as *const f64);
            let b_real = _mm256_loadu_pd(b.as_ptr().add(i) as *const f64);
            let b_imag = _mm256_loadu_pd(b.as_ptr().add(i).add(chunk_size) as *const f64);

            // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
            let ac = _mm256_mul_pd(a_real, b_real);
            let bd = _mm256_mul_pd(a_imag, b_imag);
            let ad = _mm256_mul_pd(a_real, b_imag);
            let bc = _mm256_mul_pd(a_imag, b_real);

            let res_real = _mm256_sub_pd(ac, bd);
            let res_imag = _mm256_add_pd(ad, bc);

            // Store results
            let result_ptr = result.as_mut_ptr().add(i) as *mut f64;
            _mm256_storeu_pd(result_ptr, res_real);
            _mm256_storeu_pd(result_ptr.add(chunk_size), res_imag);

            i += chunk_size;
        }
    }

    // Handle remaining elements
    while i < result_len {
        result.push(a[i] * b[i]);
        i += 1;
    }

    result
}

/// SIMD-optimized complex conjugate for complex64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn complex_conj_avx2_f32(a: &[Complex32]) -> Vec<Complex32> {
    let chunk_size = 8;
    let result_len = a.len();
    let mut result: Vec<Complex32> = Vec::with_capacity(result_len);

    let mut i = 0;
    unsafe {
        // Zero for negating imaginary part
        let zeros = _mm256_setzero_ps();

        while i + chunk_size <= result_len {
            let a_real = _mm256_loadu_ps(a.as_ptr().add(i) as *const f32);
            let a_imag = _mm256_loadu_ps(a.as_ptr().add(i).add(chunk_size) as *const f32);

            // Conjugate: negate imaginary part
            let res_imag = _mm256_sub_ps(zeros, a_imag);

            let result_ptr = result.as_mut_ptr().add(i) as *mut f32;
            _mm256_storeu_ps(result_ptr, a_real);
            _mm256_storeu_ps(result_ptr.add(chunk_size), res_imag);

            i += chunk_size;
        }
    }

    // Handle remaining elements
    while i < result_len {
        result.push(a[i].conj());
        i += 1;
    }

    result
}

/// SIMD-optimized complex magnitude for complex64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn complex_abs_avx2_f32(a: &[Complex32]) -> Vec<f32> {
    let chunk_size = 8;
    let result_len = a.len();
    let mut result: Vec<f32> = Vec::with_capacity(result_len);

    let mut i = 0;
    unsafe {
        while i + chunk_size <= result_len {
            let a_real = _mm256_loadu_ps(a.as_ptr().add(i) as *const f32);
            let a_imag = _mm256_loadu_ps(a.as_ptr().add(i).add(chunk_size) as *const f32);

            // |z| = sqrt(real^2 + imag^2)
            let real_sq = _mm256_mul_ps(a_real, a_real);
            let imag_sq = _mm256_mul_ps(a_imag, a_imag);
            let sum_sq = _mm256_add_ps(real_sq, imag_sq);

            // sqrt using AVX sqrt instruction
            let abs_val = _mm256_sqrt_ps(sum_sq);

            // Store result
            _mm256_storeu_ps(result.as_mut_ptr().add(i), abs_val);

            i += chunk_size;
        }
    }

    // Handle remaining elements
    while i < result_len {
        result.push((a[i].re * a[i].re + a[i].im * a[i].im).sqrt());
        i += 1;
    }

    result
}

/// Fallback scalar implementation for complex addition
pub fn complex_add_scalar_f32(a: &[Complex32], b: &[Complex32]) -> Vec<Complex32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

/// Fallback scalar implementation for complex multiplication
pub fn complex_mul_scalar_f32(a: &[Complex32], b: &[Complex32]) -> Vec<Complex32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect()
}

/// Fallback scalar implementation for complex addition (f64)
pub fn complex_add_scalar_f64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

/// Fallback scalar implementation for complex multiplication (f64)
pub fn complex_mul_scalar_f64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect()
}

/// Detect and use best available SIMD implementation for complex addition
pub fn complex_add_f32(a: &[Complex32], b: &[Complex32]) -> Vec<Complex32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { complex_add_avx2_f32(a, b) };
        }
    }

    complex_add_scalar_f32(a, b)
}

/// Detect and use best available SIMD implementation for complex multiplication
pub fn complex_mul_f32(a: &[Complex32], b: &[Complex32]) -> Vec<Complex32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { complex_mul_avx2_f32(a, b) };
        }
    }

    complex_mul_scalar_f32(a, b)
}

/// Detect and use best available SIMD implementation for complex addition (f64)
pub fn complex_add_f64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { complex_add_avx2_f64(a, b) };
        }
    }

    complex_add_scalar_f64(a, b)
}

/// Detect and use best available SIMD implementation for complex multiplication (f64)
pub fn complex_mul_f64(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { complex_mul_avx2_f64(a, b) };
        }
    }

    complex_mul_scalar_f64(a, b)
}

/// Complex conjugate using best available implementation
pub fn complex_conj_f32(a: &[Complex32]) -> Vec<Complex32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { complex_conj_avx2_f32(a) };
        }
    }

    a.iter().map(|x| x.conj()).collect()
}

/// Complex magnitude using best available implementation
pub fn complex_abs_f32(a: &[Complex32]) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { complex_abs_avx2_f32(a) };
        }
    }

    a.iter().map(|x| (x.re * x.re + x.im * x.im).sqrt()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_add_scalar_f32() {
        let a = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let b = vec![Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];

        let result = complex_add_scalar_f32(&a, &b);

        assert_eq!(result[0], Complex32::new(6.0, 8.0));
        assert_eq!(result[1], Complex32::new(10.0, 12.0));
    }

    #[test]
    fn test_complex_mul_scalar_f32() {
        let a = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let b = vec![Complex32::new(2.0, 1.0), Complex32::new(1.0, 2.0)];

        let result = complex_mul_scalar_f32(&a, &b);

        // (1+2i) * (2+1i) = 2 + i + 4i + 2i^2 = 2 + 5i - 2 = 5i
        assert!((result[0].re - 0.0).abs() < 1e-6);
        assert!((result[0].im - 5.0).abs() < 1e-6);

        // (3+4i) * (1+2i) = 3 + 6i + 4i + 8i^2 = 3 + 10i - 8 = -5 + 10i
        assert!((result[1].re - (-5.0)).abs() < 1e-6);
        assert!((result[1].im - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_conj_f32() {
        let a = vec![Complex32::new(1.0, 2.0), Complex32::new(-3.0, -4.0)];

        let result = complex_conj_f32(&a);

        assert_eq!(result[0], Complex32::new(1.0, -2.0));
        assert_eq!(result[1], Complex32::new(-3.0, 4.0));
    }

    #[test]
    fn test_complex_abs_f32() {
        let a = vec![Complex32::new(3.0, 4.0), Complex32::new(5.0, 12.0)];

        let result = complex_abs_f32(&a);

        // |3+4i| = 5
        assert!((result[0] - 5.0).abs() < 1e-6);
        // |5+12i| = 13
        assert!((result[1] - 13.0).abs() < 1e-6);
    }

    // Note: Direct AVX2 function testing requires compilation with -C target-feature=+avx2
    // The public API (complex_add_f32, etc.) is tested below which handles runtime detection

    #[test]
    fn test_complex_add_f32() {
        let a = vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)];
        let b = vec![Complex32::new(5.0, 6.0), Complex32::new(7.0, 8.0)];

        let result = complex_add_f32(&a, &b);

        assert_eq!(result[0], Complex32::new(6.0, 8.0));
        assert_eq!(result[1], Complex32::new(10.0, 12.0));
    }
}
