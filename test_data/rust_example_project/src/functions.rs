//! Mathematical functions for Rust NumPy port
//!
//! This module implements basic mathematical functions that match
//! NumPy's behavior exactly.

use std::f64::consts::{E, PI};

/// Element-wise addition
pub fn add(a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    if a.len() != b.len() {
        return Err("Arrays must have same length".to_string());
    }

    let result: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

    Ok(result)
}

/// Element-wise subtraction
pub fn subtract(a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    if a.len() != b.len() {
        return Err("Arrays must have same length".to_string());
    }

    let result: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();

    Ok(result)
}

/// Element-wise multiplication
pub fn multiply(a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    if a.len() != b.len() {
        return Err("Arrays must have same length".to_string());
    }

    let result: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

    Ok(result)
}

/// Element-wise division
pub fn divide(a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    if a.len() != b.len() {
        return Err("Arrays must have same length".to_string());
    }

    let result: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            if y == 0.0 {
                if x == 0.0 {
                    f64::NAN // 0/0 = NaN
                } else if x > 0.0 {
                    f64::INFINITY // positive/0 = +inf
                } else {
                    f64::NEG_INFINITY // negative/0 = -inf
                }
            } else {
                x / y
            }
        })
        .collect();

    Ok(result)
}

/// Element-wise power
pub fn power(a: &[f64], power: f64) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a
        .iter()
        .map(|&x| {
            if x < 0.0 && power.fract() != 0.0 {
                f64::NAN // Negative base with non-integer power
            } else {
                x.powf(power)
            }
        })
        .collect();

    Ok(result)
}

/// Element-wise square root
pub fn sqrt(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a
        .iter()
        .map(|&x| if x < 0.0 { f64::NAN } else { x.sqrt() })
        .collect();

    Ok(result)
}

/// Element-wise absolute value
pub fn abs(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a.iter().map(|&x| x.abs()).collect();

    Ok(result)
}

/// Element-wise sine
pub fn sin(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a.iter().map(|&x| x.sin()).collect();

    Ok(result)
}

/// Element-wise cosine
pub fn cos(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a.iter().map(|&x| x.cos()).collect();

    Ok(result)
}

/// Element-wise tangent
pub fn tan(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a
        .iter()
        .map(|&x| {
            // Handle cos(x) ≈ 0 cases where tan would be infinite
            let cos_x = x.cos();
            if cos_x.abs() < 1e-15 {
                if x.sin() > 0.0 {
                    f64::INFINITY
                } else if x.sin() < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    f64::NAN // sin(0)/cos(0) = 0/0 = NaN
                }
            } else {
                x.tan()
            }
        })
        .collect();

    Ok(result)
}

/// Element-wise exponential
pub fn exp(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a
        .iter()
        .map(|&x| {
            if x.is_nan() {
                f64::NAN
            } else if x == f64::NEG_INFINITY {
                0.0
            } else if x == f64::INFINITY {
                f64::INFINITY
            } else {
                x.exp()
            }
        })
        .collect();

    Ok(result)
}

/// Element-wise natural logarithm
pub fn log(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a
        .iter()
        .map(|&x| if x <= 0.0 { f64::NAN } else { x.ln() })
        .collect();

    Ok(result)
}

/// Element-wise base-10 logarithm
pub fn log10(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a
        .iter()
        .map(|&x| if x <= 0.0 { f64::NAN } else { x.log10() })
        .collect();

    Ok(result)
}

/// Element-wise base-2 logarithm
pub fn log2(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a
        .iter()
        .map(|&x| if x <= 0.0 { f64::NAN } else { x.log2() })
        .collect();

    Ok(result)
}

/// Element-wise floor
pub fn floor(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a.iter().map(|&x| x.floor()).collect();

    Ok(result)
}

/// Element-wise ceiling
pub fn ceil(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a.iter().map(|&x| x.ceil()).collect();

    Ok(result)
}

/// Element-wise round to nearest integer
pub fn round(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a.iter().map(|&x| x.round()).collect();

    Ok(result)
}

/// Element-wise sign
pub fn sign(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a
        .iter()
        .map(|&x| {
            if x.is_nan() {
                f64::NAN
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect();

    Ok(result)
}

/// Element-wise negative
pub fn negative(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a.iter().map(|&x| -x).collect();

    Ok(result)
}

/// Element-wise positive (returns the array unchanged)
pub fn positive(a: &[f64]) -> Result<Vec<f64>, String> {
    Ok(a.to_vec())
}

/// Check if values are NaN
pub fn isnan(a: &[f64]) -> Result<Vec<bool>, String> {
    let result: Vec<bool> = a.iter().map(|&x| x.is_nan()).collect();

    Ok(result)
}

/// Check if values are infinite
pub fn isinf(a: &[f64]) -> Result<Vec<bool>, String> {
    let result: Vec<bool> = a.iter().map(|&x| x.is_infinite()).collect();

    Ok(result)
}

/// Check if values are finite
pub fn isfinite(a: &[f64]) -> Result<Vec<bool>, String> {
    let result: Vec<bool> = a.iter().map(|&x| x.is_finite()).collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = add(&a, &b).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_subtract() {
        let a = vec![5.0, 7.0, 9.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = subtract(&a, &b).unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_multiply() {
        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0];
        let result = multiply(&a, &b).unwrap();
        assert_eq!(result, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_divide() {
        let a = vec![10.0, 12.0, 14.0];
        let b = vec![2.0, 3.0, 7.0];
        let result = divide(&a, &b).unwrap();
        assert_eq!(result, vec![5.0, 4.0, 2.0]);
    }

    #[test]
    fn test_divide_by_zero() {
        let a = vec![1.0, -1.0, 0.0];
        let b = vec![0.0, 0.0, 0.0];
        let result = divide(&a, &b).unwrap();
        assert!(result[0].is_infinite() && result[0].is_sign_positive());
        assert!(result[1].is_infinite() && result[1].is_sign_negative());
        assert!(result[2].is_nan());
    }

    #[test]
    fn test_power() {
        let a = vec![2.0, 3.0, 4.0];
        let result = power(&a, 2.0).unwrap();
        assert_eq!(result, vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_sqrt() {
        let a = vec![1.0, 4.0, 9.0, 16.0];
        let result = sqrt(&a).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sqrt_negative() {
        let a = vec![-1.0, -4.0, 1.0];
        let result = sqrt(&a).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0);
    }

    #[test]
    fn test_abs() {
        let a = vec![-1.0, 2.0, -3.0];
        let result = abs(&a).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sin() {
        let a = vec![0.0, PI / 2.0, PI];
        let result = sin(&a).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cos() {
        let a = vec![0.0, PI / 2.0, PI];
        let result = cos(&a).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp() {
        let a = vec![0.0, 1.0, 2.0];
        let result = exp(&a).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - E).abs() < 1e-10);
        assert!((result[2] - E * E).abs() < 1e-10);
    }

    #[test]
    fn test_log() {
        let a = vec![1.0, E, E * E];
        let result = log(&a).unwrap();
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_negative() {
        let a = vec![-1.0, 0.0, 1.0];
        let result = log(&a).unwrap();
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_floor() {
        let a = vec![1.2, 2.7, -0.5, -1.2];
        let result = floor(&a).unwrap();
        assert_eq!(result, vec![1.0, 2.0, -1.0, -2.0]);
    }

    #[test]
    fn test_ceil() {
        let a = vec![1.2, 2.7, -0.5, -1.2];
        let result = ceil(&a).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 0.0, -1.0]);
    }

    #[test]
    fn test_sign() {
        let a = vec![-2.0, 0.0, 2.0];
        let result = sign(&a).unwrap();
        assert_eq!(result, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_isnan() {
        let a = vec![1.0, f64::NAN, 2.0];
        let result = isnan(&a).unwrap();
        assert_eq!(result, vec![false, true, false]);
    }

    #[test]
    fn test_isinf() {
        let a = vec![1.0, f64::INFINITY, f64::NEG_INFINITY, 2.0];
        let result = isinf(&a).unwrap();
        assert_eq!(result, vec![false, true, true, false]);
    }
}
