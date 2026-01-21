//! Tests for statistics module: corrcoef() and cov()

use numpy::*;
use numpy::statistics::{corrcoef, cov};

// Helper function to extract f64 values from test arrays
fn get_f64_val<T: Clone>(arr: &Array<T>, index: usize) -> f64
where
    T: numpy::statistics::AsF64,
{
    arr.to_vec()[index].as_f64().unwrap_or(0.0)
}

#[test]
fn test_cov_1d_single() {
    let a: Array<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = cov(&a, None, true, false, 1, None, None).unwrap();

    // Variance of [1,2,3,4,5] = 2.5 (unbiased)
    assert_eq!(result.shape(), &[1, 1]);
    let val = get_f64_val(&result, 0);
    assert!((val - 2.5).abs() < 1e-10);
}

#[test]
fn test_cov_1d_two_arrays() {
    let x: Array<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Array<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0];
    let result = cov(&x, Some(&y), true, false, 1, None, None).unwrap();

    // Should return 2x2 covariance matrix
    assert_eq!(result.shape(), &[2, 2]);

    // var(x) = 2.5, var(y) = 10.0, cov(x,y) = 5.0
    let cov_xx = get_f64_val(&result, 0);
    let cov_xy = get_f64_val(&result, 1);
    let cov_yx = get_f64_val(&result, 2);
    let cov_yy = get_f64_val(&result, 3);

    assert!((cov_xx - 2.5).abs() < 1e-10);
    assert!((cov_xy - 5.0).abs() < 1e-10);
    assert!((cov_yx - 5.0).abs() < 1e-10);
    assert!((cov_yy - 10.0).abs() < 1e-10);
}

#[test]
fn test_cov_2d_rowvar_true() {
    // Each row is a variable, columns are observations
    let m: Array<f64> = Array::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = cov(&m, None, true, false, 1, None, None).unwrap();

    // Should return 2x2 covariance matrix
    assert_eq!(result.shape(), &[2, 2]);

    // Row 0: [1,2,3], mean=2, var=1
    // Row 1: [4,5,6], mean=5, var=1
    // cov(0,1) = ((1-2)*(4-5) + (2-2)*(5-5) + (3-2)*(6-5)) / 2 = 1
    let cov_00 = get_f64_val(&result, 0);
    let cov_01 = get_f64_val(&result, 1);
    let cov_10 = get_f64_val(&result, 2);
    let cov_11 = get_f64_val(&result, 3);

    assert!((cov_00 - 1.0).abs() < 1e-10);
    assert!((cov_01 - 1.0).abs() < 1e-10);
    assert!((cov_10 - 1.0).abs() < 1e-10);
    assert!((cov_11 - 1.0).abs() < 1e-10);
}

#[test]
fn test_cov_2d_rowvar_false() {
    // Each column is a variable, rows are observations
    let m: Array<f64> = Array::from_shape_vec(vec![3, 2], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let result = cov(&m, None, false, false, 1, None, None).unwrap();

    // Should return 2x2 covariance matrix (columns as variables)
    assert_eq!(result.shape(), &[2, 2]);

    // Column 0: [1,2,3], mean=2, var=1
    // Column 1: [4,5,6], mean=5, var=1
    // cov(0,1) = 1
    let cov_00 = get_f64_val(&result, 0);
    let cov_01 = get_f64_val(&result, 1);
    let cov_10 = get_f64_val(&result, 2);
    let cov_11 = get_f64_val(&result, 3);

    assert!((cov_00 - 1.0).abs() < 1e-10);
    assert!((cov_01 - 1.0).abs() < 1e-10);
    assert!((cov_10 - 1.0).abs() < 1e-10);
    assert!((cov_11 - 1.0).abs() < 1e-10);
}

#[test]
fn test_cov_bias() {
    let a: Array<f64> = array![1.0, 2.0, 3.0];

    // Unbiased (ddof=1): var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 2 = 1
    let result_unbiased = cov(&a, None, true, false, 1, None, None).unwrap();
    let val_unbiased = get_f64_val(&result_unbiased, 0);
    assert!((val_unbiased - 1.0).abs() < 1e-10);

    // Biased: var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
    let result_biased = cov(&a, None, true, true, 0, None, None).unwrap();
    let val_biased = get_f64_val(&result_biased, 0);
    assert!((val_biased - 2.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_corrcoef_1d_single() {
    let a: Array<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = corrcoef(&a, None, true, None, None).unwrap();

    // Single array correlation is always 1.0
    assert_eq!(result.shape(), &[1, 1]);
    let val = get_f64_val(&result, 0);
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn test_corrcoef_1d_two_arrays() {
    let x: Array<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Array<f64> = array![2.0, 4.0, 6.0, 8.0, 10.0];
    let result = corrcoef(&x, Some(&y), true, None, None).unwrap();

    // Perfect positive correlation
    assert_eq!(result.shape(), &[2, 2]);

    // Correlation matrix diagonal should be 1, off-diagonal should be 1 (perfect correlation)
    assert!((get_f64_val(&result, 0) - 1.0).abs() < 1e-10);
    assert!((get_f64_val(&result, 1) - 1.0).abs() < 1e-10);
    assert!((get_f64_val(&result, 2) - 1.0).abs() < 1e-10);
    assert!((get_f64_val(&result, 3) - 1.0).abs() < 1e-10);
}

#[test]
fn test_corrcoef_2d() {
    let m: Array<f64> = Array::from_shape_vec(vec![2, 5], vec![
        1.0, 2.0, 3.0, 4.0, 5.0,
        2.0, 4.0, 6.0, 8.0, 10.0
    ]);

    let result = corrcoef(&m, None, true, None, None).unwrap();

    // Perfect positive correlation between the two rows
    assert_eq!(result.shape(), &[2, 2]);

    assert!((get_f64_val(&result, 0) - 1.0).abs() < 1e-10);
    assert!((get_f64_val(&result, 1) - 1.0).abs() < 1e-10);
    assert!((get_f64_val(&result, 2) - 1.0).abs() < 1e-10);
    assert!((get_f64_val(&result, 3) - 1.0).abs() < 1e-10);
}

#[test]
fn test_corrcoef_imperfect_correlation() {
    let x: Array<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Array<f64> = array![5.0, 4.0, 3.0, 2.0, 1.0];
    let result = corrcoef(&x, Some(&y), true, None, None).unwrap();

    // Perfect negative correlation
    assert_eq!(result.shape(), &[2, 2]);

    // Off-diagonal should be -1
    assert!((get_f64_val(&result, 1) - (-1.0)).abs() < 1e-10);
    assert!((get_f64_val(&result, 2) - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_corrcoef_no_correlation() {
    let x: Array<f64> = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y: Array<f64> = array![1.0, 1.0, 1.0, 1.0, 1.0];
    let result = corrcoef(&x, Some(&y), true, None, None);

    // Should error due to zero variance in y
    assert!(result.is_err());
}

#[test]
fn test_cov_empty_array() {
    let a: Array<f64> = Array::from_vec(vec![]);
    let result = cov(&a, None, true, false, 1, None, None);
    assert!(result.is_err());
}

#[test]
fn test_corrcoef_empty_array() {
    let a: Array<f64> = Array::from_vec(vec![]);
    let result = corrcoef(&a, None, true, None, None);
    assert!(result.is_err());
}

#[test]
fn test_cov_ddof() {
    let a: Array<f64> = array![1.0, 2.0, 3.0, 4.0];

    // ddof=0 (biased): divide by n=4
    let result_ddof0 = cov(&a, None, true, false, 0, None, None).unwrap();
    let val_ddof0 = get_f64_val(&result_ddof0, 0);
    let expected_ddof0 = 5.0 / 4.0;  // sum of squared diffs / 4
    assert!((val_ddof0 - expected_ddof0).abs() < 1e-10);

    // ddof=1 (unbiased): divide by n-1=3
    let result_ddof1 = cov(&a, None, true, false, 1, None, None).unwrap();
    let val_ddof1 = get_f64_val(&result_ddof1, 0);
    let expected_ddof1 = 5.0 / 3.0;  // sum of squared diffs / 3
    assert!((val_ddof1 - expected_ddof1).abs() < 1e-10);
}
