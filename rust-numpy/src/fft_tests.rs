use crate::fft::*;
use crate::Array;
use num_complex::Complex64;

#[test]
fn test_fft_norm_enum_creation() {
    // Test from_str method
    assert_eq!(FFTNorm::from_str("backward"), Some(FFTNorm::Backward));
    assert_eq!(FFTNorm::from_str("ortho"), Some(FFTNorm::Ortho));
    assert_eq!(FFTNorm::from_str("forward"), Some(FFTNorm::Forward));
    assert_eq!(FFTNorm::from_str("invalid"), None);
}

#[test]
fn test_fft_norm_forward_scale() {
    let n = 4;
    assert_eq!(FFTNorm::Backward.forward_scale(n), 1.0);
    assert_eq!(FFTNorm::Ortho.forward_scale(n), (n as f64).sqrt());
    assert_eq!(FFTNorm::Forward.forward_scale(n), n as f64);
}

#[test]
fn test_fft_norm_inverse_scale() {
    let n = 4;
    assert_eq!(FFTNorm::Backward.inverse_scale(n), n as f64);
    assert_eq!(FFTNorm::Ortho.inverse_scale(n), (n as f64).sqrt());
    assert_eq!(FFTNorm::Forward.inverse_scale(n), 1.0);
}

#[test]
fn test_fft_backward_normalization() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let input = Array::from_vec(x);

    // Test backward normalization (default behavior)
    let y = fft(&input, None, 0, Some(FFTNorm::Backward)).unwrap();
    let x_reconstructed = ifft(&y, None, 0, Some(FFTNorm::Backward)).unwrap();

    for i in 0..4 {
        assert!((x_reconstructed.get(i).unwrap().re - *input.get(i).unwrap()).abs() < 1e-10);
    }
}

#[test]
fn test_fft_ortho_normalization() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let input = Array::from_vec(x);

    // Test ortho normalization
    let y = fft(&input, None, 0, Some(FFTNorm::Ortho)).unwrap();
    let x_reconstructed = ifft(&y, None, 0, Some(FFTNorm::Ortho)).unwrap();

    for i in 0..4 {
        assert!((x_reconstructed.get(i).unwrap().re - *input.get(i).unwrap()).abs() < 1e-10);
    }
}

#[test]
fn test_fft_forward_normalization() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let input = Array::from_vec(x);

    // Test forward normalization
    let y = fft(&input, None, 0, Some(FFTNorm::Forward)).unwrap();
    let x_reconstructed = ifft(&y, None, 0, Some(FFTNorm::Forward)).unwrap();

    for i in 0..4 {
        assert!((x_reconstructed.get(i).unwrap().re - *input.get(i).unwrap()).abs() < 1e-10);
    }
}

#[test]
fn test_rfft_normalization_roundtrip() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let input = Array::from_vec(x);

    // Test all normalization modes for rfft/irfft
    let norms = [FFTNorm::Backward, FFTNorm::Ortho, FFTNorm::Forward];

    for &norm in &norms {
        let y = rfft(&input, None, 0, Some(norm)).unwrap();
        let x_reconstructed = irfft(&y, Some(4), 0, Some(norm)).unwrap();

        for i in 0..4 {
            assert!(
                (x_reconstructed.get(i).unwrap() - input.get(i).unwrap()).abs() < 1e-10,
                "Failed for norm: {:?}",
                norm
            );
        }
    }
}

#[test]
fn test_fft2_normalization() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Array::from_shape_vec(vec![2, 2], data);

    // Test ortho normalization for 2D FFT
    let y = fft2(&input, None, None, Some(FFTNorm::Ortho)).unwrap();
    let x_reconstructed = ifft2(&y, None, None, Some(FFTNorm::Ortho)).unwrap();

    for i in 0..4 {
        assert!((x_reconstructed.get(i).unwrap().re - *input.get(i).unwrap()).abs() < 1e-10);
    }
}

#[test]
fn test_fftfreq_basic() {
    let freqs = fftfreq(8, 0.1);
    assert_eq!(freqs.len(), 8);
    assert_eq!(freqs[0], 0.0);
    assert_eq!(freqs[1], 1.25);
}

#[test]
fn test_rfftfreq_basic() {
    let freqs = rfftfreq(8, 0.1);
    assert_eq!(freqs.len(), 5);
    assert_eq!(freqs[0], 0.0);
    assert_eq!(freqs[1], 1.25);
    assert_eq!(freqs[4], 5.0);

    let freqs_odd = rfftfreq(7, 0.1);
    assert_eq!(freqs_odd.len(), 4);
    assert_eq!(freqs_odd[0], 0.0);
    assert!((freqs_odd[3] - 3.0 / (7.0 * 0.1)).abs() < 1e-10);
}

#[test]
fn test_fftshift_basic() {
    let a = Array::from_vec(vec![0, 1, 2, 3, 4]);
    let res = fftshift(&a, None);
    assert_eq!(res.to_vec(), vec![3, 4, 0, 1, 2]);
}

#[test]
fn test_ifftshift_basic() {
    let a = Array::from_vec(vec![3, 4, 0, 1, 2]);
    let res = ifftshift(&a, None);
    assert_eq!(res.to_vec(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_fft_basic() {
    let a = Array::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
    let res = fft(&a, None, 0, None).unwrap();
    let vals = res.to_vec();
    assert!((vals[0].re).abs() < 1e-10);
    assert!((vals[1].re - 2.0).abs() < 1e-10);
    assert!((vals[2].re).abs() < 1e-10);
    assert!((vals[3].re - 2.0).abs() < 1e-10);
}

#[test]
fn test_ifft_basic() {
    let a = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let arr = Array::from_vec(a);
    let res = ifft(&arr, None, 0, None).unwrap();
    let vals = res.to_vec();
    assert!((vals[0].re - 1.0).abs() < 1e-10);
    assert!((vals[1].re).abs() < 1e-10);
    assert!((vals[2].re + 1.0).abs() < 1e-10);
    assert!((vals[3].re).abs() < 1e-10);
}

#[test]
fn test_rfft_basic() {
    let a = Array::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
    let res = rfft(&a, None, 0, None).unwrap();
    let vals = res.to_vec();
    assert_eq!(vals.len(), 3);
    assert!((vals[0].re).abs() < 1e-10);
    assert!((vals[1].re - 2.0).abs() < 1e-10);
    assert!((vals[2].re).abs() < 1e-10);
}

#[test]
fn test_irfft_basic() {
    let a = vec![
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    let arr = Array::from_vec(a);
    let res = irfft(&arr, Some(4), 0, None).unwrap();
    let vals = res.to_vec();
    assert_eq!(vals.len(), 4);
    assert!((vals[0] - 1.0).abs() < 1e-10);
    assert!((vals[1]).abs() < 1e-10);
    assert!((vals[2] + 1.0).abs() < 1e-10);
    assert!((vals[3]).abs() < 1e-10);
}

fn assert_real_approx(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "real mismatch at {}: {} != {}",
            idx,
            a,
            e
        );
    }
}

fn assert_complex_approx(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a.re - e.re).abs() <= tol && (a.im - e.im).abs() <= tol,
            "complex mismatch at {}: {} != {}",
            idx,
            a,
            e
        );
    }
}

#[test]
fn test_rfft2_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Array::from_shape_vec(vec![2, 2], data);

    let result = rfft2(&input, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_irfft2_basic() {
    let data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let input = Array::from_shape_vec(vec![2, 2], data);

    let result = irfft2(&input, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_rfftn_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Array::from_shape_vec(vec![2, 2, 2], data);

    let result = rfftn(&input, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_irfftn_basic() {
    let data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let input = Array::from_shape_vec(vec![2, 2], data);

    let result = irfftn(&input, None, None, None);
    assert!(result.is_ok());
}

// Test removed: hilbert_with_params function does not exist
// TODO: Implement hilbert transform if needed

#[test]
fn test_fft_with_params_basic() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Array::from_vec(data);

    let result = fft(&input, None, 0, None);
    assert!(result.is_ok());
}

#[test]
fn test_fft_known_delta() {
    let data = vec![1.0, 0.0, 0.0, 0.0];
    let input = Array::from_vec(data);

    let result = fft(&input, None, 0, None).unwrap();
    let expected = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    assert_complex_approx(result.as_slice(), &expected, 1e-9);
}

#[test]
fn test_ifft_known_constant() {
    let data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let input = Array::from_vec(data);

    let result = ifft(&input, None, 0, None).unwrap();
    let expected = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];
    assert_complex_approx(result.as_slice(), &expected, 1e-9);
}

#[test]
fn test_rfft2_shape_with_s() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Array::from_shape_vec(vec![2, 3], data);

    let result = rfft2(&input, Some(&[4, 4]), None, None).unwrap();
    assert_eq!(result.shape(), vec![4, 3]);
    assert_eq!(result.size(), 12);
}

#[test]
fn test_rfft2_irfft2_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Array::from_shape_vec(vec![2, 3], data.clone());

    let spectrum = rfft2(&input, None, None, None).unwrap();
    let reconstructed = irfft2(&spectrum, Some(&[2, 3]), None, None).unwrap();
    assert_eq!(reconstructed.shape(), vec![2, 3]);

    let expected = data;
    assert_real_approx(reconstructed.as_slice(), &expected, 1e-9);
}

#[test]
fn test_rfftn_axes_shape() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Array::from_shape_vec(vec![2, 2, 2], data);

    let result = rfftn(&input, Some(&[4, 5]), Some(&[0, 2]), None).unwrap();
    assert_eq!(result.shape(), vec![4, 2, 3]);
    assert_eq!(result.size(), 24);
}

#[test]
fn test_rfftn_duplicate_axes_error() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Array::from_shape_vec(vec![2, 2], data);

    let result = rfftn(&input, None, Some(&[1, 1]), None);
    assert!(result.is_err());
}

#[test]
fn test_irfftn_shape_with_s() {
    let data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
    ];
    let input = Array::from_shape_vec(vec![2, 3], data);

    let result = irfftn(&input, Some(&[4, 5]), None, None).unwrap();
    assert_eq!(result.shape(), vec![4, 5]);
    assert_eq!(result.size(), 20);
}

#[test]
fn test_rfftn_irfftn_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = Array::from_shape_vec(vec![2, 3], data.clone());

    let spectrum = rfftn(&input, None, None, None).unwrap();
    let reconstructed = irfftn(&spectrum, Some(&[2, 3]), None, None).unwrap();
    assert_eq!(reconstructed.shape(), vec![2, 3]);
    assert_real_approx(reconstructed.as_slice(), &data, 1e-9);
}

#[test]
fn test_irfftn_empty_axes() {
    let data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let input = Array::from_shape_vec(vec![2, 2], data);

    let result = irfftn(&input, None, Some(&[]), None).unwrap();
    assert_eq!(result.shape(), vec![2, 2]);
    assert_eq!(result.size(), 4);
}

#[test]
fn test_irfft2_invalid_axes() {
    let data = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let input = Array::from_shape_vec(vec![2, 2], data);

    let result = irfft2(&input, None, Some(&[0]), None);
    assert!(result.is_err());
}
