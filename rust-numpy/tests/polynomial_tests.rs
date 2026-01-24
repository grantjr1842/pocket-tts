use ndarray::{array, Array1};

use numpy::polynomial::{fit, roots, Polynomial, PolynomialBase};

#[test]
fn test_poly_eval() {
    // p(x) = x^2 + 2x + 3
    let p = Polynomial::new(&array![3.0, 2.0, 1.0]).unwrap();
    let x = array![0.0, 1.0, 2.0];
    let y = p.eval(&x).unwrap();

    // p(0) = 3
    // p(1) = 1 + 2 + 3 = 6
    // p(2) = 4 + 4 + 3 = 11
    assert_eq!(y, array![3.0, 6.0, 11.0]);
}

#[test]
fn test_roots_quadratic() {
    // x^2 - 1 = 0 => x = 1, -1
    // coeffs: [-1, 0, 1]
    let p = Polynomial::new(&array![-1.0, 0.0, 1.0]).unwrap();
    let r = roots(&p).unwrap();

    // Sort roots for comparison (by real part)
    let mut r_vec = r.to_vec();
    r_vec.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

    assert!((r_vec[0].re - -1.0_f64).abs() < 1e-8_f64);
    assert!((r_vec[1].re - 1.0_f64).abs() < 1e-8_f64);
}

#[test]
fn test_roots_cubic() {
    // (x - 1)(x - 2)(x - 3) = 0
    // x^3 - 6x^2 + 11x - 6
    // coeffs: [-6, 11, -6, 1]
    let p = Polynomial::new(&array![-6.0, 11.0, -6.0, 1.0]).unwrap();
    let r = roots(&p).unwrap();

    let mut r_vec = r.to_vec();
    r_vec.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());

    assert!((r_vec[0].re - 1.0_f64).abs() < 1e-8_f64);
    assert!((r_vec[1].re - 2.0_f64).abs() < 1e-8_f64);
    assert!((r_vec[2].re - 3.0_f64).abs() < 1e-8_f64);
}

#[test]
fn test_fit_linear() {
    // y = 2x + 1
    let x = array![0.0, 1.0, 2.0, 3.0];
    let y = array![1.0, 3.0, 5.0, 7.0];

    let p = fit(&x, &y, 1).unwrap();
    let coeffs = p.coeffs();

    assert!((coeffs[0] - 1.0_f64).abs() < 1e-8_f64);
    assert!((coeffs[1] - 2.0_f64).abs() < 1e-8_f64);
}
