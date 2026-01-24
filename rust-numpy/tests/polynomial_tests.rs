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

#[test]
fn test_polynomial_add() {
    // p1(x) = x^2 + 2x + 1
    // p2(x) = 3x^2 + 4x + 5
    // sum = 4x^2 + 6x + 6
    let p1 = Polynomial::new(&array![1.0, 2.0, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![5.0, 4.0, 3.0]).unwrap();

    let result = p1.add(&p2);
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 6.0);
    assert_eq!(coeffs[1], 6.0);
    assert_eq!(coeffs[2], 4.0);
}

#[test]
fn test_polynomial_sub() {
    // p1(x) = 5x^2 + 6x + 7
    // p2(x) = 2x^2 + 3x + 4
    // diff = 3x^2 + 3x + 3
    let p1 = Polynomial::new(&array![7.0, 6.0, 5.0]).unwrap();
    let p2 = Polynomial::new(&array![4.0, 3.0, 2.0]).unwrap();

    let result = p1.sub(&p2);
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 3.0);
    assert_eq!(coeffs[1], 3.0);
    assert_eq!(coeffs[2], 3.0);
}

#[test]
fn test_polynomial_mul() {
    // p1(x) = x + 1
    // p2(x) = x + 2
    // product = x^2 + 3x + 2
    let p1 = Polynomial::new(&array![1.0, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![2.0, 1.0]).unwrap();

    let result = p1.mul(&p2);
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 2.0);
    assert_eq!(coeffs[1], 3.0);
    assert_eq!(coeffs[2], 1.0);
}

#[test]
fn test_polynomial_add_operator() {
    use std::ops::Add;

    let p1 = Polynomial::new(&array![1.0, 2.0, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![5.0, 4.0, 3.0]).unwrap();

    let result = p1 + p2;
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 6.0);
    assert_eq!(coeffs[1], 6.0);
    assert_eq!(coeffs[2], 4.0);
}

#[test]
fn test_polynomial_sub_operator() {
    use std::ops::Sub;

    let p1 = Polynomial::new(&array![7.0, 6.0, 5.0]).unwrap();
    let p2 = Polynomial::new(&array![4.0, 3.0, 2.0]).unwrap();

    let result = p1 - p2;
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 3.0);
    assert_eq!(coeffs[1], 3.0);
    assert_eq!(coeffs[2], 3.0);
}

#[test]
fn test_polynomial_mul_operator() {
    use std::ops::Mul;

    let p1 = Polynomial::new(&array![1.0, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![2.0, 1.0]).unwrap();

    let result = p1 * p2;
    let coeffs = result.coeffs();

    assert_eq!(coeffs[0], 2.0);
    assert_eq!(coeffs[1], 3.0);
    assert_eq!(coeffs[2], 1.0);
}

#[test]
fn test_polynomial_domain_window() {
    let p = Polynomial::new(&array![1.0, 2.0, 1.0]).unwrap();

    // Test domain and window access
    let domain = p.domain();
    let window = p.window();

    assert_eq!(domain, [1.0, 1.0]);
    assert_eq!(window, [1.0, 1.0]);
}

#[test]
fn test_polynomial_set_domain() {
    let mut p = Polynomial::new(&array![1.0, 2.0, 1.0]).unwrap();

    p.set_domain([0.0, 2.0]);
    assert_eq!(p.domain(), [0.0, 2.0]);

    p.set_window([-1.0, 1.0]);
    assert_eq!(p.window(), [-1.0, 1.0]);
}

#[test]
fn test_polynomial_different_degrees() {
    // Test addition with different degree polynomials
    // p1(x) = x^3 + 1 (degree 3)
    // p2(x) = x^2 + x (degree 2)
    let p1 = Polynomial::new(&array![1.0, 0.0, 0.0, 1.0]).unwrap();
    let p2 = Polynomial::new(&array![0.0, 1.0, 1.0]).unwrap();

    let result = p1.add(&p2);
    let coeffs = result.coeffs();

    // Should have degree 3: x^3 + x^2 + x + 1
    assert_eq!(coeffs.len(), 4);
    assert_eq!(coeffs[0], 1.0);
    assert_eq!(coeffs[1], 1.0);
    assert_eq!(coeffs[2], 1.0);
    assert_eq!(coeffs[3], 1.0);
}

#[test]
fn test_fit_higher_degree() {
    // Fit quadratic: y = x^2 + 2x + 1
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = array![1.0, 4.0, 9.0, 16.0, 25.0];

    let p = fit(&x, &y, 2).unwrap();
    let coeffs = p.coeffs();

    // Should approximately be [1, 2, 1]
    assert!((coeffs[0] - 1.0_f64).abs() < 1e-6);
    assert!((coeffs[1] - 2.0_f64).abs() < 1e-6);
    assert!((coeffs[2] - 1.0_f64).abs() < 1e-6);
}

