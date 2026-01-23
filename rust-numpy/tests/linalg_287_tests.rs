use numpy::Array;
use numpy::math_ufuncs::{sign, signbit, copysign, absolute, fabs};

#[test]
fn test_sign_basic() {
    let a = Array::from_vec(vec![-2.0, 0.0, 3.0, f64::NAN]);
    let res = sign(&a).unwrap();
    assert_eq!(*res.get_linear(0).unwrap(), -1.0);
    assert_eq!(*res.get_linear(1).unwrap(), 0.0);
    assert_eq!(*res.get_linear(2).unwrap(), 1.0);
    assert!(res.get_linear(3).unwrap().is_nan());
}

#[test]
fn test_signbit_basic() {
    let a = Array::from_vec(vec![-2.0, 0.0, 3.0, -0.0]);
    let res = signbit(&a).unwrap();
    assert_eq!(*res.get_linear(0).unwrap(), true);
    assert_eq!(*res.get_linear(1).unwrap(), false);
    assert_eq!(*res.get_linear(2).unwrap(), false);
    assert_eq!(*res.get_linear(3).unwrap(), true);
}

#[test]
fn test_copysign_basic() {
    let a = Array::from_vec(vec![1.0, 2.0, -3.0]);
    let b = Array::from_vec(vec![-1.0, 1.0, -1.0]);
    let res = copysign(&a, &b).unwrap();
    assert_eq!(*res.get_linear(0).unwrap(), -1.0);
    assert_eq!(*res.get_linear(1).unwrap(), 2.0);
    assert_eq!(*res.get_linear(2).unwrap(), -3.0);
}

#[test]
fn test_absolute_basic() {
    let a = Array::from_vec(vec![-1.0, 0.0, 1.0]);
    let res = absolute(&a).unwrap();
    assert_eq!(*res.get_linear(0).unwrap(), 1.0);
    assert_eq!(*res.get_linear(1).unwrap(), 0.0);
    assert_eq!(*res.get_linear(2).unwrap(), 1.0);
}

#[test]
fn test_fabs_basic() {
    let a = Array::from_vec(vec![-1.0, 0.0, 1.0]);
    let res = fabs(&a).unwrap();
    assert_eq!(*res.get_linear(0).unwrap(), 1.0);
    assert_eq!(*res.get_linear(1).unwrap(), 0.0);
    assert_eq!(*res.get_linear(2).unwrap(), 1.0);
}
