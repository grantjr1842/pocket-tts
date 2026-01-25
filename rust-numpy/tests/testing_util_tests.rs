use numpy::*;

#[test]
fn test_assert_array_equal() {
    let a = array![1, 2, 3];
    let b = array![1, 2, 3];
    let c = array![1, 2, 4];
    let d = array![1, 2];

    assert!(assert_array_equal(&a, &b).is_ok());
    assert!(assert_array_equal(&a, &c).is_err());
    assert!(assert_array_equal(&a, &d).is_err());
}

#[test]
fn test_assert_array_almost_equal() {
    let a = array![1.0, 2.0, 3.000001];
    let b = array![1.0, 2.0, 3.000002];

    assert!(assert_array_almost_equal(&a, &b, 5).is_ok());
    assert!(assert_array_almost_equal(&a, &b, 7).is_err());
}

#[test]
fn test_assert_array_shape_equal() {
    let a = array![1, 2, 3];
    let b = array![4, 5, 6];
    let c = array2![[1, 2], [3, 4]];

    assert!(assert_array_shape_equal(&a, &b).is_ok());
    assert!(assert_array_shape_equal(&a, &c).is_err());
}

#[test]
fn test_assert_allclose() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![1.0000001, 2.0000001, 3.0000001];

    assert!(assert_allclose(&a, &b, 1e-6, 1e-6).is_ok());
    assert!(assert_allclose(&a, &b, 1e-8, 1e-8).is_err());
}

#[test]
fn test_allclose() {
    let a = array![1.0, 2.0, 3.0];
    let b = array![1.0000001, 2.0000001, 3.0000001];

    assert!(allclose(&a, &b, Some(1e-6), Some(1e-6), None).unwrap());
    assert!(!allclose(&a, &b, Some(1e-8), Some(1e-8), None).unwrap());
}

#[test]
fn test_assert_array_less() {
    let a = array![1, 2, 3];
    let b = array![2, 3, 4];
    let c = array![1, 2, 3];

    assert!(assert_array_less(&a, &b).is_ok());
    assert!(assert_array_less(&a, &c).is_err());
}

#[test]
fn test_assert_equal() {
    let a = 1;
    let b = 1;
    let c = 2;

    assert!(assert_equal(&a, &b).is_ok());
    assert!(assert_equal(&a, &c).is_err());
}

#[test]
fn test_assert_almost_equal_scalar() {
    let a = 1.0;
    let b = 1.000001;
    let c = 1.001;

    assert!(assert_almost_equal(&a, &b, 5).is_ok());
    assert!(assert_almost_equal(&a, &c, 5).is_err());
}

#[test]
fn test_assert_string_equal() {
    let a = "hello";
    let b = "hello";
    let c = "world";

    assert!(assert_string_equal(a, b).is_ok());
    assert!(assert_string_equal(a, c).is_err());
}

#[test]
fn test_assert_() {
    assert!(assert_(true).is_ok());
    assert!(assert_(false).is_err());
}

#[test]
fn test_assert_with_msg() {
    assert!(assert_with_msg(true, "test message").is_ok());
    assert!(assert_with_msg(false, "test message").is_err());
}

#[test]
fn test_assert_approx_equal() {
    let a = 1.0;
    let b = 1.0001;
    let c = 1.01;

    assert!(assert_approx_equal(a, b, 3).is_ok());
    assert!(assert_approx_equal(a, c, 3).is_err());
}

#[test]
fn test_assert_raises() {
    // Test with a function that panics
    let panic_fn = || {
        panic!("This is a test panic");
    };

    assert!(assert_raises("panic", panic_fn).is_ok());

    // Test with a function that doesn't panic
    let normal_fn = || 42;

    assert!(assert_raises("panic", normal_fn).is_err());
}

#[test]
fn test_assert_raises_regex() {
    // Test with a function that panics
    let panic_fn = || {
        panic!("This is a test panic");
    };

    assert!(assert_raises_regex("panic", "test", panic_fn).is_ok());

    // Test with a function that doesn't panic
    let normal_fn = || 42;

    assert!(assert_raises_regex("panic", "test", normal_fn).is_err());
}

#[test]
fn test_assert_warns() {
    let test_fn = || 42;

    // Placeholder test - should return the result
    let result = assert_warns("warning", test_fn);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_assert_no_warnings() {
    let test_fn = || 42;

    // Placeholder test - should return the result
    let result = assert_no_warnings(test_fn);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_assert_no_gc_cycles() {
    let test_fn = || 42;

    // Placeholder test - should return the result
    let result = assert_no_gc_cycles(test_fn);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_print_assert_equal() {
    let a = 42;
    let b = 42;
    let c = 43;

    // Should pass and print success message
    assert!(print_assert_equal("test_equal", &a, &b).is_ok());

    // Should fail and print error message
    assert!(print_assert_equal("test_not_equal", &a, &c).is_err());
}

#[test]
fn test_assert_array_almost_equal_nulp() {
    use crate::array::Array;

    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let c = Array::from_vec(vec![1.0, 2.001, 3.0]);

    // Should pass - identical arrays
    assert!(assert_array_almost_equal_nulp(&a, &b, 1).is_ok());

    // Should fail - arrays differ
    assert!(assert_array_almost_equal_nulp(&a, &c, 1).is_err());
}

#[test]
fn test_assert_array_max_ulp() {
    use crate::array::Array;

    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let c = Array::from_vec(vec![1.0, 2.001, 3.0]);

    // Should pass - identical arrays
    assert!(assert_array_max_ulp(&a, &b, 1).is_ok());

    // Should fail - arrays differ
    assert!(assert_array_max_ulp(&a, &c, 1).is_err());
}
