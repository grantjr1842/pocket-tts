use crate::array::Array;
use crate::error::{NumPyError, Result};
use crate::linalg::LinalgScalar;
use num_traits::Float;
use std::fmt::Debug;
use std::panic;

/// Assert that two arrays are equal.
///
/// Raises an error if shapes or elements differ.
pub fn assert_array_equal<T>(actual: &Array<T>, desired: &Array<T>) -> Result<()>
where
    T: Clone + PartialEq + Debug + 'static,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    for (i, (a, b)) in actual.iter().zip(desired.iter()).enumerate() {
        if a != b {
            return Err(NumPyError::value_error(
                format!(
                    "Arrays differ at index {}: actual={:?}, desired={:?}",
                    i, a, b
                ),
                "".to_string(), // desired is already in the message
            ));
        }
    }

    Ok(())
}

/// Assert that two arrays are equal within a certain tolerance.
///
/// Specifically for floating point types.
pub fn assert_array_almost_equal<T>(
    actual: &Array<T>,
    desired: &Array<T>,
    decimal: usize,
) -> Result<()>
where
    T: Clone + Float + Debug + 'static,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    let threshold = T::from(10.0).unwrap().powi(-(decimal as i32));

    for (i, (a, b)) in actual.iter().zip(desired.iter()).enumerate() {
        if (*a - *b).abs() > threshold {
            return Err(NumPyError::value_error(
                format!(
                    "Arrays differ at index {} by more than {} decimal places: actual={:?}, desired={:?}",
                    i, decimal, a, b
                ),
                "".to_string(),
            ));
        }
    }

    Ok(())
}

/// Assert that two arrays are equal within tolerance
pub fn assert_allclose<T>(
    actual: &Array<T>,
    desired: &Array<T>,
    rtol: T::Real,
    atol: T::Real,
) -> Result<()>
where
    T: LinalgScalar + Debug,
    T::Real: Debug,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    for (i, (a, b)) in actual.iter().zip(desired.iter()).enumerate() {
        let diff = (*a - *b).abs();
        let b_abs = b.abs();
        if diff > atol + rtol * b_abs {
            return Err(NumPyError::value_error(
                format!(
                    "Arrays not close at index {}: actual={:?}, desired={:?} (diff={:?}, tol={:?})",
                    i,
                    a,
                    b,
                    diff,
                    atol + rtol * b_abs
                ),
                "".to_string(),
            ));
        }
    }
    Ok(())
}

/// Returns True if two arrays are element-wise equal within a tolerance.
pub(crate) fn check_allclose<T>(
    actual: &Array<T>,
    desired: &Array<T>,
    rtol: T::Real,
    atol: T::Real,
) -> bool
where
    T: LinalgScalar,
{
    if actual.shape() != desired.shape() {
        return false;
    }

    for (a, b) in actual.iter().zip(desired.iter()) {
        let diff = (*a - *b).abs();
        let b_abs = b.abs();
        if diff > atol + rtol * b_abs {
            return false;
        }
    }
    true
}

/// Assert that first array is less than second array element-wise.
pub fn assert_array_less<T>(actual: &Array<T>, desired: &Array<T>) -> Result<()>
where
    T: PartialOrd + Debug + Clone + Default + 'static,
{
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    for (i, (a, b)) in actual.iter().zip(desired.iter()).enumerate() {
        if a >= b {
            return Err(NumPyError::value_error(
                format!(
                    "Condition actual < desired failed at index {}: actual={:?}, desired={:?}",
                    i, a, b
                ),
                "".to_string(),
            ));
        }
    }
    Ok(())
}

/// Assert that two scalars are equal.
pub fn assert_equal<T>(actual: &T, desired: &T) -> Result<()>
where
    T: PartialEq + Debug,
{
    if actual != desired {
        return Err(NumPyError::value_error(
            format!(
                "Items are not equal: actual={:?}, desired={:?}",
                actual, desired
            ),
            "".to_string(),
        ));
    }
    Ok(())
}

/// Assert that two scalars are equal within a certain tolerance.
pub fn assert_almost_equal<T>(actual: &T, desired: &T, decimal: usize) -> Result<()>
where
    T: Float + Debug,
{
    let threshold = T::from(10.0).unwrap().powi(-(decimal as i32));
    if (*actual - *desired).abs() > threshold {
        return Err(NumPyError::value_error(
            format!(
                "Items differ by more than {} decimal places: actual={:?}, desired={:?}",
                decimal, actual, desired
            ),
            "".to_string(),
        ));
    }
    Ok(())
}

/// Assert that two strings are equal.
pub fn assert_string_equal(actual: &str, desired: &str) -> Result<()> {
    if actual != desired {
        return Err(NumPyError::value_error(
            format!(
                "Strings are not equal: actual={:?}, desired={:?}",
                actual, desired
            ),
            "".to_string(),
        ));
    }
    Ok(())
}

/// Assert that two arrays have the same shape.
pub fn assert_array_shape_equal<T, U>(a: &Array<T>, b: &Array<U>) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(NumPyError::shape_mismatch(
            b.shape().to_vec(),
            a.shape().to_vec(),
        ));
    }
    Ok(())
}

/// Assert that works in release mode.
///
/// This is a simple assertion function that always checks the condition,
/// unlike debug assertions that may be compiled out in release builds.
pub fn assert_(val: bool) -> Result<()> {
    if !val {
        return Err(NumPyError::value_error(
            "Assertion failed".to_string(),
            "".to_string(),
        ));
    }
    Ok(())
}

/// Assert that works in release mode with custom message.
pub fn assert_with_msg(val: bool, msg: &str) -> Result<()> {
    if !val {
        return Err(NumPyError::value_error(
            format!("Assertion failed: {}", msg),
            "".to_string(),
        ));
    }
    Ok(())
}

/// Assert that two numbers are equal up to significant digits.
///
/// This is similar to assert_almost_equal but uses significant digits
/// rather than decimal places.
pub fn assert_approx_equal<T>(actual: T, desired: T, significant: usize) -> Result<()>
where
    T: Float + Debug,
{
    if actual.is_zero() && desired.is_zero() {
        return Ok(());
    }

    let diff = (actual - desired).abs();
    let max_val = actual.abs().max(desired.abs()).max(T::from(1.0).unwrap());

    if diff / max_val > T::from(10.0).unwrap().powi(-(significant as i32)) {
        return Err(NumPyError::value_error(
            format!(
                "Items differ by more than {} significant digits: actual={:?}, desired={:?}",
                significant, actual, desired
            ),
            "".to_string(),
        ));
    }
    Ok(())
}

/// Fail unless an exception of class exception_class is thrown by callable when invoked.
///
/// This is a simplified version that works with Rust's panic mechanism.
pub fn assert_raises<F, R>(exception_class: &str, callable: F) -> Result<()>
where
    F: FnOnce() -> R + panic::UnwindSafe,
{
    let result = catch_unwind(AssertUnwindSafe(callable));

    match result {
        Ok(_) => Err(NumPyError::value_error(
            format!(
                "Expected exception '{}' but none was thrown",
                exception_class
            ),
            "".to_string(),
        )),
        Err(_) => Ok(()), // Exception was thrown as expected
    }
}

/// Fail unless an exception of class exception_class and with message that matches
/// expected_regexp is thrown by callable when invoked.
///
/// This is a simplified version that works with Rust's panic mechanism.
pub fn assert_raises_regex<F, R>(
    exception_class: &str,
    expected_regexp: &str,
    callable: F,
) -> Result<()>
where
    F: FnOnce() -> R + panic::UnwindSafe,
{
    let result = catch_unwind(AssertUnwindSafe(callable));

    match result {
        Ok(_) => Err(NumPyError::value_error(
            format!(
                "Expected exception '{}' with message matching '{}' but none was thrown",
                exception_class, expected_regexp
            ),
            "".to_string(),
        )),
        Err(panic_info) => {
            // In a real implementation, we'd extract the panic message and check regex
            // For now, we just accept any panic
            Ok(())
        }
    }
}

/// Fail unless the given callable throws the specified warning.
///
/// This is a placeholder implementation since Rust doesn't have the same
/// warning system as Python. In practice, this would need custom warning handling.
pub fn assert_warns<F, R>(_warning_class: &str, callable: F) -> Result<R>
where
    F: FnOnce() -> R,
{
    // Placeholder: Rust doesn't have Python-style warnings
    // This would need a custom warning system
    let result = callable();
    Ok(result)
}

/// Fail if the given callable produces any warnings.
///
/// This is a placeholder implementation since Rust doesn't have the same
/// warning system as Python.
pub fn assert_no_warnings<F, R>(callable: F) -> Result<R>
where
    F: FnOnce() -> R,
{
    // Placeholder: Rust doesn't have Python-style warnings
    // This would need a custom warning system
    let result = callable();
    Ok(result)
}

/// Fail if the given callable produces any reference cycles.
///
/// This is a placeholder implementation since Rust's ownership system
/// prevents most reference cycles by design.
pub fn assert_no_gc_cycles<F, R>(callable: F) -> Result<R>
where
    F: FnOnce() -> R,
{
    // Placeholder: Rust's ownership system prevents most cycles
    let result = callable();
    Ok(result)
}

/// Test if two objects are equal, and print an error message if test fails.
pub fn print_assert_equal<T>(test_string: &str, actual: &T, desired: &T) -> Result<()>
where
    T: PartialEq + Debug,
{
    if actual != desired {
        let error_msg = format!(
            "Test '{}': Objects are not equal:\n  actual: {:?}\n  desired: {:?}",
            test_string, actual, desired
        );
        println!("{}", error_msg);
        return Err(NumPyError::value_error(error_msg, "".to_string()));
    }

    println!("Test '{}': PASSED", test_string);
    Ok(())
}

/// Assert that two arrays are equal within tolerance using floating point spacing.
///
/// This compares arrays relatively to their spacing (NULP = Number of Units in the Last Place).
pub fn assert_array_almost_equal_nulp<T>(x: &Array<T>, y: &Array<T>, nulp: usize) -> Result<()>
where
    T: Float + Debug + 'static,
{
    if x.shape() != y.shape() {
        return Err(NumPyError::shape_mismatch(
            y.shape().to_vec(),
            x.shape().to_vec(),
        ));
    }

    for (i, (a, b)) in x.iter().zip(y.iter()).enumerate() {
        if a.is_nan() && b.is_nan() {
            continue; // Both NaN is considered equal
        }

        if a.is_infinite() || b.is_infinite() {
            if a != b {
                return Err(NumPyError::value_error(
                    format!(
                        "Arrays differ at index {}: actual={:?}, desired={:?} (infinite values)",
                        i, a, b
                    ),
                    "".to_string(),
                ));
            }
            continue;
        }

        // Simplified NULP comparison - in practice this would need bit-level manipulation
        let diff = (*a - *b).abs();
        let epsilon = T::from(f64::EPSILON).unwrap();
        let scale = a.abs().max(b.abs()).max(T::from(1.0).unwrap());
        let threshold = epsilon * scale * T::from(nulp as f64).unwrap();

        if diff > threshold {
            return Err(NumPyError::value_error(
                format!(
                    "Arrays differ at index {} by more than {} NULP: actual={:?}, desired={:?}",
                    i, nulp, a, b
                ),
                "".to_string(),
            ));
        }
    }

    Ok(())
}

/// Check that all items of arrays differ in at most N Units in the Last Place.
pub fn assert_array_max_ulp<T>(a: &Array<T>, b: &Array<T>, maxulp: usize) -> Result<()>
where
    T: Float + Debug + 'static,
{
    if a.shape() != b.shape() {
        return Err(NumPyError::shape_mismatch(
            b.shape().to_vec(),
            a.shape().to_vec(),
        ));
    }

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.is_nan() && y.is_nan() {
            continue; // Both NaN is considered equal
        }

        if x.is_infinite() || y.is_infinite() {
            if x != y {
                return Err(NumPyError::value_error(
                    format!(
                        "Arrays differ at index {}: actual={:?}, desired={:?} (infinite values)",
                        i, x, y
                    ),
                    "".to_string(),
                ));
            }
            continue;
        }

        // Simplified ULP comparison - in practice this would need bit-level manipulation
        let diff = (*x - *y).abs();
        let epsilon = T::from(f64::EPSILON).unwrap();
        let scale = x.abs().max(y.abs()).max(T::from(1.0).unwrap());
        let threshold = epsilon * scale * T::from(maxulp as f64).unwrap();

        if diff > threshold {
            return Err(NumPyError::value_error(
                format!(
                    "Arrays differ at index {} by more than {} ULP: actual={:?}, desired={:?}",
                    i, maxulp, x, y
                ),
                "".to_string(),
            ));
        }
    }

    Ok(())
}

pub mod exports {
    pub use super::{
        assert_, assert_allclose, assert_almost_equal, assert_approx_equal,
        assert_array_almost_equal, assert_array_almost_equal_nulp, assert_array_equal,
        assert_array_less, assert_array_max_ulp, assert_array_shape_equal, assert_equal,
        assert_no_gc_cycles, assert_no_warnings, assert_raises, assert_raises_regex,
        assert_string_equal, assert_warns, assert_with_msg, print_assert_equal,
    };
}
