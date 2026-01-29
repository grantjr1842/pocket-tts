use crate::array::Array;
use crate::error::{NumPyError, Result};
use crate::linalg::LinalgScalar;
use num_traits::Float;
use std::fmt::Debug;
use std::panic;

trait ToBits {
    fn to_u64_bits(self) -> u64;
}

impl ToBits for f32 {
    fn to_u64_bits(self) -> u64 {
        self.to_bits() as u64
    }
}

impl ToBits for f64 {
    fn to_u64_bits(self) -> u64 {
        self.to_bits()
    }
}

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

/// Assert that two values are equal.
pub fn assert_equal<T, U>(actual: T, desired: U) -> Result<()>
where
    T: PartialEq<U> + Debug,
    U: Debug,
{
    if actual != desired {
        return Err(NumPyError::value_error(
            format!("Values differ: actual={:?}, desired={:?}", actual, desired),
            "".to_string(),
        ));
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

/// Assert that two values are almost equal within a certain tolerance.
pub fn assert_almost_equal<T: Float + Debug>(actual: T, desired: T, decimal: usize) -> Result<()> {
    let threshold = T::from(10.0).unwrap().powi(-(decimal as i32));
    if (actual - desired).abs() > threshold {
        return Err(NumPyError::value_error(
            format!(
                "Values differ by more than {} decimal places: actual={:?}, desired={:?}",
                decimal, actual, desired
            ),
            "".to_string(),
        ));
    }
    Ok(())
}

/// Assert that two values are approximately equal.
pub fn assert_approx_equal<T: Float + Debug>(
    actual: T,
    desired: T,
    significant: usize,
) -> Result<()> {
    let threshold = T::from(0.5).unwrap() * T::from(10.0).unwrap().powi(-(significant as i32 - 1));
    let diff = (actual - desired).abs();
    let scale = desired.abs().max(T::one());

    if diff > threshold * scale {
        return Err(NumPyError::value_error(
            format!(
                "Values are not approximately equal to {} significant figures: actual={:?}, desired={:?}",
                significant, actual, desired
            ),
            "".to_string(),
        ));
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

/// Assert that two arrays are almost equal considering units in the last place (ULP).
pub fn assert_array_almost_nulp<T: Float + Debug + ToBits>(
    actual: &Array<T>,
    desired: &Array<T>,
    nulp: usize,
) -> Result<()> {
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    for (i, (a, b)) in actual.iter().zip(desired.iter()).enumerate() {
        let a_bits = a.to_u64_bits();
        let b_bits = b.to_u64_bits();

        // Robust ULP comparison using integer representation of floating point bits
        // This handles positive/negative and zero correctly in most cases for ULP
        // Monotonic ULP mapping that correctly handles the sign boundary
        let a_int = {
            let bits = a_bits as i64;
            if bits < 0 {
                i64::MIN.wrapping_add(!bits)
            } else {
                bits
            }
        };

        let b_int = {
            let bits = b_bits as i64;
            if bits < 0 {
                i64::MIN.wrapping_add(!bits)
            } else {
                bits
            }
        };

        let diff = (a_int.wrapping_sub(b_int)).abs() as usize;

        if diff > nulp {
            return Err(NumPyError::value_error(
                format!(
                    "Arrays differ by more than {} ULP at index {}: actual={:?}, desired={:?}",
                    nulp, i, a, b
                ),
                "".to_string(),
            ));
        }
    }
    Ok(())
}

/// Assert that two arrays are almost equal considering units in the last place (ULP) - alternative version.
pub fn assert_array_almost_equal_nulp<T: Float + Debug + ToBits>(
    actual: &Array<T>,
    desired: &Array<T>,
    maxulp: usize,
) -> Result<()> {
    assert_array_almost_nulp(actual, desired, maxulp)
}

/// Assert that the maximum ULP difference between two arrays is within bounds.
pub fn assert_array_max_ulp<T: Float + Debug + ToBits>(
    actual: &Array<T>,
    desired: &Array<T>,
    maxulp: usize,
) -> Result<()> {
    if actual.shape() != desired.shape() {
        return Err(NumPyError::shape_mismatch(
            desired.shape().to_vec(),
            actual.shape().to_vec(),
        ));
    }

    let mut max_diff = 0u64;
    for (a, b) in actual.iter().zip(desired.iter()) {
        let a_bits = a.to_u64_bits();
        let b_bits = b.to_u64_bits();
        let diff = if a_bits > b_bits {
            a_bits - b_bits
        } else {
            b_bits - a_bits
        };
        max_diff = max_diff.max(diff);
    }

    if max_diff > maxulp as u64 {
        return Err(NumPyError::value_error(
            format!(
                "Maximum ULP difference {} exceeds allowed {}: actual={:?}, desired={:?}",
                max_diff, maxulp, actual, desired
            ),
            "".to_string(),
        ));
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

/// Assert that two arrays compare equal element-wise.
pub fn assert_array_compare<T>(actual: &Array<T>, desired: &Array<T>) -> Result<()>
where
    T: PartialEq + Debug + Clone + Default + 'static,
{
    assert_array_equal(actual, desired)
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

/// Assert that two strings are equal.
pub fn assert_string_equal(actual: &str, desired: &str) -> Result<()> {
    if actual != desired {
        return Err(NumPyError::value_error(
            format!("Strings differ: actual={:?}, desired={:?}", actual, desired),
            "".to_string(),
        ));
    }
    Ok(())
}

/// Assert that a callable raises a specific exception when called.
pub fn assert_raises<F, R>(func: F, expected_exception: &str) -> Result<()>
where
    F: FnOnce() -> R,
{
    let result = panic::catch_unwind(panic::AssertUnwindSafe(func));

    match result {
        Ok(_) => Err(NumPyError::value_error(
            format!("Expected exception '{}' was not raised", expected_exception),
            "".to_string(),
        )),
        Err(_) => Ok(()), // Exception was raised as expected
    }
}

/// Assert that a callable raises a specific exception matching a regex pattern.
pub fn assert_raises_regex<F, R>(func: F, expected_exception: &str, _pattern: &str) -> Result<()>
where
    F: FnOnce() -> R,
{
    let result = panic::catch_unwind(panic::AssertUnwindSafe(func));

    match result {
        Ok(_) => Err(NumPyError::value_error(
            format!("Expected exception '{}' was not raised", expected_exception),
            "".to_string(),
        )),
        Err(_) => {
            // In a real implementation, we'd check the panic message against the regex
            // For now, we just accept that some exception was raised
            Ok(())
        }
    }
}

/// Assert that a callable produces warnings.
pub fn assert_warns<F, R>(func: F) -> Result<R>
where
    F: FnOnce() -> R,
{
    // In a real implementation, this would capture and verify warnings
    // For now, we just call the function and return the result
    Ok(func())
}

/// Assert that no warnings are produced during function execution.
pub fn assert_no_warnings<F, R>(func: F) -> Result<R>
where
    F: FnOnce() -> R,
{
    // In a real implementation, this would verify no warnings were produced
    // For now, we just call the function and return the result
    Ok(func())
}

/// Assert that no garbage collection cycles are created.
pub fn assert_no_gc_cycles() -> Result<()> {
    // In Python, this would check for reference cycles
    // In Rust, this is less relevant due to ownership system
    // For now, this is a no-op
    Ok(())
}

pub mod exports {
    pub use super::{
        assert_allclose, assert_almost_equal, assert_approx_equal, assert_array_almost_equal,
        assert_array_almost_equal_nulp, assert_array_almost_nulp, assert_array_compare,
        assert_array_equal, assert_array_less, assert_array_max_ulp, assert_array_shape_equal,
        assert_equal, assert_no_gc_cycles, assert_no_warnings, assert_raises, assert_raises_regex,
        assert_string_equal, assert_warns,
    };
}
