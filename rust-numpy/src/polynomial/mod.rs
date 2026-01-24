// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Polynomial operations
//!
//! This module provides complete compatibility with NumPy's polynomial module,
//! including all polynomial classes and operations.

pub mod chebyshev;
pub mod exports;
pub mod hermite;
pub mod hermite_e;
pub mod laguerre;
pub mod legendre;
pub mod polynomial;

// Re-export main types for convenience
pub use exports::*;

/// Common polynomial operations that work with all polynomial types
pub trait PolynomialBase<T> {
    /// Get polynomial coefficients
    fn coeffs(&self) -> &ndarray::Array1<T>;

    /// Get polynomial degree
    fn degree(&self) -> usize {
        self.coeffs().len() - 1
    }

    /// Evaluate polynomial at given points
    fn eval(&self, x: &ndarray::Array1<T>) -> Result<ndarray::Array1<T>, crate::error::NumPyError>;

    /// Compute derivative
    fn deriv(&self, m: usize) -> Result<Self, NumPyError>
    where
        Self: Sized;

    /// Compute integral with optional constant term
    fn integ(&self, m: usize, k: Option<T>) -> Result<Self, NumPyError>
    where
        Self: Sized;

    /// Get polynomial domain
    fn domain(&self) -> [T; 2];

    /// Get polynomial window
    fn window(&self) -> [T; 2];

    /// Convert to standard polynomial coefficients
    fn to_polynomial(&self) -> Result<Polynomial<T>, NumPyError>;
}

use crate::error::NumPyError;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global print style preference for polynomials
/// true = unicode, false = ascii
static POLY_PRINT_UNICODE: AtomicBool = AtomicBool::new(cfg!(unix));

/// Set the default format for the string representation of polynomials.
///
/// Values for `style` must be valid inputs to `__format__`, i.e. 'ascii'
/// or 'unicode'.
///
/// # Arguments
///
/// * `style` - Format string for default printing style. Must be either 'ascii' or 'unicode'.
///
/// # Errors
///
/// Returns an error if the style is not 'ascii' or 'unicode'.
///
/// # Examples
///
/// ```ignore
/// use numpy::polynomial::set_default_printstyle;
///
/// // Use unicode superscripts (default on Unix)
/// set_default_printstyle("unicode").unwrap();
///
/// // Use ascii format (e.g., x**2 instead of x²)
/// set_default_printstyle("ascii").unwrap();
/// ```
pub fn set_default_printstyle(style: &str) -> Result<(), NumPyError> {
    match style {
        "unicode" => {
            POLY_PRINT_UNICODE.store(true, Ordering::SeqCst);
            Ok(())
        }
        "ascii" => {
            POLY_PRINT_UNICODE.store(false, Ordering::SeqCst);
            Ok(())
        }
        _ => Err(NumPyError::invalid_value(&format!(
            "Unsupported format string '{}'. Valid options are 'ascii' and 'unicode'",
            style
        ))),
    }
}

/// Get the current print style preference
/// true = unicode, false = ascii
pub(crate) fn get_print_style() -> bool {
    POLY_PRINT_UNICODE.load(Ordering::SeqCst)
}

/// Polynomial fitting function
pub fn fit<T>(
    x: &ndarray::Array1<T>,
    y: &ndarray::Array1<T>,
    deg: usize,
) -> Result<Polynomial<T>, NumPyError>
where
    T: num_traits::Float + num_traits::Num + std::fmt::Debug + 'static,
{
    polynomial::fit(x, y, deg)
}

/// Root finding for polynomials
pub fn roots<T>(p: &Polynomial<T>) -> Result<ndarray::Array1<num_complex::Complex<T>>, NumPyError>
where
    T: num_traits::Float + num_traits::Num + std::fmt::Debug + 'static,
{
    polynomial::roots(p)
}

/// Polynomial evaluation
pub fn val<T>(p: &Polynomial<T>, x: &ndarray::Array1<T>) -> Result<ndarray::Array1<T>, NumPyError>
where
    T: num_traits::Float + num_traits::Num + std::fmt::Debug + 'static,
{
    p.eval(x)
}

/// Derivative computation
pub fn deriv<T>(p: &Polynomial<T>, m: usize) -> Result<Polynomial<T>, NumPyError>
where
    T: num_traits::Float + num_traits::Num + std::fmt::Debug + 'static,
{
    p.deriv(m)
}

/// Integration computation
pub fn integ<T>(p: &Polynomial<T>, m: usize, k: Option<T>) -> Result<Polynomial<T>, NumPyError>
where
    T: num_traits::Float + num_traits::Num + std::fmt::Debug + 'static,
{
    p.integ(m, k)
}

/// Companion matrix generation
pub fn companion<T>(p: &Polynomial<T>) -> Result<ndarray::Array2<T>, NumPyError>
where
    T: num_traits::Float + num_traits::Num + std::fmt::Debug + 'static,
{
    polynomial::companion(p)
}

/// Domain management
pub fn domain<T>(p: &Polynomial<T>, window: Option<&[T; 2]>) -> Result<[T; 2], NumPyError>
where
    T: num_traits::Float + num_traits::Num + std::fmt::Debug + 'static,
{
    polynomial::domain(p, window)
}
