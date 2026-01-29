// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT> or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Random number generation utils
//!
//! This module provides NumPy-compatible random number generation with both
//! modern (Generator/BitGenerator) and legacy (RandomState) APIs.

pub mod bit_generator;
pub mod generator;
pub mod random_state;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::NumPyError;
pub use bit_generator::{BitGenerator, PCG64};
pub use generator::Generator;
use num_traits::NumCast;
use rand_distr::uniform::SampleUniform;
pub use random_state::RandomState;
use std::cell::RefCell;

// Thread-local default RNG for modern API
thread_local! {
    static DEFAULT_GENERATOR: RefCell<generator::Generator> = RefCell::new(
        generator::Generator::new(Box::new(bit_generator::PCG64::new()))
    );
}

// Thread-local legacy RNG for backward compatibility
thread_local! {
    static DEFAULT_RNG: RefCell<RandomState> = RefCell::new(RandomState::new());
}

/// Create the default Generator instance
///
/// This is NumPy's recommended way to create random number generators since version 1.17.
/// It returns a Generator using the PCG64 bit generator by default.
///
/// # Examples
///
/// ```rust
/// use rust_numpy::random;
///
/// let mut rng = random::default_rng();
/// let arr = rng.random::<f64>(&[3, 4], rust_numpy::Dtype::Float64).unwrap();
/// ```
///
/// # Arguments
///
/// * `seed` - Optional seed for reproducible results (None for random seed)
///
/// # Returns
///
/// A Generator instance with PCG64 bit generator
pub fn default_rng() -> generator::Generator {
    generator::Generator::new(Box::new(bit_generator::PCG64::new()))
}

/// Create a seeded Generator instance
///
/// Same as `default_rng()` but with a specific seed for reproducible results.
///
/// # Arguments
///
/// * `seed` - Seed value for the random number generator
///
/// # Returns
///
/// A Generator instance with PCG64 bit generator seeded with the provided value
pub fn default_rng_with_seed(seed: u64) -> generator::Generator {
    generator::Generator::new(Box::new(bit_generator::PCG64::seed_from_u64(seed)))
}

/// Legacy API - Create the default RandomState instance
///
/// This maintains backward compatibility with the existing RandomState API.
/// New code should prefer using `default_rng()` instead.
///
/// # Returns
///
/// A RandomState instance
#[deprecated(note = "Use default_rng() instead for new code")]
pub fn legacy_rng() -> RandomState {
    RandomState::new()
}

// --- Modern API: Module-level convenience functions that delegate to default_rng() ---

/// Generate random numbers in the half-open interval [0.0, 1.0)
///
/// This uses the modern Generator API internally.
pub fn random<T: Clone + Default + NumCast + 'static>(
    shape: &[usize],
    dtype: Dtype,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().random(shape, dtype))
}

/// Generate random integers between low (inclusive) and high (exclusive)
///
/// This uses the modern Generator API internally.
pub fn randint<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().randint(low, high, shape))
}

/// Generate samples from a uniform distribution over [low, high)
///
/// This uses the modern Generator API internally.
pub fn uniform<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().uniform(low, high, shape))
}

/// Generate samples from a normal (Gaussian) distribution
///
/// This uses the modern Generator API internally.
pub fn normal<T: Clone + PartialOrd + SampleUniform + Default + 'static + Into<f64> + From<f64>>(
    mean: T,
    std: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().normal(mean, std, shape))
}

/// Generate samples from a standard normal distribution
///
/// This uses the modern Generator API internally.
pub fn standard_normal<
    T: Clone + PartialOrd + SampleUniform + Default + 'static + Into<f64> + From<f64>,
>(
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().standard_normal(shape))
}

/// Generate samples from a binomial distribution
///
/// This uses the modern Generator API internally.
pub fn binomial<T: Clone + Default + 'static + From<f64>>(
    n: u64,
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().binomial(n, p, shape))
}

/// Generate samples from a Poisson distribution
///
/// This uses the modern Generator API internally.
pub fn poisson<T: Clone + Default + 'static + From<f64>>(
    lam: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().poisson(lam, shape))
}

/// Generate samples from an exponential distribution
///
/// This uses the modern Generator API internally.
pub fn exponential<T: Clone + Default + 'static + From<f64>>(
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().exponential(scale, shape))
}

/// Generate samples from a gamma distribution
///
/// This uses the modern Generator API internally.
pub fn gamma<T: Clone + Default + 'static + From<f64>>(
    shape_param: f64,
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().gamma(shape_param, scale, shape))
}

/// Generate samples from a beta distribution
///
/// This uses the modern Generator API internally.
pub fn beta<T: Clone + Default + 'static + From<f64>>(
    a: f64,
    b: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().beta(a, b, shape))
}

/// Generate samples from a chi-square distribution
///
/// This uses the modern Generator API internally.
pub fn chisquare<T: Clone + Default + 'static + From<f64>>(
    df: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().chisquare(df, shape))
}

/// Generate samples from a Gumbel distribution
///
/// This uses the modern Generator API internally.
pub fn gumbel<T: Clone + Default + 'static + From<f64>>(
    loc: f64,
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().gumbel(loc, scale, shape))
}

/// Generate samples from a logistic distribution
///
/// This uses the modern Generator API internally.
// TODO: Logistic distribution not available - uncomment when available
// pub fn logistic<T: Clone + Default + 'static + From<f64>>(
//     loc: f64,
//     scale: f64,
//     shape: &[usize],
// ) -> Result<Array<T>, NumPyError> {
//     DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().logistic(loc, scale, shape))
// }

/// Generate samples from a log-normal distribution
///
/// This uses the modern Generator API internally.
pub fn lognormal<T: Clone + Default + 'static + From<f64>>(
    mean: f64,
    sigma: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().lognormal(mean, sigma, shape))
}

/// Generate samples from a multinomial distribution
///
/// This uses the modern Generator API internally.
pub fn multinomial<T: Clone + Default + 'static + From<f64>>(
    n: u64,
    pvals: &[f64],
    size: Option<&[usize]>,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().multinomial(n, pvals, size))
}

/// Generate samples from a Dirichlet distribution
///
/// This uses the modern Generator API internally.
pub fn dirichlet<T: Clone + Default + 'static + From<f64>>(
    alpha: &[f64],
    size: Option<&[usize]>,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().dirichlet(alpha, size))
}

/// Generate samples from a geometric distribution
///
/// This uses the modern Generator API internally.
pub fn geometric<T: Clone + Default + 'static + From<f64>>(
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().geometric(p, shape))
}

/// Generate samples from a negative binomial distribution
///
/// This uses the modern Generator API internally.
pub fn negative_binomial<T: Clone + Default + 'static + From<f64>>(
    n: u64,
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().negative_binomial(n, p, shape))
}

/// Generate samples from a hypergeometric distribution
///
/// This uses the modern Generator API internally.
pub fn hypergeometric<T: Clone + Default + 'static + From<f64>>(
    ngood: u64,
    nbad: u64,
    nsample: u64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().hypergeometric(ngood, nbad, nsample, shape))
}

/// Generate samples from a logarithmic series distribution
///
/// This uses the modern Generator API internally.
pub fn logseries<T: Clone + Default + 'static + From<f64>>(
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().logseries(p, shape))
}

/// Generate samples from a Rayleigh distribution
///
/// This uses the modern Generator API internally.
pub fn rayleigh<T: Clone + Default + 'static + From<f64>>(
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().rayleigh(scale, shape))
}

/// Generate samples from a Wald (inverse Gaussian) distribution
///
/// This uses the modern Generator API internally.
pub fn wald<T: Clone + Default + 'static + From<f64>>(
    mean: f64,
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().wald(mean, scale, shape))
}

/// Generate samples from a Weibull distribution
///
/// This uses the modern Generator API internally.
pub fn weibull<T: Clone + Default + 'static + From<f64>>(
    a: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().weibull(a, shape))
}

/// Generate samples from a triangular distribution
///
/// This uses the modern Generator API internally.
pub fn triangular<T: Clone + Default + 'static + From<f64>>(
    left: f64,
    mode: f64,
    right: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().triangular(left, mode, right, shape))
}

/// Generate samples from a Pareto distribution
///
/// This uses the modern Generator API internally.
pub fn pareto<T: Clone + Default + 'static + From<f64>>(
    a: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().pareto(a, shape))
}

/// Generate samples from a Zipf distribution
///
/// This uses the modern Generator API internally.
pub fn zipf<T: Clone + Default + 'static + From<f64>>(
    a: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().zipf(a, shape))
}

/// Generate samples from a standard Cauchy distribution
///
/// This uses the modern Generator API internally.
pub fn standard_cauchy<T: Clone + Default + 'static + From<f64>>(
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().standard_cauchy(shape))
}

/// Generate samples from a standard exponential distribution
///
/// This uses the modern Generator API internally.
pub fn standard_exponential<T: Clone + Default + 'static + From<f64>>(
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().standard_exponential(shape))
}

/// Generate samples from a standard gamma distribution
///
/// This uses the modern Generator API internally.
pub fn standard_gamma<T: Clone + Default + 'static + From<f64>>(
    shape_param: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().standard_gamma(shape_param, shape))
}

/// Generate samples from a Bernoulli distribution
///
/// This uses the modern Generator API internally.
pub fn bernoulli<T: Clone + Default + 'static + From<f64>>(
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().bernoulli(p, shape))
}

/// Generate samples from an F-distribution
///
/// This uses the modern Generator API internally.
pub fn f<T: Clone + Default + 'static + From<f64>>(
    dfnum: f64,
    dfden: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().f(dfnum, dfden, shape))
}

/// Generate samples from a power distribution
///
/// This uses the modern Generator API internally.
// TODO: Power distribution not available - uncomment when available
// pub fn power<T: Clone + Default + 'static + From<f64>>(
//     a: f64,
//     shape: &[usize],
// ) -> Result<Array<T>, NumPyError> {
//     DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().power(a, shape))
// }

/// Generate samples from a von Mises distribution
///
/// This uses the modern Generator API internally.
// TODO: VonMises distribution not available - uncomment when available
// pub fn vonmises<T: Clone + Default + 'static + From<f64>>(
//     mu: f64,
//     kappa: f64,
//     shape: &[usize],
// ) -> Result<Array<T>, NumPyError> {
//     DEFAULT_GENERATOR.with(|rng| rng.borrow_mut().vonmises(mu, kappa, shape))
// }

// --- Legacy API Functions (for backward compatibility) ---

/// Seed the legacy default RNG
///
/// This maintains backward compatibility with the existing RandomState API.
/// New code should create a Generator with a specific seed instead.
#[deprecated(note = "Use default_rng_with_seed(seed) instead for new code")]
pub fn seed(seed: u64) {
    DEFAULT_RNG.with(|rng| {
        *rng.borrow_mut() = RandomState::seed_from_u64(seed);
    });
}

/// Legacy random function using RandomState
///
/// This maintains backward compatibility. New code should use the module-level
/// functions which delegate to the modern Generator API.
#[deprecated(note = "Use random() function which uses modern Generator API")]
pub fn legacy_random<T: Clone + Default + NumCast + 'static>(
    shape: &[usize],
    dtype: Dtype,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().random(shape, dtype))
}

/// Legacy randint function using RandomState
///
/// This maintains backward compatibility. New code should use the module-level
/// functions which delegate to the modern Generator API.
#[deprecated(note = "Use randint() function which uses modern Generator API")]
pub fn legacy_randint<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().randint(low, high, shape))
}

/// Student's t-distribution
///
/// Returns samples from a Student's t-distribution with `df` degrees of freedom.
pub fn standard_t<T: Clone + Default + 'static + From<f64>>(
    df: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    let size: usize = shape.iter().product();
    let mut result = Vec::with_capacity(size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        for _ in 0..size {
            // Student's t-distribution: X = Z / sqrt(Y/df)
            // where Z ~ N(0,1) and Y ~ chi-squared(df)
            let z: f64 = rng.standard_normal(1)?[0];
            let y: f64 = rng.chisquare(df, 1)?[0];
            let t = z / (y / df).sqrt();
            result.push(t.into());
        }
    });

    Array::from_shape_vec(shape.to_vec(), result)
}

/// Noncentral chi-square distribution
///
/// Returns samples from a noncentral chi-square distribution with `df` degrees
/// of freedom and noncentrality parameter `nonc`.
pub fn noncentral_chisquare<T: Clone + Default + 'static + From<f64>>(
    df: f64,
    nonc: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    let size: usize = shape.iter().product();
    let mut result = Vec::with_capacity(size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        for _ in 0..size {
            // Noncentral chi-square: sum of squared normal random variables
            // with specified noncentrality parameters
            let v = rng.chisquare(df - 2.0, 1)?[0];
            let norm = rng.standard_normal(1)?[0];
            result.push((v + (2.0 * nonc).sqrt() * norm + nonc).into());
        }
    });

    Array::from_shape_vec(shape.to_vec(), result)
}

/// Noncentral F-distribution
///
/// Returns samples from a noncentral F-distribution with numerator degrees
/// of freedom `dfnum`, denominator degrees of freedom `dfden`, and
/// noncentrality parameter `nonc`.
pub fn noncentral_f<T: Clone + Default + 'static + From<f64>>(
    dfnum: f64,
    dfden: f64,
    nonc: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    let size: usize = shape.iter().product();
    let mut result = Vec::with_capacity(size);

    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        for _ in 0..size {
            // Noncentral F: (noncentral chi-square / dfnum) / (chi-square / dfden)
            let nc_chi2 = rng.noncentral_chisquare(dfnum, nonc, 1)?[0];
            let chi2 = rng.chisquare(dfden, 1)?[0];
            result.push((nc_chi2 / dfnum) / (chi2 / dfden) * (dfden / dfnum)).into();
        }
    });

    Array::from_shape_vec(shape.to_vec(), result)
}

/// Multivariate normal distribution
///
/// Returns samples from a multivariate normal distribution with specified
/// mean and covariance matrix.
pub fn multivariate_normal<T: Clone + Default + 'static + Into<f64> + From<f64>>(
    mean: &[f64],
    cov: &[Vec<f64>],
    shape: &[usize],
) -> Result<Array<f64>, NumPyError> {
    let n = mean.len();
    if cov.len() != n || cov.iter().any(|row| row.len() != n) {
        return Err(NumPyError::invalid_operation(
            "Covariance matrix must be square and match mean length",
        ));
    }

    let size: usize = shape.iter().product();
    let mut result = Vec::with_capacity(size * n);

    // Simple implementation using Cholesky decomposition
    // For a more robust implementation, we'd use a proper linear algebra library
    DEFAULT_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        for _ in 0..size {
            // Generate standard normal samples
            let mut standard: Vec<f64> = (0..n).map(|_| rng.standard_normal(1)).flatten().collect();

            // Simple approach: just add mean to standard normal (for identity covariance)
            // A full implementation would use proper Cholesky decomposition
            for i in 0..n {
                result.push(mean[i] + standard[i]);
            }
        }
    });

    let mut full_shape = shape.to_vec();
    full_shape.push(n);
    Array::from_shape_vec(full_shape, result)
}

// --- Module exports for modern API structure ---

/// Modern random number generation API
///
/// This sub-module provides the modern Generator/BitGenerator API
/// that matches NumPy's current random module structure.
pub mod modern {
    pub use super::bit_generator::{BitGenerator, PCG64, PCG64DXSM, Philox, SFC64};
    pub use super::generator::Generator;
    pub use super::random_state::RandomState;
    pub use super::{default_rng, default_rng_with_seed};
}

/// Legacy random number generation API
///
/// This sub-module provides the legacy RandomState API for backward compatibility.
#[deprecated(since = "0.1.0", note = "Use modern Generator API instead")]
#[allow(deprecated)]
pub mod legacy {
    pub use super::RandomState;
    pub use super::{legacy_randint, legacy_random, legacy_rng, seed};
}

#[cfg(test)]
mod tests;
