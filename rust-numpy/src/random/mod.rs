// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Random number generation utils
//!
//! This module provides NumPy-compatible random number generation with both
//! modern (Generator/BitGenerator) and legacy (RandomState) APIs.

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::NumPyError;
use num_traits::NumCast;
use rand_distr::uniform::SampleUniform;
use std::cell::RefCell;
use std::thread_local;

pub mod bit_generator;
pub mod generator;
pub mod random_state;

// Re-export BitGenerators
pub use bit_generator::{BitGenerator, PCG64, MT19937, Philox, SFC64, PCG64DXSM, ParallelParams};
pub use generator::Generator;
pub use random_state::RandomState;

// Thread-local default generator for modern API
thread_local! {
    static DEFAULT_GENERATOR: RefCell<Generator> = RefCell::new(default_rng());
}

// Thread-local default RNG for legacy API
thread_local! {
    static DEFAULT_RNG: RefCell<RandomState> = RefCell::new(RandomState::new());
}

/// Create a default random number generator using PCG64
///
/// This creates a new Generator with the default PCG64 BitGenerator.
pub fn default_rng() -> Generator {
    Generator::new(Box::new(PCG64::new()))
}

/// Create a seeded random number generator using PCG64
///
/// This creates a new Generator with a seeded PCG64 BitGenerator.
pub fn default_rng_with_seed(seed: u64) -> Generator {
    Generator::new(Box::new(PCG64::seed_from_u64(seed)))
}

/// Get the legacy default RNG
///
/// This provides access to the thread-local RandomState for backward compatibility.
#[deprecated(note = "Use default_rng() instead for new code")]
pub fn legacy_rng() -> RandomState {
    RandomState::new()
}

// --- Modern API: Module-level convenience functions that delegate to default_rng() ---

/// Generate samples from a logistic distribution
///
/// Logistic distribution is a continuous probability distribution with a characteristic S-shaped curve.
/// It's useful for modeling growth processes and probabilities.
pub fn logistic<T>(loc: f64, scale: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().logistic(loc, scale, shape))
}

/// Generate samples from a power distribution
///
/// Power distribution is a continuous probability distribution with PDF:
/// f(x) = a * x^(a-1) for x in [0, 1], 0 otherwise
pub fn power<T>(a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().power(a, shape))
}

/// Generate samples from a von Mises distribution
///
/// Von Mises distribution is a continuous probability distribution on a circle.
/// It's commonly used for modeling directional statistics.
pub fn vonmises<T>(mu: f64, kappa: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().vonmises(mu, kappa, shape))
}

/// Generate samples from a multivariate normal distribution
///
/// Multivariate normal distribution generates samples from a multivariate
/// normal distribution with specified mean and covariance matrix.
pub fn multivariate_normal<T>(
    mean: &[f64],
    cov: &[f64],
    size: Option<&[usize]>,
) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().multivariate_normal(mean, cov, size))
}

/// Generate samples from a non-central chi-square distribution
///
/// Non-central chi-square distribution is a generalization of chi-square
/// distribution where mean is not equal to the degrees of freedom.
pub fn noncentral_chisquare<T>(df: f64, nonc: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().noncentral_chisquare(df, nonc, shape))
}

/// Generate samples from a non-central F-distribution
///
/// Non-central F-distribution is a generalization of F-distribution
/// where mean is not equal to the degrees of freedom.
pub fn noncentral_f<T>(dfnum: f64, dfden: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().noncentral_f(dfnum, dfden, shape))
}

/// Generate samples from a standard t-distribution
///
/// Student's t-distribution is used for estimating population means
/// when the population standard deviation is unknown.
pub fn standard_t<T>(df: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().standard_t(df, shape))
}

/// Generate random samples from [0.0, 1.0)
pub fn random<T>(shape: &[usize], dtype: crate::dtype::Dtype) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + num_traits::NumCast + 'static,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().random(shape, dtype))
}

/// Generate random integers in [low, high)
pub fn randint<T>(low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + PartialOrd + rand_distr::uniform::SampleUniform + Default + 'static,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().randint(low, high, shape))
}

/// Generate random floats in [low, high)
pub fn uniform<T>(low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + PartialOrd + rand_distr::uniform::SampleUniform + Default + 'static,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().uniform(low, high, shape))
}

/// Generate samples from a normal distribution
pub fn normal<T>(mean: T, std: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + PartialOrd + rand_distr::uniform::SampleUniform + Default + 'static + Into<f64> + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().normal(mean, std, shape))
}

/// Generate samples from a standard normal distribution
pub fn standard_normal<T>(shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + PartialOrd + rand_distr::uniform::SampleUniform + Default + 'static + Into<f64> + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().standard_normal(shape))
}

/// Generate samples from a binomial distribution
pub fn binomial<T>(n: u64, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().binomial(n, p, shape))
}

/// Generate samples from a Poisson distribution
pub fn poisson<T>(lam: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().poisson(lam, shape))
}

/// Generate samples from an exponential distribution
pub fn exponential<T>(scale: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().exponential(scale, shape))
}

/// Generate samples from a gamma distribution
pub fn gamma<T>(shape_param: f64, scale: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().gamma(shape_param, scale, shape))
}

/// Generate samples from a beta distribution
pub fn beta<T>(a: f64, b: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().beta(a, b, shape))
}

/// Generate samples from a chi-square distribution
pub fn chisquare<T>(df: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().chisquare(df, shape))
}

/// Generate samples from a Bernoulli distribution
pub fn bernoulli<T>(p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().bernoulli(p, shape))
}

/// Generate samples from an F-distribution
pub fn f<T>(dfnum: f64, dfden: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().f(dfnum, dfden, shape))
}

/// Generate samples from a geometric distribution
pub fn geometric<T>(p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().geometric(p, shape))
}

/// Generate samples from a negative binomial distribution
pub fn negative_binomial<T>(n: u64, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().negative_binomial(n, p, shape))
}

/// Generate samples from a hypergeometric distribution
pub fn hypergeometric<T>(ngood: u64, nbad: u64, nsample: u64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().hypergeometric(ngood, nbad, nsample, shape))
}

/// Generate samples from a logarithmic series distribution
pub fn logseries<T>(p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().logseries(p, shape))
}

/// Generate samples from a Rayleigh distribution
pub fn rayleigh<T>(scale: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().rayleigh(scale, shape))
}

/// Generate samples from a Wald (inverse Gaussian) distribution
pub fn wald<T>(mean: f64, scale: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().wald(mean, scale, shape))
}

/// Generate samples from a Weibull distribution
pub fn weibull<T>(a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().weibull(a, shape))
}

/// Generate samples from a triangular distribution
pub fn triangular<T>(left: f64, mode: f64, right: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().triangular(left, mode, right, shape))
}

/// Generate samples from a Pareto distribution
pub fn pareto<T>(a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().pareto(a, shape))
}

/// Generate samples from a Zipf distribution
pub fn zipf<T>(a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().zipf(a, shape))
}

/// Generate samples from a standard Cauchy distribution
pub fn standard_cauchy<T>(shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().standard_cauchy(shape))
}

/// Generate samples from a standard exponential distribution
pub fn standard_exponential<T>(shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().standard_exponential(shape))
}

/// Generate samples from a standard gamma distribution
pub fn standard_gamma<T>(shape_param: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static + From<f64>,
{
    DEFAULT_GENERATOR.with(|rng: &RefCell<Generator>| rng.borrow_mut().standard_gamma(shape_param, shape))
}

// --- Legacy API Functions (for backward compatibility) ---

/// Seed the legacy default RNG
///
/// This maintains backward compatibility with the existing RandomState API.
/// New code should create a Generator with a specific seed instead.
#[deprecated(note = "Use default_rng_with_seed(seed) instead for new code")]
pub fn seed(seed: u64) {
    DEFAULT_RNG.with(|rng| {
        let mut rng_mut = rng.borrow_mut();
        rng_mut.seed_u64(seed);
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
    DEFAULT_RNG.with(|rng| {
        let mut rng_mut = rng.borrow_mut();
        rng_mut.random(shape, dtype)
    })
}

/// Legacy randint function using RandomState
///
/// This maintains backward compatibility. New code should use the module-level
/// randint() function which delegates to the modern Generator API.
#[deprecated(note = "Use randint() function which uses modern Generator API")]
pub fn legacy_randint<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| {
        let mut rng_mut = rng.borrow_mut();
        rng_mut.randint(low, high, shape)
    })
}

// --- Module exports for modern API structure ---

/// Modern random number generation API
///
/// This sub-module provides the modern Generator/BitGenerator API
/// that matches NumPy's current random module structure.
pub mod modern {
    pub use super::{default_rng, default_rng_with_seed};
}

/// Legacy random number generation API
///
/// This sub-module provides the legacy RandomState API for backward compatibility.
#[deprecated(since = "0.1.0", note = "Use modern Generator API instead")]
#[allow(deprecated)]
pub mod legacy {
    pub use super::{legacy_random, legacy_randint, seed};
}

#[cfg(test)]
mod tests;