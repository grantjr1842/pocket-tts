// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Random number generation utils

pub mod bit_generator;
pub mod generator;
pub mod random_state;

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::NumPyError;
use num_traits::NumCast;
use rand_distr::uniform::SampleUniform;
pub use random_state::RandomState;
use std::cell::RefCell;

thread_local! {
    static DEFAULT_RNG: RefCell<RandomState> = RefCell::new(RandomState::new());
}

// --- Global Functions (Proxies) ---

pub fn random<T: Clone + Default + NumCast + 'static>(
    shape: &[usize],
    dtype: Dtype,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().random(shape, dtype))
}

pub fn randint<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().randint(low, high, shape))
}

pub fn uniform<T: Clone + PartialOrd + SampleUniform + Default + 'static>(
    low: T,
    high: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().uniform(low, high, shape))
}

pub fn normal<T: Clone + PartialOrd + SampleUniform + Default + 'static + Into<f64> + From<f64>>(
    mean: T,
    std: T,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().normal(mean, std, shape))
}

pub fn standard_normal<
    T: Clone + PartialOrd + SampleUniform + Default + 'static + Into<f64> + From<f64>,
>(
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().standard_normal(shape))
}

pub fn binomial<T: Clone + Default + 'static + From<f64>>(
    n: u64,
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().binomial(n, p, shape))
}

pub fn poisson<T: Clone + Default + 'static + From<f64>>(
    lam: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().poisson(lam, shape))
}

pub fn exponential<T: Clone + Default + 'static + From<f64>>(
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().exponential(scale, shape))
}

pub fn gamma<T: Clone + Default + 'static + From<f64>>(
    shape_param: f64,
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().gamma(shape_param, scale, shape))
}

pub fn beta<T: Clone + Default + 'static + From<f64>>(
    a: f64,
    b: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().beta(a, b, shape))
}

pub fn chisquare<T: Clone + Default + 'static + From<f64>>(
    df: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().chisquare(df, shape))
}

pub fn gumbel<T: Clone + Default + 'static + From<f64>>(
    loc: f64,
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().gumbel(loc, scale, shape))
}

pub fn logistic<T: Clone + Default + 'static + From<f64>>(
    loc: f64,
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().logistic(loc, scale, shape))
}

pub fn lognormal<T: Clone + Default + 'static + From<f64>>(
    mean: f64,
    sigma: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().lognormal(mean, sigma, shape))
}

pub fn multinomial<T: Clone + Default + 'static + From<f64>>(
    n: u64,
    pvals: &[f64],
    size: Option<&[usize]>,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().multinomial(n, pvals, size))
}

pub fn dirichlet<T: Clone + Default + 'static + From<f64>>(
    alpha: &[f64],
    size: Option<&[usize]>,
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().dirichlet(alpha, size))
}

pub fn geometric<T: Clone + Default + 'static + From<f64>>(
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().geometric(p, shape))
}

pub fn negative_binomial<T: Clone + Default + 'static + From<f64>>(
    n: u64,
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().negative_binomial(n, p, shape))
}

pub fn hypergeometric<T: Clone + Default + 'static + From<f64>>(
    ngood: u64,
    nbad: u64,
    nsample: u64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().hypergeometric(ngood, nbad, nsample, shape))
}

pub fn logseries<T: Clone + Default + 'static + From<f64>>(
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().logseries(p, shape))
}

pub fn rayleigh<T: Clone + Default + 'static + From<f64>>(
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().rayleigh(scale, shape))
}

pub fn wald<T: Clone + Default + 'static + From<f64>>(
    mean: f64,
    scale: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().wald(mean, scale, shape))
}

pub fn weibull<T: Clone + Default + 'static + From<f64>>(
    a: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().weibull(a, shape))
}

pub fn triangular<T: Clone + Default + 'static + From<f64>>(
    left: f64,
    mode: f64,
    right: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().triangular(left, mode, right, shape))
}

pub fn pareto<T: Clone + Default + 'static + From<f64>>(
    a: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().pareto(a, shape))
}

pub fn zipf<T: Clone + Default + 'static + From<f64>>(
    a: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().zipf(a, shape))
}

pub fn standard_cauchy<T: Clone + Default + 'static + From<f64>>(
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().standard_cauchy(shape))
}

pub fn standard_exponential<T: Clone + Default + 'static + From<f64>>(
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().standard_exponential(shape))
}

pub fn standard_gamma<T: Clone + Default + 'static + From<f64>>(
    shape_param: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().standard_gamma(shape_param, shape))
}

pub fn bernoulli<T: Clone + Default + 'static + From<f64>>(
    p: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().bernoulli(p, shape))
}

pub fn f<T: Clone + Default + 'static + From<f64>>(
    dfnum: f64,
    dfden: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().f(dfnum, dfden, shape))
}

pub fn power<T: Clone + Default + 'static + From<f64>>(
    a: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().power(a, shape))
}

pub fn vonmises<T: Clone + Default + 'static + From<f64>>(
    mu: f64,
    kappa: f64,
    shape: &[usize],
) -> Result<Array<T>, NumPyError> {
    DEFAULT_RNG.with(|rng| rng.borrow_mut().vonmises(mu, kappa, shape))
}

pub fn seed(seed: u64) {
    DEFAULT_RNG.with(|rng| {
        *rng.borrow_mut() = RandomState::seed_from_u64(seed);
    });
}
