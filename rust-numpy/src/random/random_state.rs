// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::NumPyError;
use crate::random::bit_generator::{BitGenerator, PCG64};
use crate::random::generator::Generator;
use num_traits::NumCast;
use rand::RngCore;
use rand_distr::uniform::SampleUniform;

/// RandomState exposes the legacy NumPy random number generator interface.
/// Internally it now wraps a modern Generator.
pub struct RandomState {
    gen: Generator,
}

impl RandomState {
    pub fn new() -> Self {
        Self {
            gen: Generator::new(Box::new(PCG64::new())),
        }
    }

    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            gen: Generator::new(Box::new(PCG64::seed_from_u64(seed))),
        }
    }

    pub fn generator(&self) -> &Generator {
        &self.gen
    }

    pub fn generator_mut(&mut self) -> &mut Generator {
        &mut self.gen
    }

    // --- Delegation Methods ---

    pub fn random<T>(&mut self, shape: &[usize], dtype: Dtype) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + NumCast + 'static,
    {
        self.gen.random(shape, dtype)
    }

    pub fn randint<T>(&mut self, low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static,
    {
        self.gen.randint(low, high, shape)
    }

    pub fn uniform<T>(&mut self, low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static,
    {
        self.gen.uniform(low, high, shape)
    }

    pub fn normal<T>(&mut self, mean: T, std: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static + Into<f64> + From<f64>,
    {
        self.gen.normal(mean, std, shape)
    }

    pub fn standard_normal<T>(&mut self, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static + Into<f64> + From<f64>,
    {
        self.gen.standard_normal(shape)
    }

    pub fn binomial<T>(&mut self, n: u64, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.binomial(n, p, shape)
    }

    pub fn poisson<T>(&mut self, lam: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.poisson(lam, shape)
    }

    pub fn exponential<T>(&mut self, scale: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.exponential(scale, shape)
    }

    pub fn gamma<T>(
        &mut self,
        shape_param: f64,
        scale: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.gamma(shape_param, scale, shape)
    }

    pub fn beta<T>(&mut self, a: f64, b: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.beta(a, b, shape)
    }

    pub fn chisquare<T>(&mut self, df: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.chisquare(df, shape)
    }

    pub fn gumbel<T>(
        &mut self,
        loc: f64,
        scale: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.gumbel(loc, scale, shape)
    }

    pub fn logistic<T>(
        &mut self,
        loc: f64,
        scale: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.logistic(loc, scale, shape)
    }

    pub fn lognormal<T>(
        &mut self,
        mean: f64,
        sigma: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.lognormal(mean, sigma, shape)
    }

    pub fn multinomial<T>(
        &mut self,
        n: u64,
        pvals: &[f64],
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.multinomial(n, pvals, size)
    }

    pub fn dirichlet<T>(
        &mut self,
        alpha: &[f64],
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.dirichlet(alpha, size)
    }

    pub fn geometric<T>(&mut self, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.geometric(p, shape)
    }

    pub fn negative_binomial<T>(
        &mut self,
        n: u64,
        p: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.negative_binomial(n, p, shape)
    }

    pub fn hypergeometric<T>(
        &mut self,
        ngood: u64,
        nbad: u64,
        nsample: u64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.hypergeometric(ngood, nbad, nsample, shape)
    }

    pub fn logseries<T>(&mut self, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.logseries(p, shape)
    }

    pub fn rayleigh<T>(&mut self, scale: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.rayleigh(scale, shape)
    }

    pub fn wald<T>(
        &mut self,
        mean: f64,
        scale: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.wald(mean, scale, shape)
    }

    pub fn weibull<T>(&mut self, a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.weibull(a, shape)
    }

    pub fn triangular<T>(
        &mut self,
        left: f64,
        mode: f64,
        right: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.triangular(left, mode, right, shape)
    }

    pub fn pareto<T>(&mut self, a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.pareto(a, shape)
    }

    pub fn zipf<T>(&mut self, a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.zipf(a, shape)
    }

    pub fn standard_cauchy<T>(&mut self, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.standard_cauchy(shape)
    }

    pub fn standard_exponential<T>(&mut self, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.standard_exponential(shape)
    }

    pub fn standard_gamma<T>(
        &mut self,
        shape_param: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.standard_gamma(shape_param, shape)
    }

    pub fn bernoulli<T>(&mut self, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.bernoulli(p, shape)
    }

    pub fn f<T>(&mut self, dfnum: f64, dfden: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.f(dfnum, dfden, shape)
    }

    pub fn power<T>(&mut self, a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.power(a, shape)
    }

    pub fn vonmises<T>(
        &mut self,
        mu: f64,
        kappa: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.gen.vonmises(mu, kappa, shape)
    }
}

impl RngCore for RandomState {
    fn next_u32(&mut self) -> u32 {
        self.gen.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.gen.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.gen.fill_bytes(dest)
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.gen.try_fill_bytes(dest)
    }
}
