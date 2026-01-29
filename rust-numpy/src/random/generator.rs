// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::NumPyError;
use crate::random::bit_generator::BitGenerator;
use num_traits::NumCast;
use rand::distributions::Distribution;
use rand::prelude::*;
use rand::{Rng, RngCore};
use rand_distr::uniform::SampleUniform;
use rand_distr::{
    Beta, Binomial, ChiSquared, Exp, FisherF, Gamma, Gumbel, LogNormal, Normal, Poisson,
};

/// Generator for random numbers using a BitGenerator
pub struct Generator {
    bit_gen: Box<dyn BitGenerator>,
}

impl Generator {
    pub fn new(bit_gen: Box<dyn BitGenerator>) -> Self {
        Self { bit_gen }
    }

    pub fn bit_generator(&self) -> &dyn BitGenerator {
        self.bit_gen.as_ref()
    }

    pub fn bit_generator_mut(&mut self) -> &mut dyn BitGenerator {
        self.bit_gen.as_mut()
    }

    // --- Core Methods ---

    pub fn random<T>(&mut self, shape: &[usize], dtype: Dtype) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + NumCast + 'static,
    {
        let size = shape.iter().product();
        match dtype {
            Dtype::Float32 { .. } => {
                let mut data = Vec::with_capacity(size);
                for _ in 0..size {
                    let val: f32 = self.bit_gen.gen();
                    data.push(NumCast::from(val).unwrap_or_default());
                }
                Ok(Array::from_data(data, shape.to_vec()))
            }
            Dtype::Float64 { .. } => {
                let mut data = Vec::with_capacity(size);
                for _ in 0..size {
                    let val: f64 = self.bit_gen.gen();
                    data.push(NumCast::from(val).unwrap_or_default());
                }
                Ok(Array::from_data(data, shape.to_vec()))
            }
            _ => Err(NumPyError::not_implemented(
                "Dtype not supported for random()",
            )),
        }
    }

    /// Generate integers in range [low, high)
    pub fn randint<T>(&mut self, low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(self.bit_gen.gen_range(low.clone()..high.clone()));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn uniform<T>(&mut self, low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static,
    {
        self.randint(low, high, shape)
    }

    pub fn normal<T>(&mut self, mean: T, std: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static + Into<f64> + From<f64>,
    {
        let mean_f64 = mean.into();
        let std_f64 = std.into();
        let dist =
            Normal::new(mean_f64, std_f64).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn standard_normal<T>(&mut self, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + PartialOrd + SampleUniform + Default + 'static + Into<f64> + From<f64>,
    {
        self.normal(T::from(0.0), T::from(1.0), shape)
    }

    pub fn binomial<T>(&mut self, n: u64, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if !(0.0..=1.0).contains(&p) {
            return Err(NumPyError::invalid_value("p must be in [0, 1]"));
        }
        let dist = Binomial::new(n, p).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen) as f64));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn poisson<T>(&mut self, lam: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if lam < 0.0 {
            return Err(NumPyError::invalid_value("lambda must be non-negative"));
        }
        let dist = Poisson::new(lam).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen) as f64));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn exponential<T>(&mut self, scale: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if scale <= 0.0 {
            return Err(NumPyError::invalid_value("scale must be positive"));
        }
        let dist = Exp::new(1.0 / scale).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
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
        if shape_param <= 0.0 || scale <= 0.0 {
            return Err(NumPyError::invalid_value("parameters must be positive"));
        }
        let dist =
            Gamma::new(shape_param, scale).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn beta<T>(&mut self, a: f64, b: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if a <= 0.0 || b <= 0.0 {
            return Err(NumPyError::invalid_value("parameters must be positive"));
        }
        let dist = Beta::new(a, b).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn chisquare<T>(&mut self, df: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if df <= 0.0 {
            return Err(NumPyError::invalid_value("df must be positive"));
        }
        let dist = ChiSquared::new(df).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
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
        if scale <= 0.0 {
            return Err(NumPyError::invalid_value("scale must be positive"));
        }
        let dist = Gumbel::new(loc, scale).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn logistic<T>(
        &mut self,
        _loc: f64,
        _scale: f64,
        _shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        Err(NumPyError::internal_error(
            "logistic distribution not supported in this build",
        ))
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
        if sigma <= 0.0 {
            return Err(NumPyError::invalid_value("sigma must be positive"));
        }
        let dist =
            LogNormal::new(mean, sigma).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
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
        if pvals.iter().any(|&p| p < 0.0) {
            return Err(NumPyError::invalid_value(
                "probabilities must be non-negative",
            ));
        }
        let sum: f64 = pvals.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return Err(NumPyError::invalid_value("probabilities must sum to 1"));
        }

        let output_shape = if let Some(size) = size {
            let mut s = size.to_vec();
            s.push(pvals.len());
            s
        } else {
            vec![pvals.len()]
        };

        let total_size = output_shape.iter().product::<usize>() / pvals.len();
        let mut data = Vec::with_capacity(output_shape.iter().product());

        for _ in 0..total_size {
            let mut remaining = n;
            let mut remaining_prob = 1.0;
            let mut results = vec![0u64; pvals.len()];

            for (i, &p) in pvals.iter().enumerate() {
                if i == pvals.len() - 1 {
                    results[i] = remaining;
                } else {
                    if remaining == 0 || p == 0.0 {
                        results[i] = 0;
                        continue;
                    }
                    let adjusted_p = (p / remaining_prob).min(1.0);
                    let dist = Binomial::new(remaining, adjusted_p)
                        .map_err(|e| NumPyError::invalid_value(e.to_string()))?;
                    results[i] = dist.sample(&mut self.bit_gen);
                    remaining -= results[i];
                    remaining_prob -= p;
                }
            }
            for &res in &results {
                data.push(T::from(res as f64));
            }
        }
        Ok(Array::from_data(data, output_shape))
    }

    pub fn dirichlet<T>(
        &mut self,
        alpha: &[f64],
        size: Option<&[usize]>,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if alpha.iter().any(|&a| a <= 0.0) {
            return Err(NumPyError::invalid_value(
                "alpha parameters must be positive",
            ));
        }
        let k = alpha.len();
        let output_shape = if let Some(size) = size {
            let mut s = size.to_vec();
            s.push(k);
            s
        } else {
            vec![k]
        };
        let total_size = output_shape.iter().product::<usize>() / k;
        let mut data = Vec::with_capacity(output_shape.iter().product());
        for _ in 0..total_size {
            let mut samples = Vec::with_capacity(k);
            let mut sum = 0.0;
            for &a in alpha {
                let dist =
                    Gamma::new(a, 1.0).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
                let sample = dist.sample(&mut self.bit_gen);
                samples.push(sample);
                sum += sample;
            }
            for sample in samples {
                data.push(T::from(sample / sum));
            }
        }
        Ok(Array::from_data(data, output_shape))
    }

    pub fn geometric<T>(&mut self, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if !(0.0..=1.0).contains(&p) {
            return Err(NumPyError::invalid_value("p must be in [0, 1]"));
        }
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = self.bit_gen.gen();
            let sample = (1.0 - u).ln() / (1.0 - p).ln();
            data.push(T::from(sample.ceil()));
        }
        Ok(Array::from_data(data, shape.to_vec()))
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
        if !(0.0..=1.0).contains(&p) {
            return Err(NumPyError::invalid_value("p must be in [0, 1]"));
        }
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let geom_p = 1.0 - p;
            let u: f64 = self.bit_gen.gen();
            let sample = (1.0 - u).ln() / geom_p.ln();
            let result = sample.ceil() as u64 + n - 1;
            data.push(T::from(result as f64));
        }
        Ok(Array::from_data(data, shape.to_vec()))
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
        if nsample > ngood + nbad {
            return Err(NumPyError::invalid_value(
                "nsample must not exceed population size",
            ));
        }
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        let mut population: Vec<bool> = (0..ngood).map(|_| true).collect();
        population.extend((0..nbad).map(|_| false));
        for _ in 0..size {
            let mut sample_pop = population.clone();
            sample_pop.shuffle(&mut self.bit_gen);
            let good_count = sample_pop
                .iter()
                .take(nsample as usize)
                .filter(|&&x| x)
                .count();
            data.push(T::from(good_count as f64));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn logseries<T>(&mut self, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if !(0.0..1.0).contains(&p) {
            return Err(NumPyError::invalid_value("p must be in (0, 1)"));
        }
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = self.bit_gen.gen();
            let q = 1.0 - p;
            let sample = (1.0 - q.powf(1.0 - u)) / (1.0 - q);
            let count = sample.log(1.0 / q).floor() as u64 + 1;
            data.push(T::from(count as f64));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn rayleigh<T>(&mut self, scale: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if scale <= 0.0 {
            return Err(NumPyError::invalid_value("scale must be positive"));
        }
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = self.bit_gen.gen();
            let sample = scale * (-2.0 * u.ln()).sqrt();
            data.push(T::from(sample));
        }
        Ok(Array::from_data(data, shape.to_vec()))
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
        if mean <= 0.0 || scale <= 0.0 {
            return Err(NumPyError::invalid_value("parameters must be positive"));
        }
        let dist = Normal::new(0.0, 1.0).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let y: f64 = dist.sample(&mut self.bit_gen);
            let mu_y = mean * y;
            let x = mean + (mu_y * mu_y) / (2.0 * scale)
                - (mu_y / (2.0 * scale)) * ((4.0 * scale) + mu_y).sqrt();
            let u: f64 = self.bit_gen.gen();
            let sample = if u <= mean / (mean + x) {
                x.max(0.0)
            } else {
                (mean * mean) / x.max(f64::EPSILON)
            };
            data.push(T::from(sample));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn weibull<T>(&mut self, a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if a <= 0.0 {
            return Err(NumPyError::invalid_value(
                "shape parameter must be positive",
            ));
        }
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = self.bit_gen.gen();
            let sample = (-u.ln()).powf(1.0 / a);
            data.push(T::from(sample));
        }
        Ok(Array::from_data(data, shape.to_vec()))
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
        if left >= right || mode < left || mode > right {
            return Err(NumPyError::invalid_value("invalid triangular parameters"));
        }
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = self.bit_gen.gen();
            let sample = if u <= (mode - left) / (right - left) {
                left + (u * (right - left) * (mode - left)).sqrt()
            } else {
                right - ((1.0 - u) * (right - left) * (right - mode)).sqrt()
            };
            data.push(T::from(sample));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn pareto<T>(&mut self, a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if a <= 0.0 {
            return Err(NumPyError::invalid_value("a must be positive"));
        }
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = self.bit_gen.gen();
            let sample = (1.0 - u).powf(-1.0 / a) - 1.0;
            data.push(T::from(sample));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn zipf<T>(&mut self, a: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if a <= 1.0 {
            return Err(NumPyError::invalid_value("exponent must be > 1"));
        }
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = self.bit_gen.gen();
            let sample = (1.0 - u).powf(-1.0 / (a - 1.0)).floor();
            data.push(T::from(sample));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn standard_cauchy<T>(&mut self, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let u: f64 = self.bit_gen.gen();
            let sample = (std::f64::consts::PI * (u - 0.5)).tan();
            data.push(T::from(sample));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn standard_exponential<T>(&mut self, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        let dist = Exp::new(1.0).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn standard_gamma<T>(
        &mut self,
        shape_param: f64,
        shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if shape_param <= 0.0 {
            return Err(NumPyError::invalid_value("shape must be positive"));
        }
        let dist =
            Gamma::new(shape_param, 1.0).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn bernoulli<T>(&mut self, p: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        self.binomial(1, p, shape)
    }

    pub fn f<T>(&mut self, dfnum: f64, dfden: f64, shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        if dfnum <= 0.0 || dfden <= 0.0 {
            return Err(NumPyError::invalid_value("parameters must be positive"));
        }
        let dist =
            FisherF::new(dfnum, dfden).map_err(|e| NumPyError::invalid_value(e.to_string()))?;
        let size = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(T::from(dist.sample(&mut self.bit_gen)));
        }
        Ok(Array::from_data(data, shape.to_vec()))
    }

    pub fn power<T>(&mut self, _a: f64, _shape: &[usize]) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        Err(NumPyError::internal_error(
            "power distribution not supported in this build",
        ))
    }

    pub fn vonmises<T>(
        &mut self,
        _mu: f64,
        _kappa: f64,
        _shape: &[usize],
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static + From<f64>,
    {
        Err(NumPyError::internal_error(
            "vonmises distribution not supported in this build",
        ))
    }

    // --- Utility Methods ---

    /// Shuffle an array in-place
    ///
    /// This method shuffles the contents of an array along the first axis.
    /// For multi-dimensional arrays, it shuffles each sub-array independently.
    pub fn shuffle<T>(&mut self, arr: &mut Array<T>) -> Result<(), NumPyError>
    where
        T: Clone + Default + 'static,
    {
        if arr.ndim() == 1 {
            // For 1D arrays, shuffle the elements directly
            let data = arr.data.as_slice_mut();
            data.shuffle(&mut self.bit_gen);
        } else {
            // For multi-dimensional arrays, shuffle along the first axis
            let shape = arr.shape().to_vec();
            if shape.is_empty() {
                return Ok(());
            }

            let first_axis_size = shape[0];
            let element_size: usize = shape[1..].iter().product();

            let data = arr.data.as_slice_mut();

            // Create indices for shuffling
            let mut indices: Vec<usize> = (0..first_axis_size).collect();
            indices.shuffle(&mut self.bit_gen);

            // Reorder the data according to shuffled indices
            let mut new_data = Vec::with_capacity(data.len());
            for &idx in &indices {
                let start = idx * element_size;
                let end = start + element_size;
                new_data.extend_from_slice(&data[start..end]);
            }

            // Replace the data
            data.clone_from_slice(&new_data);
        }

        Ok(())
    }

    /// Generate a random permutation of integers
    ///
    /// Returns an array containing a random permutation of integers from 0 to n-1.
    pub fn permutation(&mut self, n: usize) -> Result<Array<i64>, NumPyError> {
        let mut indices: Vec<i64> = (0..n as i64).collect();
        indices.shuffle(&mut self.bit_gen);
        Ok(Array::from_data(indices, vec![n]))
    }

    /// Generate a random sample from an array
    ///
    /// Returns a random sample of size k from the given array without replacement.
    pub fn choice<T>(&mut self, a: &[T], size: usize, replace: bool) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static,
    {
        if a.is_empty() {
            return Err(NumPyError::invalid_value("Cannot choose from empty array"));
        }

        if !replace && size > a.len() {
            return Err(NumPyError::invalid_value(
                "Cannot take a larger sample than population when replace=False",
            ));
        }

        let mut data = Vec::with_capacity(size);

        if replace {
            // Sampling with replacement
            for _ in 0..size {
                let idx = self.bit_gen.gen_range(0..a.len());
                data.push(a[idx].clone());
            }
        } else {
            // Sampling without replacement
            let mut indices: Vec<usize> = (0..a.len()).collect();
            indices.shuffle(&mut self.bit_gen);

            for i in 0..size {
                data.push(a[indices[i]].clone());
            }
        }

        Ok(Array::from_data(data, vec![size]))
    }

    /// Generate a random sample from a range
    ///
    /// Returns a random sample of integers from the range [low, high).
    pub fn integers(&mut self, low: i64, high: i64, size: usize) -> Result<Array<i64>, NumPyError> {
        if low >= high {
            return Err(NumPyError::invalid_value("low must be less than high"));
        }

        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(self.bit_gen.gen_range(low..high));
        }

        Ok(Array::from_data(data, vec![size]))
    }

    /// Generate random bytes
    ///
    /// Returns an array of random bytes.
    pub fn bytes(&mut self, length: usize) -> Result<Array<u8>, NumPyError> {
        let mut data = vec![0u8; length];
        self.bit_gen.fill_bytes(&mut data);
        Ok(Array::from_data(data, vec![length]))
    }

    /// Generate random floating point numbers in [0.0, 1.0)
    ///
    /// Returns an array of random floats in the half-open interval [0.0, 1.0).
    pub fn random_floats(&mut self, size: usize) -> Result<Array<f64>, NumPyError> {
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(self.bit_gen.gen::<f64>());
        }
        Ok(Array::from_data(data, vec![size]))
    }

    /// Generate random floating point numbers in [low, high)
    ///
    /// Returns an array of random floats in the specified range.
    pub fn random_floats_range(
        &mut self,
        low: f64,
        high: f64,
        size: usize,
    ) -> Result<Array<f64>, NumPyError> {
        if low >= high {
            return Err(NumPyError::invalid_value("low must be less than high"));
        }

        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            let val: f64 = self.bit_gen.gen();
            data.push(low + val * (high - low));
        }
        Ok(Array::from_data(data, vec![size]))
    }

    /// Generate random boolean values
    ///
    /// Returns an array of random boolean values.
    pub fn random_bools(&mut self, size: usize) -> Result<Array<bool>, NumPyError> {
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(self.bit_gen.gen::<bool>());
        }
        Ok(Array::from_data(data, vec![size]))
    }
}

impl RngCore for Generator {
    fn next_u32(&mut self) -> u32 {
        self.bit_gen.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.bit_gen.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.bit_gen.fill_bytes(dest)
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.bit_gen.try_fill_bytes(dest)
    }
}
