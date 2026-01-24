use crate::error::NumPyError;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "rayon")]
pub struct ParallelReductionConfig {
    pub min_size_for_parallel: usize,
    pub max_threads: Option<usize>,
}

#[cfg(feature = "rayon")]
impl Default for ParallelReductionConfig {
    fn default() -> Self {
        let num_threads = rayon::current_num_threads();
        Self {
            min_size_for_parallel: std::cmp::max(1024, 100000 / num_threads),
            max_threads: Some(num_threads),
        }
    }
}

#[cfg(feature = "rayon")]
pub fn parallel_sum<T>(data: &[T], config: &ParallelReductionConfig) -> T
where
    T: Send + Sync + Clone + std::ops::Add<Output = T> + std::iter::Sum,
{
    if data.len() < config.min_size_for_parallel {
        data.iter().cloned().sum()
    } else {
        let chunks = data
            .par_chunks(data.len() / config.max_threads.unwrap_or(rayon::current_num_threads()));
        let sums: Vec<T> = chunks.map(|chunk| chunk.iter().cloned().sum()).collect();
        sums.iter().cloned().sum()
    }
}

#[cfg(feature = "rayon")]
pub fn parallel_prod<T>(data: &[T], config: &ParallelReductionConfig) -> T
where
    T: Send + Sync + Clone + std::ops::Mul<Output = T> + std::iter::Product,
{
    if data.len() < config.min_size_for_parallel {
        data.iter().cloned().product()
    } else {
        let chunks = data
            .par_chunks(data.len() / config.max_threads.unwrap_or(rayon::current_num_threads()));
        let prods: Vec<T> = chunks
            .map(|chunk| chunk.iter().cloned().product())
            .collect();
        prods.iter().cloned().product()
    }
}

#[cfg(feature = "rayon")]
pub fn parallel_min<T>(data: &[T], config: &ParallelReductionConfig) -> T
where
    T: Send + Sync + Clone + PartialOrd,
{
    if data.len() < config.min_size_for_parallel {
        data.iter().min().unwrap().clone()
    } else {
        let chunks = data
            .par_chunks(data.len() / config.max_threads.unwrap_or(rayon::current_num_threads()));
        let mins: Vec<T> = chunks
            .map(|chunk| chunk.iter().min().unwrap().clone())
            .collect();
        mins.into_iter().min().unwrap()
    }
}

#[cfg(feature = "rayon")]
pub fn parallel_max<T>(data: &[T], config: &ParallelReductionConfig) -> T
where
    T: Send + Sync + Clone + PartialOrd,
{
    if data.len() < config.min_size_for_parallel {
        data.iter().max().unwrap().clone()
    } else {
        let chunks = data
            .par_chunks(data.len() / config.max_threads.unwrap_or(rayon::current_num_threads()));
        let maxs: Vec<T> = chunks
            .map(|chunk| chunk.iter().max().unwrap().clone())
            .collect();
        maxs.into_iter().max().unwrap()
    }
}
