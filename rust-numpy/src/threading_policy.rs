use crate::error::NumPyError;
use std::sync::atomic::{AtomicUsize, Ordering};

static GLOBAL_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct ThreadingPolicy {
    pub max_threads: Option<usize>,
    pub min_array_size: usize,
    pub enable_nested_parallel: bool,
}

impl Default for ThreadingPolicy {
    fn default() -> Self {
        let num_cores = 8;
        Self {
            max_threads: Some(num_cores),
            min_array_size: std::cmp::max(1000, 100000 / num_cores),
            enable_nested_parallel: false,
        }
    }
}

impl ThreadingPolicy {
    pub fn from_env() -> Self {
        let mut policy = Self::default();

        if let Ok(max_threads) = std::env::var("RUST_NUMPY_MAX_THREADS") {
            if let Ok(threads) = max_threads.parse::<usize>() {
                policy.max_threads = Some(threads);
            }
        }

        if let Ok(min_size) = std::env::var("RUST_NUMPY_MIN_ARRAY_SIZE") {
            if let Ok(size) = min_size.parse::<usize>() {
                policy.min_array_size = size;
            }
        }

        policy.enable_nested_parallel = std::env::var("RUST_NUMPY_NESTED_PARALLEL")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);

        policy
    }

    pub fn should_use_parallel(&self, array_size: usize) -> bool {
        array_size >= self.min_array_size
    }

    pub fn get_thread_count(&self, array_size: usize) -> usize {
        if !self.should_use_parallel(array_size) {
            return 1;
        }

        let max_threads = self.max_threads.unwrap_or(1);
        let optimal_threads = std::cmp::min(
            max_threads,
            std::cmp::max(1, array_size / self.min_array_size),
        );

        optimal_threads
    }
}

pub fn set_global_thread_policy(policy: &ThreadingPolicy) -> Result<(), NumPyError> {
    GLOBAL_THREAD_COUNT.store(
        policy.max_threads.unwrap_or(num_cpus::get()),
        Ordering::Relaxed,
    );
    Ok(())
}

pub fn get_global_thread_count() -> usize {
    GLOBAL_THREAD_COUNT.load(Ordering::Relaxed)
}

pub struct ThreadGuard {
    old_nested_value: bool,
}

impl ThreadGuard {
    pub fn new(enable_nested: bool) -> Self {
        #[cfg(feature = "rayon")]
        {
            let old_nested_value = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .map(|_| rayon::current_num_threads() > 1)
                .unwrap_or(false);

            if enable_nested {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(get_global_thread_count())
                    .build()
                    .unwrap()
                    .install(|| {});
            }

            Self { old_nested_value }
        }

        #[cfg(not(feature = "rayon"))]
        {
            Self {
                old_nested_value: false,
            }
        }
    }
}

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        #[cfg(feature = "rayon")]
        {
            if !self.old_nested_value {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap()
                    .install(|| {});
            }
        }
    }
}
