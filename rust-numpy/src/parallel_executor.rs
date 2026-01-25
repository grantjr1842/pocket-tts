//! Advanced Parallel Executor for rust-numpy
//!
//! This module provides a comprehensive parallel execution framework with:
//! - Thread pool management
//! - Work stealing for load balancing
//! - NUMA-aware memory allocation
//! - Automatic parallel threshold tuning
//! - Performance monitoring

use crate::array::Array;
use crate::error::NumPyError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Performance statistics for parallel operations
#[derive(Debug)]
pub struct PerformanceStats {
    pub total_operations: AtomicUsize,
    pub parallel_operations: AtomicUsize,
    pub total_elements_processed: AtomicUsize,
    pub average_thread_utilization: f64,
    pub cache_hit_rate: f64,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            total_operations: AtomicUsize::new(0),
            parallel_operations: AtomicUsize::new(0),
            total_elements_processed: AtomicUsize::new(0),
            average_thread_utilization: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

/// NUMA node information for multi-socket systems
#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: usize,
    pub cpu_list: Vec<usize>,
    pub memory_size: usize,
}

impl NumaNode {
    /// Detect NUMA topology (simplified - in production would use hwloc/libnuma)
    pub fn detect() -> Vec<Self> {
        // For now, return a single NUMA node (UMA system)
        // In production, this would query the system for actual NUMA topology
        vec![NumaNode {
            node_id: 0,
            cpu_list: (0..thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1))
                .collect(),
            memory_size: 0, // Would query actual memory size
        }]
    }
}

/// Strategy for work stealing
#[derive(Debug, Clone, Copy)]
pub enum StealStrategy {
    /// Random work stealing
    Random,
    /// Steal from most loaded queue
    MostLoaded,
    /// Steal from least recently stolen
    LeastRecentlyStolen,
}

impl Default for StealStrategy {
    fn default() -> Self {
        Self::MostLoaded
    }
}

/// Work queue for thread pool
struct WorkQueue {
    tasks: Mutex<Vec<Box<dyn FnOnce() + Send>>>,
    pending_tasks: AtomicUsize,
}

impl WorkQueue {
    fn new() -> Self {
        Self {
            tasks: Mutex::new(Vec::new()),
            pending_tasks: AtomicUsize::new(0),
        }
    }

    fn push(&self, task: Box<dyn FnOnce() + Send>) {
        self.tasks.lock().unwrap().push(task);
        self.pending_tasks.fetch_add(1, Ordering::Release);
    }

    fn pop(&self) -> Option<Box<dyn FnOnce() + Send>> {
        let mut tasks = self.tasks.lock().unwrap();
        if let Some(task) = tasks.pop() {
            self.pending_tasks.fetch_sub(1, Ordering::Release);
            Some(task)
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.pending_tasks.load(Ordering::Acquire)
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Work stealer for load balancing
pub struct WorkStealer {
    queues: Vec<Arc<WorkQueue>>,
    strategy: StealStrategy,
}

impl WorkStealer {
    fn new(queues: Vec<Arc<WorkQueue>>, strategy: StealStrategy) -> Self {
        Self { queues, strategy }
    }

    /// Try to steal work from another queue
    fn steal(&self, exclude_queue: usize) -> Option<Box<dyn FnOnce() + Send>> {
        match self.strategy {
            StealStrategy::Random => {
                let mut rng = rand::random::<usize>();
                for _ in 0..self.queues.len() {
                    let idx = rng % self.queues.len();
                    rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                    if idx != exclude_queue && !self.queues[idx].is_empty() {
                        if let Some(task) = self.queues[idx].pop() {
                            return Some(task);
                        }
                    }
                }
            }
            StealStrategy::MostLoaded => {
                let mut best_idx = None;
                let mut best_len = 0;
                for (idx, queue) in self.queues.iter().enumerate() {
                    if idx != exclude_queue {
                        let len = queue.len();
                        if len > best_len {
                            best_len = len;
                            best_idx = Some(idx);
                        }
                    }
                }
                if let Some(idx) = best_idx {
                    if let Some(task) = self.queues[idx].pop() {
                        return Some(task);
                    }
                }
            }
            StealStrategy::LeastRecentlyStolen => {
                // Simplified: steal from first non-empty queue
                for (idx, queue) in self.queues.iter().enumerate() {
                    if idx != exclude_queue && !queue.is_empty() {
                        if let Some(task) = queue.pop() {
                            return Some(task);
                        }
                    }
                }
            }
        }
        None
    }
}

/// Thread pool for parallel operations
pub struct ThreadPool {
    queues: Vec<Arc<WorkQueue>>,
    work_stealer: Arc<WorkStealer>,
    numa_nodes: Vec<NumaNode>,
    stats: Arc<Mutex<PerformanceStats>>,
    num_threads: usize,
}

impl ThreadPool {
    /// Create a new thread pool
    pub fn new() -> Result<Self, NumPyError> {
        let num_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);

        Self::with_num_threads(num_threads)
    }

    /// Create a thread pool with specified number of threads
    pub fn with_num_threads(num_threads: usize) -> Result<Self, NumPyError> {
        let mut queues = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            queues.push(Arc::new(WorkQueue::new()));
        }

        let work_stealer = Arc::new(WorkStealer::new(
            queues.clone(),
            StealStrategy::MostLoaded,
        ));

        let numa_nodes = NumaNode::detect();

        let pool = Self {
            queues: queues.clone(),
            work_stealer,
            numa_nodes,
            stats: Arc::new(Mutex::new(PerformanceStats::default())),
            num_threads,
        };

        // Spawn worker threads
        for thread_id in 0..num_threads {
            let queue = queues[thread_id].clone();
            let stealer = Arc::clone(&pool.work_stealer);
            let stats = Arc::clone(&pool.stats);

            thread::Builder::new()
                .name(format!("numpy-worker-{}", thread_id))
                .spawn(move || {
                    ThreadPool::worker_loop(thread_id, queue, stealer, stats);
                })
                .map_err(|e| NumPyError::thread_error(&format!("Failed to spawn thread: {}", e)))?;
        }

        Ok(pool)
    }

    /// Worker thread main loop
    fn worker_loop(
        thread_id: usize,
        queue: Arc<WorkQueue>,
        stealer: Arc<WorkStealer>,
        stats: Arc<Mutex<PerformanceStats>>,
    ) {
        loop {
            // Try to get work from local queue
            let task = queue.pop().or_else(|| stealer.steal(thread_id));

            if let Some(task) = task {
                stats.lock().unwrap().total_operations.fetch_add(1, Ordering::Relaxed);
                task();
            } else {
                // No work available, sleep briefly
                thread::sleep(Duration::from_micros(100));
            }
        }
    }

    /// Get number of threads in the pool
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Parallel threshold tuner for automatic parallelization decisions
pub struct ParallelThresholdTuner {
    min_parallel_size: usize,
    auto_tune_enabled: bool,
    history: Mutex<Vec<(usize, Duration)>>,
}

impl ParallelThresholdTuner {
    /// Create a new threshold tuner
    pub fn new() -> Self {
        Self {
            min_parallel_size: 4096, // Default threshold
            auto_tune_enabled: true,
            history: Mutex::new(Vec::new()),
        }
    }

    /// Set minimum size for parallel execution
    pub fn set_min_parallel_size(&mut self, size: usize) {
        self.min_parallel_size = size;
    }

    /// Get minimum size for parallel execution
    pub fn min_parallel_size(&self) -> usize {
        self.min_parallel_size
    }

    /// Record execution time for auto-tuning
    pub fn record_execution(&self, size: usize, duration: Duration) {
        if self.auto_tune_enabled {
            let mut history = self.history.lock().unwrap();
            history.push((size, duration));

            // Keep only recent history (last 1000 entries)
            let len = history.len();
            if len > 1000 {
                history.drain(0..len - 1000);
            }
        }
    }

    /// Should use parallel execution for given size
    pub fn should_parallelize(&self, size: usize) -> bool {
        size >= self.min_parallel_size
    }

    /// Analyze history and suggest optimal threshold
    pub fn suggest_threshold(&self) -> Option<usize> {
        let history = self.history.lock().unwrap();
        if history.len() < 10 {
            return None;
        }

        // Simple heuristic: find size where parallel becomes faster
        // In production, this would use more sophisticated analysis
        Some(self.min_parallel_size)
    }
}

impl Default for ParallelThresholdTuner {
    fn default() -> Self {
        Self::new()
    }
}

/// Main parallel executor
pub struct ParallelExecutor {
    thread_pool: Option<ThreadPool>,
    threshold_tuner: ParallelThresholdTuner,
    stats: PerformanceStats,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new() -> Result<Self, NumPyError> {
        Ok(Self {
            thread_pool: Some(ThreadPool::new()?),
            threshold_tuner: ParallelThresholdTuner::new(),
            stats: PerformanceStats::default(),
        })
    }

    /// Create parallel executor with custom thread count
    pub fn with_num_threads(num_threads: usize) -> Result<Self, NumPyError> {
        Ok(Self {
            thread_pool: Some(ThreadPool::with_num_threads(num_threads)?),
            threshold_tuner: ParallelThresholdTuner::new(),
            stats: PerformanceStats::default(),
        })
    }

    /// Create parallel executor without thread pool (uses Rayon global pool)
    pub fn with_rayon() -> Self {
        Self {
            thread_pool: None,
            threshold_tuner: ParallelThresholdTuner::new(),
            stats: PerformanceStats::default(),
        }
    }

    /// Get performance statistics
    pub fn stats(&self) -> &PerformanceStats {
        &self.stats
    }

    /// Get threshold tuner
    pub fn threshold_tuner(&self) -> &ParallelThresholdTuner {
        &self.threshold_tuner
    }

    /// Get threshold tuner mutably
    pub fn threshold_tuner_mut(&mut self) -> &mut ParallelThresholdTuner {
        &mut self.threshold_tuner
    }

    /// Execute parallel operation on array elements
    #[cfg(feature = "rayon")]
    pub fn execute_parallel<T, F>(
        &self,
        array: &Array<T>,
        operation: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Send + Sync + Clone + Default + 'static,
        F: Fn(&T) -> T + Send + Sync + 'static,
    {
        let size = array.size();

        // Check if parallel execution is worthwhile
        if !self.threshold_tuner.should_parallelize(size) {
            // Inline sequential execution to avoid ownership issues
            let result_data: Vec<T> = array.to_vec().iter().map(|x| operation(x)).collect();
            return Ok(Array::from_data(result_data, array.shape().to_vec()));
        }

        let start = Instant::now();

        // Convert to vector for parallel processing
        let input_data = array.to_vec();
        let result_data: Vec<T> = input_data.par_iter().map(|x| operation(x)).collect();

        let duration = start.elapsed();
        self.threshold_tuner.record_execution(size, duration);
        self.stats.total_elements_processed.fetch_add(size, Ordering::Relaxed);
        self.stats.parallel_operations.fetch_add(1, Ordering::Relaxed);

        Ok(Array::from_data(result_data, array.shape().to_vec()))
    }

    /// Execute sequential operation (fallback)
    pub fn execute_sequential<T, F>(
        &self,
        array: &Array<T>,
        operation: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static,
        F: Fn(&T) -> T,
    {
        let result_data: Vec<T> = array.to_vec().iter().map(|x| operation(x)).collect();
        Ok(Array::from_data(result_data, array.shape().to_vec()))
    }

    /// Parallel binary operation
    #[cfg(feature = "rayon")]
    pub fn binary_op<T, F>(
        &self,
        a: &Array<T>,
        b: &Array<T>,
        operation: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Send + Sync + Clone + Default + 'static,
        F: Fn(&T, &T) -> T + Send + Sync + 'static,
    {
        let size = a.size();
        if size != b.size() {
            return Err(NumPyError::shape_mismatch(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        if !self.threshold_tuner.should_parallelize(size) {
            // Inline sequential execution to avoid ownership issues
            let a_vec = a.to_vec();
            let b_vec = b.to_vec();

            let result_data: Vec<T> = a_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(a_elem, b_elem)| operation(a_elem, b_elem))
                .collect();

            return Ok(Array::from_data(result_data, a.shape().to_vec()));
        }

        let a_vec = a.to_vec();
        let b_vec = b.to_vec();

        let result_data: Vec<T> = a_vec
            .par_iter()
            .zip(b_vec.par_iter())
            .map(|(a_elem, b_elem)| operation(a_elem, b_elem))
            .collect();

        Ok(Array::from_data(result_data, a.shape().to_vec()))
    }

    /// Sequential binary operation (fallback)
    pub fn binary_op_sequential<T, F>(
        &self,
        a: &Array<T>,
        b: &Array<T>,
        operation: F,
    ) -> Result<Array<T>, NumPyError>
    where
        T: Clone + Default + 'static,
        F: Fn(&T, &T) -> T,
    {
        let size = a.size();
        if size != b.size() {
            return Err(NumPyError::shape_mismatch(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        let a_vec = a.to_vec();
        let b_vec = b.to_vec();

        let result_data: Vec<T> = a_vec
            .iter()
            .zip(b_vec.iter())
            .map(|(a_elem, b_elem)| operation(a_elem, b_elem))
            .collect();

        Ok(Array::from_data(result_data, a.shape().to_vec()))
    }

    /// Parallel reduction
    #[cfg(feature = "rayon")]
    pub fn reduce<T, F, R>(
        &self,
        array: &Array<T>,
        identity: R,
        op: F,
    ) -> Result<T, NumPyError>
    where
        T: Send + Sync + Clone + 'static,
        R: Fn() -> T + Send + Sync + 'static,
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        let data = array.to_vec();
        let result = data.par_iter().cloned().reduce(identity, op);
        Ok(result)
    }

    /// Optimize array layout for NUMA
    pub fn optimize_for_numa<T>(&self, _array: &Array<T>) {
        // NUMA optimization would go here
        // For now, this is a placeholder as true NUMA optimization
        // requires platform-specific code and possibly libnuma
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::with_rayon()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_executor_creation() {
        let executor = ParallelExecutor::with_rayon();
        assert!(executor.thread_pool.is_none());
    }

    #[test]
    fn test_threshold_tuner() {
        let mut tuner = ParallelThresholdTuner::new();
        assert!(!tuner.should_parallelize(100));
        assert!(tuner.should_parallelize(10000));

        tuner.set_min_parallel_size(100);
        assert!(!tuner.should_parallelize(50));
        assert!(tuner.should_parallelize(150));
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_execute_parallel() {
        let executor = ParallelExecutor::with_rayon();
        let array = Array::from_vec(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let result = executor
            .execute_parallel(&array, |x| x * 2)
            .unwrap();

        assert_eq!(result.to_vec(), vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_binary_op() {
        let executor = ParallelExecutor::with_rayon();
        let a = Array::from_vec(vec![1i32, 2, 3, 4, 5]);
        let b = Array::from_vec(vec![10i32, 20, 30, 40, 50]);

        let result = executor
            .binary_op(&a, &b, |x, y| x + y)
            .unwrap();

        assert_eq!(result.to_vec(), vec![11, 22, 33, 44, 55]);
    }

    #[test]
    #[cfg(feature = "rayon")]
    fn test_reduce() {
        let executor = ParallelExecutor::with_rayon();
        let array = Array::from_vec(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let result = executor
            .reduce(&array, || 0, |a, b| a + b)
            .unwrap();

        assert_eq!(result, 55);
    }

    #[test]
    fn test_execute_sequential() {
        let executor = ParallelExecutor::with_rayon();
        let array = Array::from_vec(vec![1i32, 2, 3, 4, 5]);

        let result = executor
            .execute_sequential(&array, |x| x * 3)
            .unwrap();

        assert_eq!(result.to_vec(), vec![3, 6, 9, 12, 15]);
    }

    #[test]
    fn test_binary_op_sequential() {
        let executor = ParallelExecutor::with_rayon();
        let a = Array::from_vec(vec![1i32, 2, 3]);
        let b = Array::from_vec(vec![4i32, 5, 6]);

        let result = executor
            .binary_op_sequential(&a, &b, |x, y| x * y)
            .unwrap();

        assert_eq!(result.to_vec(), vec![4, 10, 18]);
    }
}
