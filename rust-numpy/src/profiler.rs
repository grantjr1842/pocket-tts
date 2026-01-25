//! Performance profiler for ufunc operations
//!
//! This module provides a comprehensive profiling system for tracking performance
//! metrics, detecting bottlenecks, and generating optimization recommendations.

use crate::performance_metrics::{
    Bottleneck, BottleneckType, MemoryTracker, OptimizationRecommendation,
    PerformanceMetrics, PerformanceReport,
};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Global profiler instance
static GLOBAL_PROFILER: Mutex<Option<Profiler>> = Mutex::new(None);

/// Performance profiler for ufunc operations
#[derive(Debug)]
pub struct Profiler {
    /// Metrics collected for each operation
    metrics: HashMap<String, PerformanceMetrics>,

    /// Memory tracking
    memory_tracker: MemoryTracker,

    /// Whether profiling is enabled
    enabled: bool,

    /// Baseline metrics for regression detection
    baseline_metrics: HashMap<String, PerformanceMetrics>,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            memory_tracker: MemoryTracker::new(),
            enabled: false,
            baseline_metrics: HashMap::new(),
        }
    }

    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Profile an operation and return its result
    pub fn profile_operation<T, F>(&mut self, name: &str, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        if !self.enabled {
            return operation();
        }

        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        // Get or create metrics for this operation
        let entry = self.metrics.entry(name.to_string()).or_default();

        // Update timing
        entry.operation_time = duration;

        // Calculate throughput if elements were processed
        if entry.elements_processed > 0 {
            entry.calculate_throughput();
        }

        result
    }

    /// Profile an operation with element count
    pub fn profile_operation_with_elements<T, F>(
        &mut self,
        name: &str,
        elements: usize,
        operation: F,
    ) -> T
    where
        F: FnOnce() -> T,
    {
        if !self.enabled {
            return operation();
        }

        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        let entry = self.metrics.entry(name.to_string()).or_default();
        entry.operation_time = duration;
        entry.elements_processed = elements;
        entry.calculate_throughput();

        result
    }

    /// Profile an operation with memory tracking
    pub fn profile_operation_with_memory<T, F>(
        &mut self,
        name: &str,
        allocated: usize,
        freed: usize,
        operation: F,
    ) -> T
    where
        F: FnOnce() -> T,
    {
        if !self.enabled {
            return operation();
        }

        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();

        let entry = self.metrics.entry(name.to_string()).or_default();
        entry.operation_time = duration;
        entry.memory_allocated = allocated;
        entry.memory_freed = freed;

        self.memory_tracker.record_allocation(allocated);
        self.memory_tracker.record_deallocation(freed);

        result
    }

    /// Set SIMD efficiency for an operation
    pub fn set_simd_efficiency(&mut self, name: &str, efficiency: f64) {
        if self.enabled {
            let entry = self.metrics.entry(name.to_string()).or_default();
            entry.simd_efficiency = efficiency.clamp(0.0, 1.0);
        }
    }

    /// Set parallel efficiency for an operation
    pub fn set_parallel_efficiency(&mut self, name: &str, efficiency: f64) {
        if self.enabled {
            let entry = self.metrics.entry(name.to_string()).or_default();
            entry.parallel_efficiency = efficiency.clamp(0.0, 1.0);
        }
    }

    /// Get metrics for a specific operation
    pub fn get_metrics(&self, name: &str) -> Option<&PerformanceMetrics> {
        self.metrics.get(name)
    }

    /// Get all metrics
    pub fn get_all_metrics(&self) -> &HashMap<String, PerformanceMetrics> {
        &self.metrics
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.metrics.clear();
        self.memory_tracker.reset();
    }

    /// Set baseline metrics for regression detection
    pub fn set_baseline(&mut self, name: &str, metrics: PerformanceMetrics) {
        self.baseline_metrics.insert(name.to_string(), metrics);
    }

    /// Detect performance regressions compared to baseline
    pub fn detect_regressions(&self) -> Vec<String> {
        let mut regressions = Vec::new();

        for (name, current) in &self.metrics {
            if let Some(baseline) = self.baseline_metrics.get(name) {
                // Check if operation time increased by more than 20%
                if current.operation_time.as_secs_f64() > 0.0
                    && baseline.operation_time.as_secs_f64() > 0.0
                {
                    let ratio = current.operation_time.as_secs_f64()
                        / baseline.operation_time.as_secs_f64();
                    if ratio > 1.2 {
                        regressions.push(format!(
                            "{}: {:.2}x slower than baseline ({:?} vs {:?})",
                            name, ratio, current.operation_time, baseline.operation_time
                        ));
                    }
                }

                // Check if throughput decreased by more than 20%
                if current.throughput > 0.0 && baseline.throughput > 0.0 {
                    let ratio = baseline.throughput / current.throughput;
                    if ratio > 1.2 {
                        regressions.push(format!(
                            "{}: {:.2}x lower throughput than baseline ({:.2} vs {:.2} elements/s)",
                            name, ratio, current.throughput, baseline.throughput
                        ));
                    }
                }
            }
        }

        regressions
    }

    /// Detect performance bottlenecks
    pub fn detect_bottlenecks(&self) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        for (name, metrics) in &self.metrics {
            // Check for memory allocation bottleneck
            if metrics.memory_allocated > 10_000_000 {
                // > 10MB
                bottlenecks.push(Bottleneck {
                    operation: name.clone(),
                    bottleneck_type: BottleneckType::MemoryAllocation,
                    severity: (metrics.memory_allocated as f64 / 100_000_000.0).min(1.0),
                    description: format!(
                        "High memory allocation: {} MB",
                        metrics.memory_allocated / 1_000_000
                    ),
                });
            }

            // Check for SIMD inefficiency
            if metrics.simd_efficiency < 0.7 && metrics.simd_efficiency > 0.0 {
                bottlenecks.push(Bottleneck {
                    operation: name.clone(),
                    bottleneck_type: BottleneckType::SimdInefficiency,
                    severity: 1.0 - metrics.simd_efficiency,
                    description: format!(
                        "Low SIMD efficiency: {:.1}%",
                        metrics.simd_efficiency * 100.0
                    ),
                });
            }

            // Check for parallel inefficiency
            if metrics.parallel_efficiency < 0.7 && metrics.parallel_efficiency > 0.0 {
                bottlenecks.push(Bottleneck {
                    operation: name.clone(),
                    bottleneck_type: BottleneckType::ParallelInefficiency,
                    severity: 1.0 - metrics.parallel_efficiency,
                    description: format!(
                        "Low parallel efficiency: {:.1}%",
                        metrics.parallel_efficiency * 100.0
                    ),
                });
            }

            // Check for slow computation
            if metrics.operation_time.as_millis() > 100 {
                bottlenecks.push(Bottleneck {
                    operation: name.clone(),
                    bottleneck_type: BottleneckType::SlowComputation,
                    severity: (metrics.operation_time.as_secs_f64() / 1.0).min(1.0),
                    description: format!(
                        "Slow operation: {:.2} ms",
                        metrics.operation_time.as_secs_f64() * 1000.0
                    ),
                });
            }
        }

        // Sort by severity descending
        bottlenecks.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap());

        bottlenecks
    }

    /// Analyze performance and generate a report
    pub fn analyze_performance(&self) -> PerformanceReport {
        let mut report = PerformanceReport::new();

        // Detect bottlenecks
        let bottlenecks = self.detect_bottlenecks();
        for bottleneck in bottlenecks {
            report.add_bottleneck(bottleneck.clone());

            // Generate recommendations based on bottleneck type
            let recommendation = match bottleneck.bottleneck_type {
                BottleneckType::MemoryAllocation => OptimizationRecommendation {
                    operation: bottleneck.operation.clone(),
                    priority: bottleneck.severity,
                    description: "Consider reusing buffers or implementing in-place operations".to_string(),
                    expected_improvement: 1.5,
                },
                BottleneckType::SimdInefficiency => OptimizationRecommendation {
                    operation: bottleneck.operation.clone(),
                    priority: bottleneck.severity * 0.8,
                    description: "Enable SIMD vectorization or optimize data alignment".to_string(),
                    expected_improvement: 4.0,
                },
                BottleneckType::ParallelInefficiency => OptimizationRecommendation {
                    operation: bottleneck.operation.clone(),
                    priority: bottleneck.severity * 0.7,
                    description: "Adjust chunk sizes or improve workload distribution".to_string(),
                    expected_improvement: 2.0,
                },
                BottleneckType::SlowComputation => OptimizationRecommendation {
                    operation: bottleneck.operation.clone(),
                    priority: bottleneck.severity * 0.6,
                    description: "Consider algorithmic optimizations or memoization".to_string(),
                    expected_improvement: 2.0,
                },
                BottleneckType::CacheMisses => OptimizationRecommendation {
                    operation: bottleneck.operation.clone(),
                    priority: bottleneck.severity * 0.9,
                    description: "Improve data locality or use cache-friendly algorithms".to_string(),
                    expected_improvement: 3.0,
                },
                BottleneckType::MemoryCopies => OptimizationRecommendation {
                    operation: bottleneck.operation.clone(),
                    priority: bottleneck.severity * 0.7,
                    description: "Eliminate unnecessary memory copies or use views".to_string(),
                    expected_improvement: 2.0,
                },
            };

            report.add_recommendation(recommendation);
        }

        // Calculate efficiency scores
        for (name, metrics) in &self.metrics {
            let mut score = 1.0;

            // Penalize low SIMD efficiency
            if metrics.simd_efficiency > 0.0 {
                score *= metrics.simd_efficiency;
            }

            // Penalize low parallel efficiency
            if metrics.parallel_efficiency > 0.0 {
                score *= metrics.parallel_efficiency;
            }

            // Penalize high memory usage
            if metrics.memory_allocated > 0 {
                let memory_penalty = 1.0 - (metrics.memory_allocated as f64 / 100_000_000.0).min(0.5);
                score *= memory_penalty;
            }

            report.set_efficiency_score(name.clone(), score.clamp(0.0, 1.0));
        }

        report
    }

    /// Print a summary of collected metrics
    pub fn print_summary(&self) {
        if !self.enabled {
            println!("Profiling is disabled");
            return;
        }

        if self.metrics.is_empty() {
            println!("No metrics collected");
            return;
        }

        println!("\n=== Performance Profiler Summary ===");
        println!("Operations tracked: {}", self.metrics.len());
        println!("Memory allocated: {} bytes", self.memory_tracker.allocated);
        println!("Memory freed: {} bytes", self.memory_tracker.freed);
        println!("Current usage: {} bytes\n", self.memory_tracker.current_usage);

        for (name, metrics) in &self.metrics {
            println!("{}: {}", name, metrics.summary());
        }
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Get or create the global profiler instance
pub fn global_profiler() -> &'static Mutex<Option<Profiler>> {
    &GLOBAL_PROFILER
}

/// Initialize the global profiler
pub fn init_profiler() {
    let mut guard = GLOBAL_PROFILER.lock().unwrap();
    if guard.is_none() {
        *guard = Some(Profiler::new());
    }
}

/// Enable the global profiler
pub fn enable_profiling() {
    init_profiler();
    let mut guard = GLOBAL_PROFILER.lock().unwrap();
    if let Some(ref mut profiler) = *guard {
        profiler.enable();
    }
}

/// Disable the global profiler
pub fn disable_profiling() {
    let mut guard = GLOBAL_PROFILER.lock().unwrap();
    if let Some(ref mut profiler) = *guard {
        profiler.disable();
    }
}

/// Check if global profiling is enabled
pub fn is_profiling_enabled() -> bool {
    let guard = GLOBAL_PROFILER.lock().unwrap();
    guard
        .as_ref()
        .map(|p| p.is_enabled())
        .unwrap_or(false)
}

/// Profile an operation using the global profiler
pub fn profile<T, F>(name: &str, operation: F) -> T
where
    F: FnOnce() -> T,
{
    let mut guard = GLOBAL_PROFILER.lock().unwrap();
    if let Some(ref mut profiler) = *guard {
        profiler.profile_operation(name, operation)
    } else {
        operation()
    }
}

/// Profile an operation with element count using the global profiler
pub fn profile_with_elements<T, F>(name: &str, elements: usize, operation: F) -> T
where
    F: FnOnce() -> T,
{
    let mut guard = GLOBAL_PROFILER.lock().unwrap();
    if let Some(ref mut profiler) = *guard {
        profiler.profile_operation_with_elements(name, elements, operation)
    } else {
        operation()
    }
}

/// Get the performance report from the global profiler
pub fn get_performance_report() -> Option<PerformanceReport> {
    let guard = GLOBAL_PROFILER.lock().unwrap();
    guard.as_ref().map(|p| p.analyze_performance())
}

/// Print the performance summary from the global profiler
pub fn print_profiler_summary() {
    let guard = GLOBAL_PROFILER.lock().unwrap();
    if let Some(ref profiler) = *guard {
        profiler.print_summary();
    } else {
        println!("Profiler not initialized");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert!(!profiler.is_enabled());
        assert_eq!(profiler.get_all_metrics().len(), 0);
    }

    #[test]
    fn test_profiler_enable_disable() {
        let mut profiler = Profiler::new();
        assert!(!profiler.is_enabled());

        profiler.enable();
        assert!(profiler.is_enabled());

        profiler.disable();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_profile_operation() {
        let mut profiler = Profiler::new();
        profiler.enable();

        let result = profiler.profile_operation("test_op", || {
            thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert_eq!(profiler.get_all_metrics().len(), 1);

        let metrics = profiler.get_metrics("test_op").unwrap();
        assert!(metrics.operation_time.as_millis() >= 10);
    }

    #[test]
    fn test_profile_operation_with_elements() {
        let mut profiler = Profiler::new();
        profiler.enable();

        profiler.profile_operation_with_elements("add", 1000, || {
            // Simulated operation
        });

        let metrics = profiler.get_metrics("add").unwrap();
        assert_eq!(metrics.elements_processed, 1000);
        assert!(metrics.throughput > 0.0);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut profiler = Profiler::new();
        profiler.enable();

        // Add a slow operation
        profiler.profile_operation("slow_op", || {
            thread::sleep(std::time::Duration::from_millis(150));
        });

        // Set low SIMD efficiency
        profiler.set_simd_efficiency("slow_op", 0.5);

        let bottlenecks = profiler.detect_bottlenecks();
        assert!(!bottlenecks.is_empty());
    }

    #[test]
    fn test_performance_report() {
        let mut profiler = Profiler::new();
        profiler.enable();

        profiler.profile_operation("test", || 42);

        let report = profiler.analyze_performance();
        assert_eq!(report.efficiency_scores.len(), 1);
    }

    #[test]
    fn test_regression_detection() {
        let mut profiler = Profiler::new();
        profiler.enable();

        // Set baseline
        let baseline = PerformanceMetrics {
            operation_time: Duration::from_millis(100),
            ..Default::default()
        };
        profiler.set_baseline("test", baseline);

        // Run slower operation
        profiler.profile_operation("test", || {
            thread::sleep(Duration::from_millis(150));
        });

        let regressions = profiler.detect_regressions();
        assert!(!regressions.is_empty());
    }

    #[test]
    fn test_reset() {
        let mut profiler = Profiler::new();
        profiler.enable();

        profiler.profile_operation("test", || 42);
        assert_eq!(profiler.get_all_metrics().len(), 1);

        profiler.reset();
        assert_eq!(profiler.get_all_metrics().len(), 0);
    }

    #[test]
    fn test_global_profiler() {
        init_profiler();
        enable_profiling();

        let result = profile("global_test", || 123);
        assert_eq!(result, 123);

        assert!(is_profiling_enabled());

        disable_profiling();
        assert!(!is_profiling_enabled());

        // Clean up
        let mut guard = GLOBAL_PROFILER.lock().unwrap();
        *guard = None;
    }
}
