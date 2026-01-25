//! Performance metrics collection for ufunc operations
//!
//! This module provides comprehensive performance monitoring capabilities including
//! operation timing, memory usage tracking, SIMD efficiency, and cache performance.

use std::time::Duration;
use std::collections::HashMap;

/// Performance metrics for a single operation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Operation execution time
    pub operation_time: Duration,

    /// Memory allocated (in bytes)
    pub memory_allocated: usize,

    /// Memory freed (in bytes)
    pub memory_freed: usize,

    /// Cache misses (hardware counter if available)
    pub cache_misses: u64,

    /// SIMD efficiency (0.0 to 1.0, where 1.0 is perfect SIMD utilization)
    pub simd_efficiency: f64,

    /// Parallel execution efficiency (0.0 to 1.0)
    pub parallel_efficiency: f64,

    /// Number of elements processed
    pub elements_processed: usize,

    /// Throughput (elements per second)
    pub throughput: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            operation_time: Duration::default(),
            memory_allocated: 0,
            memory_freed: 0,
            cache_misses: 0,
            simd_efficiency: 0.0,
            parallel_efficiency: 0.0,
            elements_processed: 0,
            throughput: 0.0,
        }
    }
}

impl PerformanceMetrics {
    /// Calculate throughput based on operation time and elements processed
    pub fn calculate_throughput(&mut self) {
        if self.operation_time.as_secs_f64() > 0.0 && self.elements_processed > 0 {
            self.throughput = self.elements_processed as f64 / self.operation_time.as_secs_f64();
        }
    }

    /// Get a summary of the metrics
    pub fn summary(&self) -> String {
        format!(
            "PerformanceMetrics {{ \
             time: {:?}, \
             memory: {}B allocated, {}B freed, \
             cache misses: {}, \
             SIMD efficiency: {:.2}%, \
             parallel efficiency: {:.2}%, \
             elements: {}, \
             throughput: {:.2} elements/s \
             }}",
            self.operation_time,
            self.memory_allocated,
            self.memory_freed,
            self.cache_misses,
            self.simd_efficiency * 100.0,
            self.parallel_efficiency * 100.0,
            self.elements_processed,
            self.throughput
        )
    }
}

/// Bottleneck identified during performance analysis
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Name of the operation with the bottleneck
    pub operation: String,

    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,

    /// Severity (0.0 to 1.0)
    pub severity: f64,

    /// Description of the bottleneck
    pub description: String,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq)]
pub enum BottleneckType {
    /// Memory allocation bottleneck
    MemoryAllocation,

    /// Cache inefficiency
    CacheMisses,

    /// Low SIMD utilization
    SimdInefficiency,

    /// Poor parallel scaling
    ParallelInefficiency,

    /// Slow computation
    SlowComputation,

    /// Excessive memory copies
    MemoryCopies,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Operation this recommendation applies to
    pub operation: String,

    /// Priority (0.0 to 1.0, higher is more important)
    pub priority: f64,

    /// Description of the optimization
    pub description: String,

    /// Expected improvement factor (e.g., 2.0 for 2x speedup)
    pub expected_improvement: f64,
}

/// Performance report containing analysis and recommendations
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,

    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,

    /// Efficiency scores for each operation (0.0 to 1.0)
    pub efficiency_scores: HashMap<String, f64>,

    /// Overall efficiency score
    pub overall_efficiency: f64,
}

impl PerformanceReport {
    /// Create a new empty performance report
    pub fn new() -> Self {
        Self {
            bottlenecks: Vec::new(),
            recommendations: Vec::new(),
            efficiency_scores: HashMap::new(),
            overall_efficiency: 0.0,
        }
    }

    /// Add a bottleneck to the report
    pub fn add_bottleneck(&mut self, bottleneck: Bottleneck) {
        self.bottlenecks.push(bottleneck);
    }

    /// Add a recommendation to the report
    pub fn add_recommendation(&mut self, recommendation: OptimizationRecommendation) {
        self.recommendations.push(recommendation);
    }

    /// Set efficiency score for an operation
    pub fn set_efficiency_score(&mut self, operation: String, score: f64) {
        self.efficiency_scores.insert(operation, score);
        self.recalculate_overall_efficiency();
    }

    /// Recalculate overall efficiency from individual scores
    pub fn recalculate_overall_efficiency(&mut self) {
        if self.efficiency_scores.is_empty() {
            self.overall_efficiency = 0.0;
        } else {
            let sum: f64 = self.efficiency_scores.values().sum();
            self.overall_efficiency = sum / self.efficiency_scores.len() as f64;
        }
    }

    /// Get a summary of the report
    pub fn summary(&self) -> String {
        format!(
            "PerformanceReport {{ \
             bottlenecks: {}, \
             recommendations: {}, \
             operations tracked: {}, \
             overall efficiency: {:.2}% \
             }}",
            self.bottlenecks.len(),
            self.recommendations.len(),
            self.efficiency_scores.len(),
            self.overall_efficiency * 100.0
        )
    }
}

impl Default for PerformanceReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory tracking for allocations and deallocations
#[derive(Debug, Clone, Default)]
pub struct MemoryTracker {
    /// Total bytes allocated
    pub allocated: usize,

    /// Total bytes freed
    pub freed: usize,

    /// Current memory usage (allocated - freed)
    pub current_usage: usize,
}

impl MemoryTracker {
    /// Create a new memory tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an allocation
    pub fn record_allocation(&mut self, size: usize) {
        self.allocated += size;
        self.current_usage += size;
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        self.freed += size;
        self.current_usage = self.current_usage.saturating_sub(size);
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.allocated.max(self.current_usage)
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_default() {
        let metrics = PerformanceMetrics::default();
        assert_eq!(metrics.operation_time, Duration::default());
        assert_eq!(metrics.memory_allocated, 0);
        assert_eq!(metrics.throughput, 0.0);
    }

    #[test]
    fn test_performance_metrics_throughput_calculation() {
        let mut metrics = PerformanceMetrics {
            operation_time: Duration::from_secs(1),
            elements_processed: 1000,
            ..Default::default()
        };
        metrics.calculate_throughput();
        assert_eq!(metrics.throughput, 1000.0);
    }

    #[test]
    fn test_memory_tracker() {
        let mut tracker = MemoryTracker::new();
        tracker.record_allocation(100);
        tracker.record_allocation(50);
        tracker.record_deallocation(30);

        assert_eq!(tracker.allocated, 150);
        assert_eq!(tracker.freed, 30);
        assert_eq!(tracker.current_usage, 120);
    }

    #[test]
    fn test_performance_report() {
        let mut report = PerformanceReport::new();
        report.set_efficiency_score("add".to_string(), 0.8);
        report.set_efficiency_score("multiply".to_string(), 0.9);

        assert_eq!(report.efficiency_scores.len(), 2);
        assert!((report.overall_efficiency - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_bottleneck_creation() {
        let bottleneck = Bottleneck {
            operation: "add".to_string(),
            bottleneck_type: BottleneckType::SimdInefficiency,
            severity: 0.7,
            description: "Low SIMD utilization detected".to_string(),
        };

        assert_eq!(bottleneck.bottleneck_type, BottleneckType::SimdInefficiency);
        assert_eq!(bottleneck.severity, 0.7);
    }

    #[test]
    fn test_optimization_recommendation() {
        let rec = OptimizationRecommendation {
            operation: "add".to_string(),
            priority: 0.9,
            description: "Enable SIMD optimizations".to_string(),
            expected_improvement: 2.0,
        };

        assert_eq!(rec.operation, "add");
        assert_eq!(rec.priority, 0.9);
        assert_eq!(rec.expected_improvement, 2.0);
    }
}
