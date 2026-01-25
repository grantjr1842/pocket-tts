use crate::array::Array;
use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Kernel trait for dynamic registration
pub trait Kernel: Send + Sync {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Get kernel signature (input types -> output types)
    fn signature(&self) -> KernelSignature;

    /// Execute the kernel
    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
    ) -> Result<()>;

    /// Check if kernel supports specific dtypes
    fn supports_dtypes(&self, input_dtypes: &[Dtype]) -> bool {
        self.signature().input_types.len() == input_dtypes.len()
            && self
                .signature()
                .input_types
                .iter()
                .zip(input_dtypes.iter())
                .all(|(expected, actual)| expected.can_cast_to(actual))
    }

    /// Get performance hint for optimization
    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::General
    }
}

/// Kernel signature defining input and output types
#[derive(Debug, Clone)]
pub struct KernelSignature {
    pub input_types: Vec<Dtype>,
    pub output_types: Vec<Dtype>,
}

impl KernelSignature {
    pub fn new(input_types: Vec<Dtype>, output_types: Vec<Dtype>) -> Self {
        Self {
            input_types,
            output_types,
        }
    }

    pub fn matches_input(&self, input_dtypes: &[Dtype]) -> bool {
        if self.input_types.len() != input_dtypes.len() {
            return false;
        }

        self.input_types
            .iter()
            .zip(input_dtypes.iter())
            .all(|(expected, actual)| expected.can_cast_to(actual))
    }
}

/// Performance hints for kernel optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceHint {
    /// General purpose kernel
    General,
    /// Vectorized operations (SIMD friendly)
    Vectorized,
    /// Memory bandwidth bound
    MemoryBound,
    /// Compute bound
    ComputeBound,
    /// Small input size optimized
    SmallInput,
    /// Large input size optimized
    LargeInput,
}

/// Dynamic kernel registry with runtime registration
pub struct DynamicKernelRegistry {
    /// Map from kernel name to multiple implementations
    kernels: RwLock<HashMap<String, Vec<Arc<dyn Kernel>>>>,
    /// Optimization cache for dtype-specific lookups
    optimization_cache: RwLock<HashMap<String, HashMap<Vec<Dtype>, Arc<dyn Kernel>>>>,
}

impl DynamicKernelRegistry {
    /// Create new dynamic registry
    pub fn new() -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
            optimization_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new kernel implementation
    pub fn register(&self, kernel: Arc<dyn Kernel>) -> Result<()> {
        let name = kernel.name().to_string();
        let mut kernels = self.kernels.write().map_err(|_| {
            NumPyError::internal_error("Failed to acquire write lock for kernel registry")
        })?;

        kernels.entry(name.clone()).or_default().push(kernel);

        // Clear optimization cache for this kernel name
        let mut cache = self.optimization_cache.write().map_err(|_| {
            NumPyError::internal_error("Failed to acquire write lock for optimization cache")
        })?;
        cache.remove(&name);

        Ok(())
    }

    /// Find best kernel for given name and input dtypes
    pub fn find_kernel(
        &self,
        name: &str,
        input_dtypes: &[Dtype],
    ) -> Result<Option<Arc<dyn Kernel>>> {
        // Check optimization cache first
        {
            let cache = self.optimization_cache.read().map_err(|_| {
                NumPyError::internal_error("Failed to acquire read lock for optimization cache")
            })?;

            if let Some(kernel_cache) = cache.get(name) {
                if let Some(kernel) = kernel_cache.get(input_dtypes) {
                    return Ok(Some(kernel.clone()));
                }
            }
        }

        // Search through available kernels
        let kernels = self.kernels.read().map_err(|_| {
            NumPyError::internal_error("Failed to acquire read lock for kernel registry")
        })?;

        if let Some(implementations) = kernels.get(name) {
            // Find kernels that support the input dtypes
            let mut candidates: Vec<_> = implementations
                .iter()
                .filter(|kernel| kernel.supports_dtypes(input_dtypes))
                .collect();

            if candidates.is_empty() {
                return Ok(None);
            }

            // Select best kernel based on performance hints and dtype compatibility
            let best_kernel = self.select_best_kernel(&candidates, input_dtypes);

            // Cache the result
            if let Some(ref kernel) = best_kernel {
                let mut cache = self.optimization_cache.write().map_err(|_| {
                    NumPyError::internal_error(
                        "Failed to acquire write lock for optimization cache",
                    )
                })?;

                let kernel_cache = cache.entry(name.to_string()).or_default();
                kernel_cache.insert(input_dtypes.to_vec(), kernel.clone());
            }

            Ok(best_kernel)
        } else {
            Ok(None)
        }
    }

    /// Select best kernel based on performance characteristics
    fn select_best_kernel(
        &self,
        candidates: &[&Arc<dyn Kernel>],
        input_dtypes: &[Dtype],
    ) -> Option<Arc<dyn Kernel>> {
        if candidates.len() == 1 {
            return Some(candidates[0].clone());
        }

        // Scoring system for kernel selection
        let mut best_score = 0;
        let mut best_kernel = None;

        for candidate in candidates {
            let mut score = 0;

            // Prefer exact dtype matches over casts
            for (expected, actual) in candidate
                .signature()
                .input_types
                .iter()
                .zip(input_dtypes.iter())
            {
                if expected == actual {
                    score += 10; // Exact match bonus
                } else if expected.can_cast_to(actual) {
                    score += 5; // Castable match
                }
            }

            // Performance hint bonuses
            match candidate.performance_hint() {
                PerformanceHint::Vectorized => score += 20,
                PerformanceHint::SmallInput if self.is_small_input(input_dtypes) => score += 15,
                PerformanceHint::LargeInput if !self.is_small_input(input_dtypes) => score += 15,
                PerformanceHint::MemoryBound if self.is_memory_bound(input_dtypes) => score += 10,
                PerformanceHint::ComputeBound if self.is_compute_bound(input_dtypes) => score += 10,
                _ => {}
            }

            if score > best_score {
                best_score = score;
                best_kernel = Some(candidate.clone());
            }
        }

        best_kernel
    }

    /// Helper to determine if input is small
    fn is_small_input(&self, _input_dtypes: &[Dtype]) -> bool {
        // This could be enhanced with actual size information
        // For now, assume small if all types are primitive
        _input_dtypes.iter().all(|dt| dt.kind().is_primitive())
    }

    /// Helper to determine if operation is memory bound
    fn is_memory_bound(&self, _input_dtypes: &[Dtype]) -> bool {
        // Simple heuristic: operations on larger types are more memory bound
        _input_dtypes
            .iter()
            .any(|dt| dt.kind() == DtypeKind::Complex)
    }

    /// Helper to determine if operation is compute bound
    fn is_compute_bound(&self, _input_dtypes: &[Dtype]) -> bool {
        // Simple heuristic: operations on floats are more compute bound
        _input_dtypes.iter().any(|dt| dt.kind() == DtypeKind::Float)
    }

    /// List all registered kernel names
    pub fn list_kernels(&self) -> Result<Vec<String>> {
        let kernels = self.kernels.read().map_err(|_| {
            NumPyError::internal_error("Failed to acquire read lock for kernel registry")
        })?;

        Ok(kernels.keys().cloned().collect())
    }

    /// Get all implementations for a kernel name
    pub fn get_implementations(&self, name: &str) -> Result<Vec<Arc<dyn Kernel>>> {
        let kernels = self.kernels.read().map_err(|_| {
            NumPyError::internal_error("Failed to acquire read lock for kernel registry")
        })?;

        Ok(kernels
            .get(name)
            .map_or_else(Vec::new, |impls| impls.clone()))
    }

    /// Clear optimization cache (useful for testing or when kernels are updated)
    pub fn clear_cache(&self) -> Result<()> {
        let mut cache = self.optimization_cache.write().map_err(|_| {
            NumPyError::internal_error("Failed to acquire write lock for optimization cache")
        })?;
        cache.clear();
        Ok(())
    }

    /// Get registry statistics
    pub fn stats(&self) -> Result<RegistryStats> {
        let kernels = self.kernels.read().map_err(|_| {
            NumPyError::internal_error("Failed to acquire read lock for kernel registry")
        })?;

        let cache = self.optimization_cache.read().map_err(|_| {
            NumPyError::internal_error("Failed to acquire read lock for optimization cache")
        })?;

        let total_kernels: usize = kernels.values().map(|v| v.len()).sum();
        let cache_entries: usize = cache.values().map(|m| m.len()).sum();

        Ok(RegistryStats {
            kernel_names: kernels.len(),
            total_implementations: total_kernels,
            cache_entries,
        })
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub kernel_names: usize,
    pub total_implementations: usize,
    pub cache_entries: usize,
}

impl Default for DynamicKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global dynamic kernel registry
lazy_static::lazy_static! {
    pub static ref DYNAMIC_KERNEL_REGISTRY: DynamicKernelRegistry = DynamicKernelRegistry::new();
}

/// Register a kernel globally
pub fn register_kernel(kernel: Arc<dyn Kernel>) -> Result<()> {
    DYNAMIC_KERNEL_REGISTRY.register(kernel)
}

/// Find a kernel globally
pub fn find_kernel(name: &str, input_dtypes: &[Dtype]) -> Result<Option<Arc<dyn Kernel>>> {
    DYNAMIC_KERNEL_REGISTRY.find_kernel(name, input_dtypes)
}

/// List all kernel names globally
pub fn list_kernels() -> Result<Vec<String>> {
    DYNAMIC_KERNEL_REGISTRY.list_kernels()
}

/// Get registry statistics globally
pub fn get_registry_stats() -> Result<RegistryStats> {
    DYNAMIC_KERNEL_REGISTRY.stats()
}

// Re-export ArrayView traits
use crate::ufunc::{ArrayView, ArrayViewMut};
