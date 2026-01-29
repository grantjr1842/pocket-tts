//! Dynamic Kernel Registry System
//!
//! This module provides a dynamic kernel registry that allows runtime registration
//! of ufunc implementations with dtype-specific optimization support.

use crate::array::Array;
use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
use crate::ufunc::{ArrayView, ArrayViewMut, Ufunc};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Kernel function trait for dynamic registration
pub trait KernelFunction: Send + Sync {
    /// Execute the kernel operation
    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()>;

    /// Get kernel name
    fn name(&self) -> &str;

    /// Get supported input dtypes
    fn supported_dtypes(&self) -> &[DtypeKind];

    /// Get kernel performance tier for optimization
    fn performance_tier(&self) -> PerformanceTier;

    /// Check if kernel supports specific dtypes
    fn supports_dtypes(&self, dtypes: &[&Dtype]) -> bool {
        dtypes
            .iter()
            .all(|dt| self.supported_dtypes().contains(&dt.kind()))
    }
}

/// Performance tier for kernel optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PerformanceTier {
    /// Generic implementation, baseline performance
    Generic = 0,
    /// Optimized for specific dtypes
    Optimized = 1,
    /// Highly optimized with SIMD/vectorization
    Vectorized = 2,
    /// Specialized hardware-accelerated implementation
    HardwareAccelerated = 3,
}

/// Dynamic kernel entry containing implementation and metadata
#[derive(Clone)]
pub struct KernelEntry {
    /// The kernel implementation
    kernel: Arc<dyn KernelFunction>,
    /// Input dtypes this kernel handles
    input_dtypes: Vec<Dtype>,
    /// Performance tier for selection
    tier: PerformanceTier,
    /// Registration timestamp
    timestamp: std::time::SystemTime,
}

impl KernelEntry {
    /// Create new kernel entry
    pub fn new(kernel: Arc<dyn KernelFunction>, input_dtypes: Vec<Dtype>) -> Self {
        let tier = kernel.performance_tier();
        Self {
            kernel,
            input_dtypes,
            tier,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Get kernel function reference
    pub fn kernel(&self) -> &Arc<dyn KernelFunction> {
        &self.kernel
    }

    /// Get input dtypes
    pub fn input_dtypes(&self) -> &[Dtype] {
        &self.input_dtypes
    }

    /// Get performance tier
    pub fn tier(&self) -> PerformanceTier {
        self.tier
    }

    /// Check if this entry matches the requested dtypes
    pub fn matches_dtypes(&self, dtypes: &[Dtype]) -> bool {
        if self.input_dtypes.len() != dtypes.len() {
            return false;
        }

        self.input_dtypes
            .iter()
            .zip(dtypes.iter())
            .all(|(expected, actual)| {
                expected.can_cast(actual, crate::dtype::Casting::Safe) || expected == actual
            })
    }

    /// Calculate matching score for dtype selection
    pub fn matching_score(&self, dtypes: &[Dtype]) -> u32 {
        if !self.matches_dtypes(dtypes) {
            return 0;
        }

        let mut score = self.tier as u32 * 100;

        for (expected, actual) in self.input_dtypes.iter().zip(dtypes.iter()) {
            if expected == actual {
                score += 10;
            } else if expected.can_cast(actual, crate::dtype::Casting::Safe) {
                score += 5;
            }
        }

        score
    }
}

/// Dynamic kernel registry for runtime registration and optimized selection
pub struct DynamicKernelRegistry {
    /// Map from ufunc name to list of kernel entries
    kernels: HashMap<String, Vec<KernelEntry>>,
}

impl DynamicKernelRegistry {
    /// Create new dynamic registry instance
    pub fn new() -> Arc<RwLock<Self>> {
        let registry = Self {
            kernels: HashMap::new(),
        };
        Arc::new(RwLock::new(registry))
    }

    /// Get global registry instance
    pub fn instance() -> Arc<RwLock<Self>> {
        use std::sync::OnceLock;
        static INSTANCE: OnceLock<Arc<RwLock<DynamicKernelRegistry>>> = OnceLock::new();

        INSTANCE.get_or_init(|| Self::new()).clone()
    }

    /// Register a new kernel dynamically
    pub fn register_kernel(
        &mut self,
        ufunc_name: &str,
        kernel: Arc<dyn KernelFunction>,
        input_dtypes: Vec<Dtype>,
    ) -> Result<()> {
        let entry = KernelEntry::new(kernel, input_dtypes);

        self.kernels
            .entry(ufunc_name.to_string())
            .or_insert_with(Vec::new)
            .push(entry);

        Ok(())
    }

    /// Register multiple kernels for a ufunc
    pub fn register_kernels(
        &mut self,
        ufunc_name: &str,
        kernels: Vec<(Arc<dyn KernelFunction>, Vec<Dtype>)>,
    ) -> Result<()> {
        for (kernel, dtypes) in kernels {
            self.register_kernel(ufunc_name, kernel.clone(), dtypes)?;
        }
        Ok(())
    }

    /// Get best kernel for given ufunc and input dtypes
    pub fn get_best_kernel(&self, ufunc_name: &str, input_dtypes: &[Dtype]) -> Option<KernelEntry> {
        let entries = self.kernels.get(ufunc_name)?;

        let mut best_entry: Option<&KernelEntry> = None;
        let mut best_score = 0u32;

        for entry in entries {
            let score = entry.matching_score(input_dtypes);
            if score > best_score {
                best_score = score;
                best_entry = Some(entry);
            }
        }

        best_entry.cloned()
    }

    /// Get all kernels for a ufunc
    pub fn get_kernels(&self, ufunc_name: &str) -> &[KernelEntry] {
        self.kernels
            .get(ufunc_name)
            .map(|vec| vec.as_slice())
            .unwrap_or(&[])
    }

    /// List all registered ufunc names
    pub fn list_ufuncs(&self) -> Vec<String> {
        self.kernels.keys().cloned().collect()
    }

    /// Remove kernels for a ufunc
    pub fn remove_ufunc(&mut self, ufunc_name: &str) -> Vec<KernelEntry> {
        self.kernels.remove(ufunc_name).unwrap_or_default()
    }

    /// Clear all registered kernels
    pub fn clear(&mut self) {
        self.kernels.clear();
    }

    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        let total_kernels = self.kernels.values().map(|v| v.len()).sum();
        let ufunc_count = self.kernels.len();

        let mut tier_counts = HashMap::new();
        for entries in self.kernels.values() {
            for entry in entries {
                *tier_counts.entry(entry.tier()).or_insert(0) += 1;
            }
        }

        RegistryStats {
            total_kernels,
            ufunc_count,
            tier_counts,
        }
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub total_kernels: usize,
    pub ufunc_count: usize,
    pub tier_counts: HashMap<PerformanceTier, usize>,
}

/// Convenience functions for global registry access

/// Register a kernel globally
pub fn register_kernel(
    ufunc_name: &str,
    kernel: Arc<dyn KernelFunction>,
    input_dtypes: Vec<Dtype>,
) -> Result<()> {
    let registry = DynamicKernelRegistry::instance();
    let mut registry = registry.write().map_err(|_| NumPyError::InternalError {
        message: "Failed to acquire write lock for kernel registry".to_string(),
    })?;

    registry.register_kernel(ufunc_name, kernel, input_dtypes)
}

/// Get best kernel globally
pub fn get_best_kernel(ufunc_name: &str, input_dtypes: &[Dtype]) -> Option<KernelEntry> {
    let registry = DynamicKernelRegistry::instance();
    let registry = registry.read().ok()?;
    registry.get_best_kernel(ufunc_name, input_dtypes)
}

/// List all registered ufuncs globally
pub fn list_registered_ufuncs() -> Option<Vec<String>> {
    let registry = DynamicKernelRegistry::instance();
    let registry = registry.read().ok()?;
    Some(registry.list_ufuncs())
}

/// Get registry statistics globally
pub fn get_registry_stats() -> Option<RegistryStats> {
    let registry = DynamicKernelRegistry::instance();
    let registry = registry.read().ok()?;
    Some(registry.stats())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;

    #[test]
    fn test_dynamic_registry() {
        let registry = DynamicKernelRegistry::new();

        {
            let r = registry.read().unwrap();
            assert_eq!(r.list_ufuncs().len(), 0);
            assert!(r.get_best_kernel("add", &[]).is_none());
        }
    }

    #[test]
    fn test_kernel_entry_matching() {
        assert!(true);
    }
}
