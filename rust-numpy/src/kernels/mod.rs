use crate::error::{NumPyError, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub mod dtype_kernels;

/// Ufunc operation types for kernel registration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UfuncType {
    Add,
    Subtract,
    Multiply,
    Divide,
    Negative,
    Absolute,
    Greater,
    Less,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
    LogicalNot,
    Maximum,
    Minimum,
}

impl UfuncType {
    /// Convert ufunc type to string name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Divide => "divide",
            Self::Negative => "negative",
            Self::Absolute => "absolute",
            Self::Greater => "greater",
            Self::Less => "less",
            Self::Equal => "equal",
            Self::NotEqual => "not_equal",
            Self::LogicalAnd => "logical_and",
            Self::LogicalOr => "logical_or",
            Self::LogicalNot => "logical_not",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
        }
    }
}

/// UfuncKernel trait for dtype-specific optimizations
/// This trait provides a specialized interface for high-performance kernel implementations
pub trait UfuncKernel<T>: Send + Sync {
    /// Get kernel name
    fn name(&self) -> &str;
    
    /// Execute kernel on input and output arrays
    /// 
    /// # Parameters
    /// - `input`: Array of input arrays (slices for each input)
    /// - `output`: Mutable slice for output
    /// 
    /// # Returns
    /// - `Ok(())` if execution succeeded
    /// - `Err(NumPyError)` if execution failed
    fn execute(&self, input: &[&[T]], output: &mut [T]) -> Result<()>;
    
    /// Check if kernel is vectorized (uses SIMD)
    fn is_vectorized(&self) -> bool { false }
    
    /// Get performance hint for this kernel
    fn performance_hint(&self) -> UfuncPerformanceHint { UfuncPerformanceHint::General }
    
    /// Get supported array layout preference
    fn layout_preference(&self) -> ArrayLayoutPreference { ArrayLayoutPreference::Any }
}

/// Performance hints for ufunc kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UfuncPerformanceHint {
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

/// Array layout preferences for kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayLayoutPreference {
    /// Any layout is fine
    Any,
    /// Prefer C-contiguous (row-major)
    CContiguous,
    /// Prefer F-contiguous (column-major)
    FContiguous,
    /// Require contiguous arrays
    Contiguous,
}

/// Type-erased kernel for storage in registry
pub struct ErasedUfuncKernel {
    kernel: Box<dyn std::any::Any + Send + Sync>,
    type_id: std::any::TypeId,
    name: String,
    ufunc_type: UfuncType,
    is_vectorized: bool,
    performance_hint: UfuncPerformanceHint,
    layout_preference: ArrayLayoutPreference,
}

impl ErasedUfuncKernel {
    /// Create a new type-erased kernel
    pub fn new<T: 'static, K: UfuncKernel<T> + 'static>(kernel: K, ufunc_type: UfuncType) -> Self {
        let name = kernel.name().to_string();
        let is_vectorized = kernel.is_vectorized();
        let performance_hint = kernel.performance_hint();
        let layout_preference = kernel.layout_preference();

        Self {
            kernel: Box::new(kernel),
            type_id: std::any::TypeId::of::<T>(),
            name,
            ufunc_type,
            is_vectorized,
            performance_hint,
            layout_preference,
        }
    }
    
    /// Get the type ID of the kernel
    pub const fn type_id(&self) -> std::any::TypeId  { self.type_id }
    
    /// Get the ufunc type
    pub const fn ufunc_type(&self) -> UfuncType { self.ufunc_type }
    
    /// Get kernel name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Check if kernel is vectorized
    pub const fn is_vectorized(&self) -> bool { self.is_vectorized }
    
    /// Get performance hint
    pub const fn performance_hint(&self) -> UfuncPerformanceHint { self.performance_hint }
    
    /// Get layout preference
    pub const fn layout_preference(&self) -> ArrayLayoutPreference { self.layout_preference }
    
    /// Downcast kernel to concrete type
    ///
    /// TODO: This method needs architectural redesign. We cannot safely cast from
    /// `dyn Any` to `dyn UfuncKernel<T>` because they have different vtables.
    /// The kernel storage approach needs to be redesigned to support proper retrieval.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&dyn UfuncKernel<T>> {
        // Cannot safely cast between trait objects with different vtables
        // This requires redesign of the kernel storage approach
        None
    }
}

/// Kernel registry for ufunc-specific kernels
/// 
/// This registry stores dtype-specific kernels optimized for different operations.
/// Kernels are indexed by (TypeId, UfuncType) for efficient lookup.
pub struct UfuncKernelRegistry {
    /// Map from (TypeId, UfuncType) to kernel implementations
    kernels: RwLock<HashMap<(std::any::TypeId, UfuncType), ErasedUfuncKernel>>,
    /// Performance metrics for kernels
    metrics: RwLock<HashMap<String, UfuncKernelMetrics>>,
    /// Kernel selection cache
    selection_cache: RwLock<HashMap<(std::any::TypeId, UfuncType), String>>,
}

impl UfuncKernelRegistry {
    /// Create a new ufunc kernel registry
    pub fn new() -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
            metrics: RwLock::new(HashMap::new()),
            selection_cache: RwLock::new(HashMap::new()),
        }
    }
    
    /// Register a kernel for a specific dtype and ufunc type
    pub fn register<T: 'static, K: UfuncKernel<T> + 'static>(
        &mut self,
        ufunc_type: UfuncType,
        kernel: K,
    ) -> Result<()> {
        let type_id = std::any::TypeId::of::<T>();
        let key = (type_id, ufunc_type);
        let erased = ErasedUfuncKernel::new(kernel, ufunc_type);
        
        let mut kernels = self.kernels.write().map_err(|_| {
            NumPyError::internal_error("Failed to acquire write lock for kernel registry")
        })?;
        
        kernels.insert(key, erased);
        
        let mut cache = self.selection_cache.write().map_err(|_| {
            NumPyError::internal_error("Failed to acquire write lock for selection cache")
        })?;
        
        cache.retain(|&(tid, _), _| tid != type_id);
        
        Ok(())
    }
    
    /// Get a kernel name for a specific dtype and ufunc type
    pub fn get_kernel_name<T: 'static>(&self, ufunc_type: UfuncType) -> Option<String> {
        let type_id = std::any::TypeId::of::<T>();
        let kernels = self.kernels.read().ok()?;

        kernels.get(&(type_id, ufunc_type)).map(|k| k.name().to_string())
    }

    /// Check if a kernel exists for a specific dtype and ufunc type
    pub fn has_kernel<T: 'static>(&self, ufunc_type: UfuncType) -> bool {
        let type_id = std::any::TypeId::of::<T>();
        let kernels = self.kernels.read().ok();

        match kernels {
            Some(k) => k.contains_key(&(type_id, ufunc_type)),
            None => false,
        }
    }

    /// Find best available kernel for a dtype and ufunc type
    /// This may return a kernel that requires type casting
    pub fn find_best_kernel<T: 'static>(&self, ufunc_type: UfuncType) -> Option<(String, bool)> {
        let type_id = std::any::TypeId::of::<T>();
        
        {
            let cache = self.selection_cache.read().ok()?;
            if let Some(kernel_name) = cache.get(&(type_id, ufunc_type)) {
                return Some((kernel_name.clone(), false));
            }
        }
        
        let kernels = self.kernels.read().ok()?;
        let mut candidates: Vec<_> = kernels
            .iter()
            .filter(|((tid, ut), _)| *ut == ufunc_type && self.is_compatible_type(*tid, type_id))
            .collect();
        
        if candidates.is_empty() {
            return None;
        }
        
        candidates.sort_by(|a, b| {
            let a_kernel = &a.1;
            let b_kernel = &b.1;
            
            let a_exact = a_kernel.type_id() == type_id;
            let b_exact = b_kernel.type_id() == type_id;
            
            match (a_exact, b_exact) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => {
                    match (a_kernel.is_vectorized(), b_kernel.is_vectorized()) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => std::cmp::Ordering::Equal,
                    }
                }
            }
        });
        
        if let Some(&(_, kernel)) = candidates.first() {
            let kernel_name = kernel.name().to_string();
            let needs_cast = kernel.type_id() != type_id;
            
            if let Ok(mut cache) = self.selection_cache.write() {
                cache.insert((type_id, ufunc_type), kernel_name.clone());
            }
            
            Some((kernel_name, needs_cast))
        } else {
            None
        }
    }
    
    /// Check if a kernel type is compatible with a target type
    fn is_compatible_type(&self, kernel_type: std::any::TypeId, target_type: std::any::TypeId) -> bool {
        if kernel_type == target_type {
            return true;
        }
        
        false
    }
    
    /// List all registered kernel names for a ufunc type
    pub fn list_kernels(&self, ufunc_type: UfuncType) -> Result<Vec<String>> {
        let kernels = self.kernels.read().map_err(|_| {
            NumPyError::internal_error("Failed to acquire read lock for kernel registry")
        })?;
        
        let names: Vec<String> = kernels
            .iter()
            .filter(|((_, ut), _)| *ut == ufunc_type)
            .map(|(_, kernel)| kernel.name().to_string())
            .collect();
        
        Ok(names)
    }
    
    /// Get registry statistics
    pub fn stats(&self) -> Result<UfuncKernelRegistryStats> {
        let kernels = self.kernels.read().map_err(|_| {
            NumPyError::internal_error("Failed to acquire read lock for kernel registry")
        })?;
        
        let total_kernels = kernels.len();
        let vectorized_kernels = kernels
            .values()
            .filter(|k| k.is_vectorized())
            .count();
        
        let mut ufunc_counts: HashMap<UfuncType, usize> = HashMap::new();
        for ((_, ufunc_type), _) in kernels.iter() {
            *ufunc_counts.entry(*ufunc_type).or_insert(0) += 1;
        }
        
        Ok(UfuncKernelRegistryStats {
            total_kernels,
            vectorized_kernels,
            ufunc_counts,
        })
    }
    
    /// Clear the selection cache
    pub fn clear_cache(&self) -> Result<()> {
        let mut cache = self.selection_cache.write().map_err(|_| {
            NumPyError::internal_error("Failed to acquire write lock for selection cache")
        })?;
        cache.clear();
        Ok(())
    }
}

/// Statistics for the ufunc kernel registry
#[derive(Debug, Clone)]
pub struct UfuncKernelRegistryStats {
    /// Total number of registered kernels
    pub total_kernels: usize,
    /// Number of vectorized kernels
    pub vectorized_kernels: usize,
    /// Count of kernels per ufunc type
    pub ufunc_counts: HashMap<UfuncType, usize>,
}

/// Performance metrics for ufunc kernels
#[derive(Debug, Clone)]
pub struct UfuncKernelMetrics {
    /// Number of times kernel was called
    pub call_count: u64,
    /// Total execution time in microseconds
    pub total_time_us: u64,
    /// Number of elements processed
    pub total_elements: u64,
}

impl UfuncKernelMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self {
            call_count: 0,
            total_time_us: 0,
            total_elements: 0,
        }
    }
    
    /// Get average execution time
    pub fn avg_time_us(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_time_us as f64 / self.call_count as f64
        }
    }
    
    /// Get throughput in elements per second
    pub fn throughput(&self) -> f64 {
        if self.total_time_us == 0 {
            0.0
        } else {
            (self.total_elements as f64 * 1_000_000.0) / self.total_time_us as f64
        }
    }
}

impl Default for UfuncKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Global ufunc kernel registry
lazy_static::lazy_static! {
    pub static ref UFUNC_KERNEL_REGISTRY: Arc<UfuncKernelRegistry> = Arc::new(UfuncKernelRegistry::new());
}

/// Register a kernel globally
pub fn register_ufunc_kernel<T: 'static, K: UfuncKernel<T> + 'static>(
    _ufunc_type: UfuncType,
    _kernel: K,
) -> Result<()> {
    Err(NumPyError::internal_error("Global registration not yet implemented"))
}

/// Get a kernel globally
/// TODO: This function needs redesign to handle lifetimes properly
// pub fn get_ufunc_kernel<T: 'static>(ufunc_type: UfuncType) -> Option<&'static dyn UfuncKernel<T>> {
//     UFUNC_KERNEL_REGISTRY.get(ufunc_type)
// }

/// Find best kernel globally
pub fn find_best_ufunc_kernel<T: 'static>(
    ufunc_type: UfuncType,
) -> Option<(String, bool)> {
    UFUNC_KERNEL_REGISTRY.find_best_kernel::<T>(ufunc_type)
}

/// List all kernels for a ufunc type globally
pub fn list_ufunc_kernels(ufunc_type: UfuncType) -> Result<Vec<String>> {
    UFUNC_KERNEL_REGISTRY.list_kernels(ufunc_type)
}

/// Get registry statistics globally
pub fn get_ufunc_registry_stats() -> Result<UfuncKernelRegistryStats> {
    UFUNC_KERNEL_REGISTRY.stats()
}