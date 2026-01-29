use crate::error::{NumPyError, Result};
use crate::ufunc::ArrayView;
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
            UfuncType::Add => "add",
            UfuncType::Subtract => "subtract",
            UfuncType::Multiply => "multiply",
            UfuncType::Divide => "divide",
            UfuncType::Negative => "negative",
            UfuncType::Absolute => "absolute",
            UfuncType::Greater => "greater",
            UfuncType::Less => "less",
            UfuncType::Equal => "equal",
            UfuncType::NotEqual => "not_equal",
            UfuncType::LogicalAnd => "logical_and",
            UfuncType::LogicalOr => "logical_or",
            UfuncType::LogicalNot => "logical_not",
            UfuncType::Maximum => "maximum",
            UfuncType::Minimum => "minimum",
        }
    }
}

/// UfuncKernel trait for dtype-specific optimizations
/// This trait provides a specialized interface for high-performance kernel implementations
impl std::str::FromStr for UfuncType {
    type Err = crate::error::NumPyError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "add" => Ok(UfuncType::Add),
            "subtract" => Ok(UfuncType::Subtract),
            "multiply" => Ok(UfuncType::Multiply),
            "divide" => Ok(UfuncType::Divide),
            "negative" => Ok(UfuncType::Negative),
            "absolute" => Ok(UfuncType::Absolute),
            "greater" => Ok(UfuncType::Greater),
            "less" => Ok(UfuncType::Less),
            "equal" => Ok(UfuncType::Equal),
            "not_equal" => Ok(UfuncType::NotEqual),
            "logical_and" => Ok(UfuncType::LogicalAnd),
            "logical_or" => Ok(UfuncType::LogicalOr),
            "logical_not" => Ok(UfuncType::LogicalNot),
            "maximum" => Ok(UfuncType::Maximum),
            "minimum" => Ok(UfuncType::Minimum),
            _ => Err(crate::error::NumPyError::value_error(
                format!("Unknown ufunc: {}", s),
                "",
            )),
        }
    }
}

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

/// Trait for type-erased kernel storage
pub trait ErasedKernelTrait: Send + Sync {
    fn name(&self) -> &str;
    fn ufunc_type(&self) -> UfuncType;
    fn is_vectorized(&self) -> bool;
    fn performance_hint(&self) -> UfuncPerformanceHint;
    fn layout_preference(&self) -> ArrayLayoutPreference;
    fn item_type_id(&self) -> std::any::TypeId;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Type-erased kernel container
pub struct ErasedUfuncKernel<T> {
    kernel: Box<dyn UfuncKernel<T> + Send + Sync>,
    name: String,
    ufunc_type: UfuncType,
    is_vectorized: bool,
    performance_hint: UfuncPerformanceHint,
    layout_preference: ArrayLayoutPreference,
    _marker: std::marker::PhantomData<T>,
}

impl<T: 'static + Send + Sync> ErasedKernelTrait for ErasedUfuncKernel<T> {
    fn name(&self) -> &str {
        &self.name
    }
    fn ufunc_type(&self) -> UfuncType {
        self.ufunc_type
    }
    fn is_vectorized(&self) -> bool {
        self.is_vectorized
    }
    fn performance_hint(&self) -> UfuncPerformanceHint {
        self.performance_hint
    }
    fn layout_preference(&self) -> ArrayLayoutPreference {
        self.layout_preference
    }
    fn item_type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<T>()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T: 'static> ErasedUfuncKernel<T> {
    /// Create a new type-erased kernel
    pub fn new<K: UfuncKernel<T> + 'static>(kernel: K, ufunc_type: UfuncType) -> Self {
        let name = kernel.name().to_string();
        Self {
            is_vectorized: kernel.is_vectorized(),
            performance_hint: kernel.performance_hint(),
            layout_preference: kernel.layout_preference(),
            kernel: Box::new(kernel),
            name,
            ufunc_type,
            _marker: std::marker::PhantomData,
        }
    }

    /// Downcast kernel to concrete type
    pub fn downcast_ref(&self) -> &dyn UfuncKernel<T> {
        &*self.kernel
    }
}

/// Kernel registry for ufunc-specific kernels
/// 
/// This registry stores dtype-specific kernels optimized for different operations.
/// Kernels are indexed by (TypeId, UfuncType) for efficient lookup.
pub struct UfuncKernelRegistry {
    /// Map from (TypeId, UfuncType) to kernel implementations
    kernels: RwLock<HashMap<(std::any::TypeId, UfuncType), Box<dyn ErasedKernelTrait>>>,
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
    pub fn register<T: 'static + Send + Sync, K: UfuncKernel<T> + 'static>(
        &mut self,
        ufunc_type: UfuncType,
        kernel: K,
    ) -> Result<()> {
        let type_id = std::any::TypeId::of::<T>();
        let key = (type_id, ufunc_type);
        let erased = ErasedUfuncKernel::<T>::new(kernel, ufunc_type);

        let mut kernels = self.kernels.write().map_err(|_| {
            NumPyError::internal_error("Failed to acquire write lock for kernel registry")
        })?;

        kernels.insert(key, Box::new(erased));

        let mut cache = self.selection_cache.write().map_err(|_| {
            NumPyError::internal_error("Failed to acquire write lock for selection cache")
        })?;
        
        cache.retain(|&(tid, _), _| tid != type_id);
        
        Ok(())
    }
    
    /// Get a kernel for a specific dtype and ufunc type
    pub fn with_kernel<T: 'static, R, F>(&self, ufunc_type: UfuncType, f: F) -> Option<R>
    where
        F: FnOnce(&dyn UfuncKernel<T>) -> R,
    {
        let type_id = std::any::TypeId::of::<T>();
        let kernels = self.kernels.read().ok()?;

        let kernel_any = kernels.get(&(type_id, ufunc_type))?;
        let erased = kernel_any.as_any().downcast_ref::<ErasedUfuncKernel<T>>()?;
        Some(f(&*erased.kernel))
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

            let a_exact = a_kernel.item_type_id() == type_id;
            let b_exact = b_kernel.item_type_id() == type_id;

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

        if let Some((_, kernel) ) = candidates.first() {
            let kernel_name = kernel.name().to_string();
            let needs_cast = kernel.item_type_id() != type_id;

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
pub fn with_ufunc_kernel<T: 'static, R, F>(ufunc_type: UfuncType, f: F) -> Option<R>
where
    F: FnOnce(&dyn UfuncKernel<T>) -> R,
{
    UFUNC_KERNEL_REGISTRY.with_kernel(ufunc_type, f)
}

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