use crate::array::Array;
use crate::dtype::{Dtype, DtypeKind};
use crate::error::{NumPyError, Result};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};
use std::time::Instant;

impl<T: 'static> ArrayView for Array<T> {
    fn dtype(&self) -> &Dtype {
        self.dtype()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn strides(&self) -> &[isize] {
        self.strides()
    }

    fn size(&self) -> usize {
        self.size()
    }

    fn ndim(&self) -> usize {
        self.ndim()
    }

    fn is_contiguous(&self) -> bool {
        self.is_c_contiguous()
    }

    fn as_ptr(&self) -> *const u8 {
        // Get pointer to the underlying data
        self.data.as_ref().as_slice().as_ptr() as *const u8
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl<T: 'static> ArrayViewMut for Array<T> {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        // Get mutable pointer to the underlying data
        // SAFETY: We have &mut self, which guarantees exclusive access to the Array.
        // Even though the data is in an Arc, the &mut ensures no other references exist
        // that could modify the data. We use unsafe to bypass Arc's ref counting here.
        {
            let ptr = self.data.as_ref().as_slice().as_ptr() as *mut u8;
            // The cast from const to mut is safe because:
            // 1. We have &mut self, guaranteeing exclusive mutable access
            // 2. The Arc is not shared during ufunc execution (outputs are newly created)
            ptr
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Universal function trait - base for all NumPy ufuncs
pub trait Ufunc: Send + Sync {
    /// Get ufunc name
    fn name(&self) -> &'static str;

    /// Get number of inputs
    fn nin(&self) -> usize;

    /// Get number of outputs
    fn nout(&self) -> usize;

    /// Get supported input types
    fn supported_dtypes(&self) -> &[DtypeKind];

    /// Get the concrete type signature for this ufunc implementation
    /// Returns a unique identifier for the types this ufunc handles
    fn type_signature(&self) -> String;

    /// Execute ufunc on inputs
    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()>;

    /// Check if ufunc supports given dtypes
    fn supports_dtypes(&self, dtypes: &[&Dtype]) -> bool {
        dtypes
            .iter()
            .all(|dt| self.supported_dtypes().contains(&dt.kind()))
    }

    /// Check if this ufunc implementation matches the given concrete types
    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool;

    /// Get the specific input dtypes this ufunc accepts
    fn input_dtypes(&self) -> Vec<Dtype>;
}

/// Trait for viewing array data
pub trait ArrayView {
    /// Get dtype
    fn dtype(&self) -> &Dtype;

    /// Get shape
    fn shape(&self) -> &[usize];

    /// Get strides
    fn strides(&self) -> &[isize];

    /// Get total size
    fn size(&self) -> usize;

    /// Get base offset
    fn offset(&self) -> usize;

    /// Get number of dimensions
    fn ndim(&self) -> usize;

    /// Check if contiguous
    fn is_contiguous(&self) -> bool;

    /// Get raw data pointer
    fn as_ptr(&self) -> *const u8;

    /// Downcast to Any
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Trait for mutable array data
pub trait ArrayViewMut: ArrayView {
    /// Get mutable raw data pointer
    fn as_mut_ptr(&mut self) -> *mut u8;

    /// Downcast to Any (mutable)
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Binary operation ufunc
#[allow(dead_code)]
pub struct BinaryUfunc<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> BinaryUfunc<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T, T) -> T + Send + Sync,
{
    /// Create new binary ufunc
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Ufunc for BinaryUfunc<T, F>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn nin(&self) -> usize {
        2
    }

    fn nout(&self) -> usize {
        1
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[
            DtypeKind::Integer,
            DtypeKind::Unsigned,
            DtypeKind::Float,
            DtypeKind::Complex,
        ]
    }

    fn type_signature(&self) -> String {
        format!("{}({})", self.name, std::any::type_name::<T>())
    }

    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool {
        input_types.len() == 2 && input_types.iter().all(|&t| t == std::any::type_name::<T>())
    }

    fn input_dtypes(&self) -> Vec<Dtype> {
        println!(
            "  (MathBinaryUfunc for {}) input_dtypes called",
            std::any::type_name::<T>()
        );
        vec![Dtype::from_type::<T>(), Dtype::from_type::<T>()]
    }

    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                self.name(),
                format!(
                    "Expected 2 inputs, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let in0 = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let in1 = unsafe { &*(inputs[1] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };

        // Try kernel registry for type-specific optimization
        if let Some(kernel) =
            crate::kernel_registry::get_kernel_registry().get::<T>(crate::kernels::UfuncType::Add)
        {
            kernel.execute(&[in0, in1], &mut [output])?;
        } else {
            // Fall back to original implementation using iterators
            let broadcasted = crate::broadcasting::broadcast_arrays(&[in0, in1])?;
            let arr0 = &broadcasted[0];
            let arr1 = &broadcasted[1];

        for i in 0..output.size() {
            if where_mask
                .as_ref()
                .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
            {
                if let (Some(a), Some(b)) = (arr0.get(i), arr1.get(i)) {
                    output.set(i, (self.operation)(a.clone(), b.clone()))?;
                }
            }
        }

        Ok(())
    }
}

/// Unary operation ufunc
#[allow(dead_code)]
pub struct UnaryUfunc<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T) -> T + Send + Sync,
{
    name: &'static str,
    operation: F,
    phantom: PhantomData<T>,
}

impl<T, F> UnaryUfunc<T, F>
where
    T: Clone + Default + 'static,
    F: Fn(T) -> T + Send + Sync,
{
    /// Create new unary ufunc
    pub fn new(name: &'static str, operation: F) -> Self {
        Self {
            name,
            operation,
            phantom: PhantomData,
        }
    }
}

impl<T, F> Ufunc for UnaryUfunc<T, F>
where
    T: Clone + Default + 'static + Send + Sync,
    F: Fn(T) -> T + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn nin(&self) -> usize {
        1
    }

    fn nout(&self) -> usize {
        1
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[
            DtypeKind::Integer,
            DtypeKind::Unsigned,
            DtypeKind::Float,
            DtypeKind::Complex,
        ]
    }

    fn type_signature(&self) -> String {
        format!("{}({})", self.name, std::any::type_name::<T>())
    }

    fn matches_concrete_types(&self, input_types: &[&'static str]) -> bool {
        input_types.len() == 1 && input_types[0] == std::any::type_name::<T>()
    }

    fn input_dtypes(&self) -> Vec<Dtype> {
        vec![Dtype::from_type::<T>()]
    }

    fn execute(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
        where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(NumPyError::ufunc_error(
                self.name(),
                format!(
                    "Expected 1 input, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let input = unsafe { &*(inputs[0] as *const _ as *const Array<T>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<T>) };

        // Handle where_mask
        let mask = if let Some(m) = where_mask {
            Some(crate::broadcasting::broadcast_to(m, output.shape())?)
        } else {
            None
        };

        // Fast path for C-contiguous arrays
        if input.is_c_contiguous()
            && output.is_c_contiguous()
            && input.shape() == output.shape()
            && mask
                .as_ref()
                .map_or(true, |m| m.is_c_contiguous() && m.shape() == output.shape())
        {
            let in_slice = input.data.as_slice();
            let out_slice = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut T, output.size())
            };

            for i in 0..output.size() {
                if mask
                    .as_ref()
                    .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
                {
                    out_slice[i] = (self.operation)(in_slice[input.offset + i].clone());
                }
            }
            return Ok(());
        }

        for i in 0..input.size() {
            if mask
                .as_ref()
                .map_or(true, |m| *m.get_linear(i).unwrap_or(&false))
            {
                if let Some(a) = input.get(i) {
                    output.set(i, (self.operation)(a.clone()))?;
                }
            }
        }

        Ok(())
    }
}

/// Ufunc registry for looking up functions by name and type
/// This registry stores multiple ufunc implementations per name, keyed by their concrete types
pub struct UfuncRegistry {
    /// Map from ufunc name to a list of implementations for different types
    ufuncs: std::collections::HashMap<String, Vec<Box<dyn Ufunc>>>,
}

impl UfuncRegistry {
    /// Create new registry
    pub fn new() -> Self {
        let mut registry = Self {
            ufuncs: std::collections::HashMap::new(),
        };

        // Register basic ufuncs
        registry.register_basic_ufuncs();
        registry.register_comparison_ufuncs();
        registry.register_math_ufuncs();
        registry.register_bitwise_ufuncs();
        registry
    }

    /// Register a ufunc - stores ALL implementations, not just the last one
    pub fn register(&mut self, ufunc: Box<dyn Ufunc>) {
        let name = ufunc.name().to_string();

        self.ufuncs.entry(name).or_default().push(ufunc);
    }

    /// Get ufunc by name (returns first match, deprecated in favor of get_by_dtypes)
    pub fn get(&self, name: &str) -> Option<&dyn Ufunc> {
        self.ufuncs
            .get(name)
            .and_then(|ufuncs| ufuncs.first())
            .map(|uf| uf.as_ref())
    }

    /// Get ufunc by name and concrete input types - this is the PROPER way to lookup ufuncs
    pub fn get_by_dtypes(&self, name: &str, input_types: &[&'static str]) -> Option<&dyn Ufunc> {
        self.ufuncs.get(name).and_then(|ufuncs| {
            ufuncs
                .iter()
                .find(|uf| uf.matches_concrete_types(input_types))
                .map(|uf| uf.as_ref())
        })
    }

    /// Resolve ufunc by name and input dtypes, allowing for casting
    /// Returns the matched ufunc and the specific dtypes it expects (so caller knows what to cast to)
    pub fn resolve_ufunc(
        &self,
        name: &str,
        input_dtypes: &[Dtype],
        casting: crate::dtype::Casting,
    ) -> Option<(&dyn Ufunc, Vec<Dtype>)> {
        if let Some(candidates) = self.ufuncs.get(name) {
            // First pass: look for exact match
            for ufunc in candidates {
                let target_dtypes = ufunc.input_dtypes();
                if target_dtypes.len() == input_dtypes.len()
                    && target_dtypes
                        .iter()
                        .zip(input_dtypes.iter())
                        .all(|(t, i)| t == i)
                {
                    return Some((ufunc.as_ref(), target_dtypes));
                }
            }

            // Second pass: look for match with casting
            for ufunc in candidates {
                let target_dtypes = ufunc.input_dtypes();

                if target_dtypes.len() != input_dtypes.len() {
                    continue;
                }

                let all_castable = input_dtypes
                    .iter()
                    .zip(target_dtypes.iter())
                    .all(|(in_dt, target_dt)| in_dt.can_cast(target_dt, casting));

                if all_castable {
                    return Some((ufunc.as_ref(), target_dtypes));
                }
            }
        }
        None
    }

    /// List all registered ufunc names
    pub fn list(&self) -> Vec<&str> {
        self.ufuncs.keys().map(|s| s.as_str()).collect()
    }

    /// Get all implementations for a given ufunc name
    pub fn get_all(&self, name: &str) -> Vec<&dyn Ufunc> {
        self.ufuncs.get(name).map_or_else(Vec::new, |ufuncs| {
            ufuncs.iter().map(|uf| uf.as_ref()).collect()
        })
    }

    /// Register basic mathematical ufuncs
    fn register_basic_ufuncs(&mut self) {
        // Addition
        self.register(Box::new(BinaryUfunc::new("add", |a: f64, b: f64| a + b)));
        self.register(Box::new(BinaryUfunc::new("add", |a: f32, b: f32| a + b)));
        self.register(Box::new(BinaryUfunc::new("add", |a: i64, b: i64| a + b)));

        // Subtraction
        self.register(Box::new(BinaryUfunc::new("subtract", |a: f64, b: f64| {
            a - b
        })));
        self.register(Box::new(BinaryUfunc::new("subtract", |a: f32, b: f32| {
            a - b
        })));
        self.register(Box::new(BinaryUfunc::new("subtract", |a: i64, b: i64| {
            a - b
        })));

        // Multiplication
        self.register(Box::new(BinaryUfunc::new("multiply", |a: f64, b: f64| {
            a * b
        })));
        self.register(Box::new(BinaryUfunc::new("multiply", |a: f32, b: f32| {
            a * b
        })));
        self.register(Box::new(BinaryUfunc::new("multiply", |a: i64, b: i64| {
            a * b
        })));

        // Division
        self.register(Box::new(BinaryUfunc::new("divide", |a: f64, b: f64| a / b)));
        self.register(Box::new(BinaryUfunc::new("divide", |a: f32, b: f32| a / b)));
        self.register(Box::new(BinaryUfunc::new("divide", |a: i64, b: i64| a / b)));

        // Unary operations
        self.register(Box::new(UnaryUfunc::new("negative", |a: f64| -a)));
        self.register(Box::new(UnaryUfunc::new("negative", |a: f32| -a)));
        self.register(Box::new(UnaryUfunc::new("negative", |a: i64| -a)));

        self.register(Box::new(UnaryUfunc::new("absolute", |a: f64| a.abs())));
        self.register(Box::new(UnaryUfunc::new("absolute", |a: f32| a.abs())));
        self.register(Box::new(UnaryUfunc::new("absolute", |a: i64| a.abs())));
    }

    /// Register mathematical ufuncs
    fn register_math_ufuncs(&mut self) {
        use crate::math_ufuncs::register_math_ufuncs;
        register_math_ufuncs(self);
    }

    /// Register bitwise ufuncs
    fn register_bitwise_ufuncs(&mut self) {
        use crate::bitwise::register_bitwise_ufuncs;
        register_bitwise_ufuncs(self);
    }

    /// Register comparison ufuncs
    fn register_comparison_ufuncs(&mut self) {
        use crate::comparison_ufuncs::{ComparisonUfunc, ExtremaUfunc, LogicalUnaryUfunc};

        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &f64, b: &f64| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &f32, b: &f32| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &i64, b: &i64| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &i32, b: &i32| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &i16, b: &i16| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &i8, b: &i8| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &u64, b: &u64| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &u32, b: &u32| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &u16, b: &u16| a > b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater",
            |a: &u8, b: &u8| a > b,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &f64, b: &f64| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &f32, b: &f32| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &i64, b: &i64| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &i32, b: &i32| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &i16, b: &i16| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new("less", |a: &i8, b: &i8| {
            a < b
        })));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &u64, b: &u64| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &u32, b: &u32| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less",
            |a: &u16, b: &u16| a < b,
        )));
        self.register(Box::new(ComparisonUfunc::new("less", |a: &u8, b: &u8| {
            a < b
        })));

        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &f64, b: &f64| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &f32, b: &f32| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &i64, b: &i64| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &i32, b: &i32| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &i16, b: &i16| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &i8, b: &i8| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &u64, b: &u64| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &u32, b: &u32| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &u16, b: &u16| a >= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "greater_equal",
            |a: &u8, b: &u8| a >= b,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &f64, b: &f64| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &f32, b: &f32| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &i64, b: &i64| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &i32, b: &i32| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &i16, b: &i16| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &i8, b: &i8| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &u64, b: &u64| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &u32, b: &u32| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &u16, b: &u16| a <= b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "less_equal",
            |a: &u8, b: &u8| a <= b,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &f64, b: &f64| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &f32, b: &f32| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &i64, b: &i64| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &i32, b: &i32| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &i16, b: &i16| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new("equal", |a: &i8, b: &i8| {
            a == b
        })));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &u64, b: &u64| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &u32, b: &u32| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "equal",
            |a: &u16, b: &u16| a == b,
        )));
        self.register(Box::new(ComparisonUfunc::new("equal", |a: &u8, b: &u8| {
            a == b
        })));

        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &f64, b: &f64| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &f32, b: &f32| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &i64, b: &i64| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &i32, b: &i32| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &i16, b: &i16| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &i8, b: &i8| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &u64, b: &u64| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &u32, b: &u32| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &u16, b: &u16| a != b,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "not_equal",
            |a: &u8, b: &u8| a != b,
        )));

        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &f64, b: &f64| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &f32, b: &f32| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &i64, b: &i64| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &i32, b: &i32| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &i16, b: &i16| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new("maximum", |a: &i8, b: &i8| {
            if a >= b {
                *a
            } else {
                *b
            }
        })));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &u64, b: &u64| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &u32, b: &u32| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "maximum",
            |a: &u16, b: &u16| if a >= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new("maximum", |a: &u8, b: &u8| {
            if a >= b {
                *a
            } else {
                *b
            }
        })));

        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &f64, b: &f64| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &f32, b: &f32| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &i64, b: &i64| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &i32, b: &i32| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &i16, b: &i16| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new("minimum", |a: &i8, b: &i8| {
            if a <= b {
                *a
            } else {
                *b
            }
        })));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &u64, b: &u64| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &u32, b: &u32| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new(
            "minimum",
            |a: &u16, b: &u16| if a <= b { *a } else { *b },
        )));
        self.register(Box::new(ExtremaUfunc::new("minimum", |a: &u8, b: &u8| {
            if a <= b {
                *a
            } else {
                *b
            }
        })));

        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &f64, b: &f64| *a != 0.0 && *b != 0.0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &f32, b: &f32| *a != 0.0 && *b != 0.0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &i64, b: &i64| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &i32, b: &i32| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &i16, b: &i16| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &i8, b: &i8| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &u64, b: &u64| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &u32, b: &u32| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &u16, b: &u16| *a != 0 && *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_and",
            |a: &u8, b: &u8| *a != 0 && *b != 0,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &f64, b: &f64| *a != 0.0 || *b != 0.0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &f32, b: &f32| *a != 0.0 || *b != 0.0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &i64, b: &i64| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &i32, b: &i32| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &i16, b: &i16| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &i8, b: &i8| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &u64, b: &u64| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &u32, b: &u32| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &u16, b: &u16| *a != 0 || *b != 0,
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_or",
            |a: &u8, b: &u8| *a != 0 || *b != 0,
        )));

        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &f64, b: &f64| (*a != 0.0) != (*b != 0.0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &f32, b: &f32| (*a != 0.0) != (*b != 0.0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &i64, b: &i64| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &i32, b: &i32| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &i16, b: &i16| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &i8, b: &i8| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &u64, b: &u64| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &u32, b: &u32| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &u16, b: &u16| (*a != 0) != (*b != 0),
        )));
        self.register(Box::new(ComparisonUfunc::new(
            "logical_xor",
            |a: &u8, b: &u8| (*a != 0) != (*b != 0),
        )));

        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &f64| *a == 0.0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &f32| *a == 0.0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &i64| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &i32| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &i16| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new("logical_not", |a: &i8| {
            *a == 0
        })));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &u64| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &u32| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new(
            "logical_not",
            |a: &u16| *a == 0,
        )));
        self.register(Box::new(LogicalUnaryUfunc::new("logical_not", |a: &u8| {
            *a == 0
        })));
    }
}

impl Default for UfuncRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for UfuncRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_impls: usize = self.ufuncs.values().map(|v| v.len()).sum();
        write!(
            f,
            "UfuncRegistry({} ufunc names, {} implementations)",
            self.ufuncs.len(),
            total_impls
        )
    }
}

// Global ufunc registry (doc comment removed to avoid warning)
lazy_static::lazy_static! {
    pub static ref UFUNC_REGISTRY: UfuncRegistry = UfuncRegistry::new();
}

/// Get ufunc by name
pub fn get_ufunc(name: &str) -> Option<&'static dyn Ufunc> {
    UFUNC_REGISTRY.get(name)
}

/// Get ufunc by name and input type
pub fn get_ufunc_typed<T: 'static>(name: &str) -> Option<&'static dyn Ufunc> {
    UFUNC_REGISTRY.get_by_dtypes(name, &[std::any::type_name::<T>()])
}

/// Get binary ufunc by name and input type
pub fn get_ufunc_typed_binary<T: 'static>(name: &str) -> Option<&'static dyn Ufunc> {
    UFUNC_REGISTRY.get_by_dtypes(
        name,
        &[std::any::type_name::<T>(), std::any::type_name::<T>()],
    )
}

/// List all available ufuncs
pub fn list_ufuncs() -> Vec<&'static str> {
    UFUNC_REGISTRY.list()
}

// ============================================================================
// Advanced Ufunc Features
// ============================================================================

/// Metadata for a ufunc
#[derive(Debug, Clone)]
pub struct UfuncMetadata {
    /// Name of the ufunc
    pub name: String,
    /// Number of inputs
    pub nin: usize,
    /// Number of outputs
    pub nout: usize,
    /// Documentation string
    pub doc: Option<String>,
    /// Supported dtype kinds
    pub supported_dtypes: Vec<DtypeKind>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

impl UfuncMetadata {
    /// Create new ufunc metadata
    pub fn new(name: String, nin: usize, nout: usize) -> Self {
        Self {
            name,
            nin,
            nout,
            doc: None,
            supported_dtypes: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Set documentation
    pub fn with_doc(mut self, doc: String) -> Self {
        self.doc = Some(doc);
        self
    }

    /// Add supported dtype
    pub fn with_dtype(mut self, dtype: DtypeKind) -> Self {
        self.supported_dtypes.push(dtype);
        self
    }

    /// Add custom attribute
    pub fn with_attribute(mut self, key: String, value: String) -> Self {
        self.attributes.insert(key, value);
        self
    }
}

/// Performance metrics for ufunc execution
#[derive(Debug, Clone)]
pub struct UfuncPerformanceMetrics {
    /// Number of times the ufunc was called
    pub call_count: u64,
    /// Total time spent in microseconds
    pub total_time_us: u64,
    /// Number of elements processed
    pub total_elements: u64,
    /// Last execution time in microseconds
    pub last_exec_time_us: u64,
}

impl UfuncPerformanceMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self {
            call_count: 0,
            total_time_us: 0,
            total_elements: 0,
            last_exec_time_us: 0,
        }
    }

    /// Get average time per call in microseconds
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

/// Custom ufunc trait with inner loop support
pub trait CustomUfunc: Send + Sync {
    /// Get ufunc name
    fn name(&self) -> &str;

    /// Get number of inputs
    fn nin(&self) -> usize;

    /// Get number of outputs
    fn nout(&self) -> usize;

    /// Execute ufunc on array inputs (high-level interface)
    fn call(&self, inputs: &[&dyn ArrayView], outputs: &mut [&mut dyn ArrayViewMut]) -> Result<()>;

    /// Inner loop implementation (low-level interface for performance)
    ///
    /// This mimics NumPy's inner loop mechanism for maximum performance.
    /// The inner loop operates on raw pointers with stride information.
    ///
    /// # Parameters
    /// - `args`: Raw pointers to input and output data buffers
    /// - `dimensions`: Array sizes for each dimension
    /// - `steps`: Byte strides for each argument
    /// - `data`: User-provided data pointer (can be used for closure state)
    ///
    /// # Safety
    /// Implementations must ensure proper memory safety when working with raw pointers.
    fn inner_loop(
        &self,
        args: &[&[u8]],
        dimensions: &[isize],
        steps: &[isize],
        data: &mut [*mut u8],
    ) -> Result<()>;

    /// Get ufunc metadata
    fn metadata(&self) -> UfuncMetadata;

    /// Check if ufunc supports the given dtypes
    fn supports_dtypes(&self, dtypes: &[&Dtype]) -> bool {
        let metadata = self.metadata();
        dtypes
            .iter()
            .all(|dt| metadata.supported_dtypes.contains(&dt.kind()))
    }
}

/// Generalized ufunc (gufunc) signature
///
/// A gufunc operates on sub-arrays rather than scalars, following NumPy's gufunc specification.
#[derive(Debug, Clone)]
pub struct GufuncSignature {
    /// Signature string (e.g., "(i,j),(j,k)->(i,k)" for matrix multiplication)
    pub signature: String,
    /// Core dimensions for each input (e.g., [[2,3], [3,4]] for 2D x 2D inputs)
    pub input_core_dims: Vec<Vec<usize>>,
    /// Core dimensions for each output
    pub output_core_dims: Vec<Vec<usize>>,
    /// Number of loop dimensions (broadcasted dimensions)
    pub num_loop_dims: usize,
}

impl GufuncSignature {
    /// Parse a gufunc signature string
    ///
    /// Example: "(i,j),(j,k)->(i,k)" represents matrix multiplication
    /// Note: This implementation uses placeholder sizes (1) for symbolic dimension names
    /// Real implementations would track dimension names for validation
    pub fn parse(signature: &str) -> Result<Self> {
        // Parse signature like "(i,j),(j,k)->(i,k)"
        let parts: Vec<&str> = signature.split("->").collect();
        if parts.len() != 2 {
            return Err(NumPyError::invalid_value(
                "Invalid gufunc signature: must contain '->'",
            ));
        }

        let input_part = parts[0];
        let output_part = parts[1];

        // For symbolic dimensions (i, j, k, m, n), we use placeholder size of 1
        // In a full implementation, we would track dimension names to ensure consistency
        let parse_dims = |part: &str| -> Result<Vec<Vec<usize>>> {
            part.split(')')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let inner = s.trim_start_matches('(');
                    if inner.is_empty() {
                        Ok(Vec::new())
                    } else {
                        // Count the number of dimensions, using 1 as placeholder size
                        let count = inner.split(',').count();
                        Ok(vec![1; count]) // Placeholder sizes
                    }
                })
                .collect()
        };

        let input_core_dims = parse_dims(input_part)?;
        let output_core_dims = parse_dims(output_part)?;

        Ok(Self {
            signature: signature.to_string(),
            input_core_dims,
            output_core_dims,
            num_loop_dims: 0, // Will be determined from actual arrays
        })
    }

    /// Calculate the number of loop dimensions from array shapes
    pub fn calculate_loop_dims(&mut self, input_shapes: &[&[usize]]) -> Result<usize> {
        if input_shapes.len() != self.input_core_dims.len() {
            return Err(NumPyError::invalid_value(format!(
                "Expected {} inputs, got {}",
                self.input_core_dims.len(),
                input_shapes.len()
            )));
        }

        // Find minimum number of dimensions excluding core dims
        let min_loop_dims = input_shapes
            .iter()
            .zip(self.input_core_dims.iter())
            .map(|(shape, core)| shape.len().saturating_sub(core.len()))
            .min()
            .unwrap_or(0);

        self.num_loop_dims = min_loop_dims;
        Ok(min_loop_dims)
    }
}

/// Generalized ufunc implementation
pub trait GeneralizedUfunc: Send + Sync {
    /// Get ufunc name
    fn name(&self) -> &str;

    /// Get gufunc signature
    fn signature(&self) -> &GufuncSignature;

    /// Execute the gufunc on arrays
    ///
    /// Unlike regular ufuncs, gufuncs operate on sub-arrays defined by the signature.
    fn call_gufunc(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
    ) -> Result<()>;

    /// Inner loop for gufunc - operates on sub-arrays
    fn inner_loop_gufunc(
        &self,
        args: &[&[u8]],
        core_dims: &[Vec<usize>],
        steps: &[isize],
        data: &mut [*mut u8],
    ) -> Result<()>;

    /// Get metadata
    fn metadata(&self) -> UfuncMetadata;
}

/// Custom ufunc registry for user-defined ufuncs
pub struct CustomUfuncRegistry {
    /// Map of custom ufuncs by name
    custom_ufuncs: HashMap<String, Arc<dyn CustomUfunc>>,
    /// Map of generalized ufuncs by name
    gufuncs: HashMap<String, Arc<dyn GeneralizedUfunc>>,
    /// Performance metrics for each ufunc
    metrics: HashMap<String, Arc<RwLock<UfuncPerformanceMetrics>>>,
    /// Enable/disable performance tracking
    profiling_enabled: Arc<RwLock<bool>>,
}

impl CustomUfuncRegistry {
    /// Create a new custom ufunc registry
    pub fn new() -> Self {
        Self {
            custom_ufuncs: HashMap::new(),
            gufuncs: HashMap::new(),
            metrics: HashMap::new(),
            profiling_enabled: Arc::new(RwLock::new(false)),
        }
    }

    /// Register a custom ufunc
    pub fn register_custom<T>(&mut self, ufunc: T)
    where
        T: CustomUfunc + 'static,
    {
        let name = ufunc.name().to_string();
        let metadata = ufunc.metadata();
        let _supported_dtypes = metadata.supported_dtypes.clone();

        self.custom_ufuncs.insert(name.clone(), Arc::new(ufunc));
        self.metrics.insert(
            name.clone(),
            Arc::new(RwLock::new(UfuncPerformanceMetrics::new())),
        );

        // Store dtype info for validation
        let _metrics = self.metrics.get(&name).unwrap().write().unwrap();
        // Could add more metadata tracking here
    }

    /// Register a generalized ufunc
    pub fn register_gufunc<T>(&mut self, gufunc: T)
    where
        T: GeneralizedUfunc + 'static,
    {
        let name = gufunc.name().to_string();
        self.gufuncs.insert(name.clone(), Arc::new(gufunc));
        self.metrics.insert(
            name.clone(),
            Arc::new(RwLock::new(UfuncPerformanceMetrics::new())),
        );
    }

    /// Call a custom ufunc by name
    pub fn call_custom(
        &self,
        name: &str,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
    ) -> Result<()> {
        let ufunc = self
            .custom_ufuncs
            .get(name)
            .ok_or_else(|| NumPyError::ufunc_error(name, "Custom ufunc not found"))?;

        // Profile execution if enabled
        let start = if *self.profiling_enabled.read().unwrap() {
            Some(Instant::now())
        } else {
            None
        };

        let result = ufunc.call(inputs, outputs);

        // Update metrics
        if let Some(start_time) = start {
            let elapsed = start_time.elapsed().as_micros() as u64;
            let total_elements: u64 = inputs.iter().map(|a| a.size() as u64).sum();

            if let Some(metrics) = self.metrics.get(name) {
                let mut m = metrics.write().unwrap();
                m.call_count += 1;
                m.total_time_us += elapsed;
                m.total_elements += total_elements;
                m.last_exec_time_us = elapsed;
            }
        }

        result
    }

    /// Call a generalized ufunc by name
    pub fn call_gufunc(
        &self,
        name: &str,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
    ) -> Result<()> {
        let gufunc = self
            .gufuncs
            .get(name)
            .ok_or_else(|| NumPyError::ufunc_error(name, "Gufunc not found"))?;

        // Profile execution if enabled
        let start = if *self.profiling_enabled.read().unwrap() {
            Some(Instant::now())
        } else {
            None
        };

        let result = gufunc.call_gufunc(inputs, outputs);

        // Update metrics
        if let Some(start_time) = start {
            let elapsed = start_time.elapsed().as_micros() as u64;
            let total_elements: u64 = inputs.iter().map(|a| a.size() as u64).sum();

            if let Some(metrics) = self.metrics.get(name) {
                let mut m = metrics.write().unwrap();
                m.call_count += 1;
                m.total_time_us += elapsed;
                m.total_elements += total_elements;
                m.last_exec_time_us = elapsed;
            }
        }

        result
    }

    /// Get performance metrics for a ufunc
    pub fn get_metrics(&self, name: &str) -> Option<UfuncPerformanceMetrics> {
        self.metrics.get(name).map(|m| {
            let m = m.read().unwrap();
            UfuncPerformanceMetrics {
                call_count: m.call_count,
                total_time_us: m.total_time_us,
                total_elements: m.total_elements,
                last_exec_time_us: m.last_exec_time_us,
            }
        })
    }

    /// Enable or disable performance profiling
    pub fn set_profiling_enabled(&self, enabled: bool) {
        *self.profiling_enabled.write().unwrap() = enabled;
    }

    /// Check if profiling is enabled
    pub fn is_profiling_enabled(&self) -> bool {
        *self.profiling_enabled.read().unwrap()
    }

    /// List all registered custom ufunc names
    pub fn list_custom_ufuncs(&self) -> Vec<&str> {
        self.custom_ufuncs.keys().map(|s| s.as_str()).collect()
    }

    /// List all registered gufunc names
    pub fn list_gufuncs(&self) -> Vec<&str> {
        self.gufuncs.keys().map(|s| s.as_str()).collect()
    }

    /// Get metadata for a custom ufunc
    pub fn get_custom_metadata(&self, name: &str) -> Option<UfuncMetadata> {
        self.custom_ufuncs.get(name).map(|u| u.metadata())
    }

    /// Get metadata for a gufunc
    pub fn get_gufunc_metadata(&self, name: &str) -> Option<UfuncMetadata> {
        self.gufuncs.get(name).map(|u| u.metadata())
    }
}

impl Default for CustomUfuncRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Global custom ufunc registry
lazy_static::lazy_static! {
    pub static ref CUSTOM_UFUNC_REGISTRY: CustomUfuncRegistry = CustomUfuncRegistry::new();
}

/// Register a custom ufunc globally
pub fn register_custom_ufunc<T>(_ufunc: T)
where
    T: CustomUfunc + 'static,
{
    // Use mutable access - note: in practice you'd need interior mutability
    // This is a simplified example
}

/// Register a generalized ufunc globally
pub fn register_gufunc<T>(_gufunc: T)
where
    T: GeneralizedUfunc + 'static,
{
    // Similar to above
}

/// Get the global custom ufunc registry
pub fn custom_ufunc_registry() -> &'static CustomUfuncRegistry {
    &CUSTOM_UFUNC_REGISTRY
}
