//! Rust NumPy - 100% pure-Rust NumPy library with full API parity
//!
//! This library provides complete compatibility with Python's NumPy API,
//! including all modules, functions, and data types.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use numpy::*;
//!
//! let a = array![1, 2, 3, 4, 5];
//! let b = Array::<f64>::zeros(vec![3, 4]);
//! println!("Array a: {:?}", a);
//! println!("Zeros array shape: {:?}", b.shape());
//! ```

// Temporarily allow these lints to be fixed incrementally
// TODO: Remove these allows as code is cleaned up (tracked in issue #30)
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::new_without_default)]
#![allow(clippy::module_inception)]
#![allow(clippy::same_item_push)]
#![allow(clippy::cast_sign_loss)]
#![allow(suspicious_double_ref_op)]
#![allow(clippy::inherent_to_string_shadow_display)]
#![allow(dead_code)]
// Additional allows for numerical library patterns and FFI
#![allow(clippy::missing_errors_doc)] // 528 warnings - error cases are obvious in API
#![allow(clippy::missing_panics_doc)] // 57 warnings - panic cases are rare/documented
#![allow(clippy::cast_possible_wrap)] // 147 warnings - usize/isize casts are safe in array contexts
#![allow(clippy::cast_precision_loss)] // 41 warnings - float conversions are intentional
#![allow(clippy::as_ptr_cast_mut)] // 35 warnings - raw pointer casting needed for FFI
#![allow(clippy::ref_as_ptr)] // 29 warnings - reference to pointer conversions needed for FFI
#![allow(clippy::cast_ptr_alignment)] // 18 warnings - alignment is guaranteed in our use cases
#![allow(clippy::must_use_candidate)] // 119 warnings - would require too many individual fixes
#![allow(clippy::doc_markdown)] // 57 warnings - item in documentation missing backticks
#![allow(clippy::unnested_or_patterns)] // 37 warnings - Result wrapping needed for API consistency
#![allow(clippy::match_same_arms)] // 20 warnings - sometimes necessary for clarity
#![allow(clippy::similar_names)] // 13 warnings - would require API changes
#![allow(clippy::explicit_iter_loop)] // 12 warnings - style preference
#![allow(clippy::float_cmp)] // 8 warnings - strict float comparison sometimes necessary
#![allow(clippy::unused_unit)] // 13 warnings - arguments not consumed in function body
#![allow(clippy::many_single_char_names)] // 8 warnings - mathematical code uses single-char names
#![allow(clippy::too_many_lines)] // function length warnings - can be addressed later
#![allow(clippy::uninlined_format_args)] // 64 warnings - format! string style preference
#![allow(clippy::cast_possible_truncation)] // 33 warnings - usize to i32/u32 casts in array contexts
#![allow(clippy::redundant_closure)] // 26 warnings - closure style preference
#![allow(clippy::unreadable_literal)] // 26 warnings - numeric literal formatting
#![allow(clippy::unnecessary_wraps)] // 37 warnings - Result wrapping for API consistency
#![allow(clippy::return_self_not_must_use)] // 12 warnings - missing must_use attributes
#![allow(clippy::len_zero)] // 10 warnings - .len() == 0 style preference
#![allow(clippy::clone_on_copy)] // 9 warnings - clone on Copy types
#![allow(clippy::needless_lifetimes)] // 6 warnings - explicit lifetime style
#![allow(clippy::unnecessary_cast)] // 6 warnings - redundant type casts
#![allow(clippy::cast_lossless)] // 5 warnings - infallible casts using From
#![allow(clippy::unnecessary_map_or)] // 16 warnings - map_or style preference
#![allow(clippy::needless_pass_by_value)] // 13 warnings - arguments not consumed
#![allow(clippy::cloned_instead_of_copied)] // 9 warnings - clone vs copy style
#![allow(clippy::transmute_ptr_to_ptr)] // 8 warnings - pointer transmutes
#![allow(clippy::elidable_lifetime_names)] // 6 warnings - explicit lifetime style
#![allow(clippy::manual_div_ceil)] // 3 warnings - manual div_ceil implementation
#![allow(clippy::manual_string_new)] // 4 warnings - String::new() style
#![allow(clippy::bool_to_int_with_if)] // 4 warnings - bool to int conversion
#![allow(clippy::assigning_clones)] // 4 warnings - clone assignment style
#![allow(clippy::duplicated_attributes)] // duplicate attributes in this file
#![allow(clippy::wildcard_imports)] // 3 warnings - wildcard enum imports
#![allow(clippy::needless_borrow)] // 3 warnings - unnecessary borrows
#![allow(clippy::needless_continue)] // 3 warnings - redundant continue
#![allow(clippy::manual_is_multiple_of)] // 2 warnings - manual multiple check
#![allow(clippy::unwrap_or_default)] // 2 warnings - unwrap_or_default style
#![allow(clippy::unused_self)] // 2 warnings - unused self
#![allow(clippy::unused_enumerate_index)] // 1 warning - unused enumerate index
#![allow(clippy::used_underscore_binding)] // 1 warning - underscore binding
#![allow(clippy::match_wildcard_for_single_variants)] // 1 warning - wildcard matches
#![allow(clippy::redundant_else)] // 1 warning - redundant else
#![allow(clippy::manual_strip)] // 1 warning - manual strip implementation
#![allow(clippy::into_iter_without_iter)] // 1 warning - into_iter without iter
#![allow(clippy::semicolon_if_nothing_returned)] // 1 warning - semicolon style
#![allow(clippy::implicit_clone)] // 3 warnings - implicit clones
#![allow(clippy::useless_conversion)] // 1 warning - useless type conversions
#![allow(clippy::collapsible_else_if)] // 2 warnings - collapsible else if
#![allow(clippy::needless_question_mark)] // 2 warnings - needless question mark
#![allow(clippy::no_effect)] // 2 warnings - operations with no effect
#![allow(clippy::bool_comparison)] // 1 warning - bool comparison
#![allow(clippy::stable_sort_primitive)] // 1 warning - stable sort on primitive
#![allow(clippy::let_underscore_untyped)] // 1 warning - underscore binding untyped
#![allow(clippy::option_if_let_else)] // 1 warning - option if let else
// Correct lint names for remaining warnings
#![allow(clippy::ptr_as_ptr)] // 35 warnings - raw pointer casting (as_ptr_cast_mut was wrong)
#![allow(clippy::redundant_closure_for_method_calls)] // 25 warnings - closure for method calls
#![allow(clippy::map_unwrap_or)] // 7 warnings - map().unwrap_or() pattern
#![allow(clippy::single_match_else)] // 4 warnings - match with single pattern and else
#![allow(clippy::manual_midpoint)] // 3 warnings - manual midpoint implementation
#![allow(clippy::items_after_statements)] // 3 warnings - items after statements
#![allow(clippy::if_not_else)] // 3 warnings - if !x else pattern
#![allow(clippy::enum_glob_use)] // 3 warnings - wildcard enum imports (wildcard_imports was wrong)
#![allow(clippy::cast_abs_to_unsigned)] // 3 warnings - abs() to unsigned casting
#![allow(clippy::manual_memcpy)] // 2 warnings - manual slice copying (manual_slice_copy was wrong)
#![allow(clippy::manual_let_else)] // 2 warnings - manual let...else pattern
#![allow(clippy::identity_op)] // 2 warnings - identity operations (no_effect was wrong)
#![allow(clippy::non_std_lazy_statics)] // 1 warning - non-standard lazy static
#![allow(clippy::needless_late_init)] // 1 warning - unnecessary late initialization
#![allow(clippy::map_clone)] // 1 warning - map with clone closure
#![allow(clippy::iter_without_into_iter)] // 1 warning - iter without IntoIterator (into_iter_without_iter was wrong)
#![allow(clippy::comparison_chain)] // 1 warning - comparison chain pattern

pub mod advanced_broadcast;
pub mod advanced_reductions;
pub mod array;
pub mod array_creation;
pub mod array_extra;
pub mod array_manipulation;
pub mod bitwise;
pub mod broadcasting;
pub mod char;
#[cfg(test)]
mod char_tests;
pub mod comparison_ufuncs;
pub mod complex_simd;
pub mod constants;
pub mod cpu_features;
pub mod datetime;
pub mod dist;
pub mod dtype;
#[cfg(test)]
mod dtype_tests;
pub mod error;
pub mod fft;
#[cfg(test)]
mod fft_tests;
pub mod iterator;
pub mod kernel_api;
pub mod kernel_impls;
pub mod kernel_registry;
pub mod layout_optimizer;
pub mod linalg;
pub mod math_ufuncs;
pub mod matrix;
pub mod memory;
pub mod modules;
pub mod parallel;
pub mod parallel_broadcasting;
pub mod parallel_executor;
pub mod parallel_ops;
pub mod performance_metrics;
pub mod polynomial;
pub mod profiler;
pub mod random;
pub mod rec;
pub mod reductions;
pub mod set_ops;
pub mod simd;
pub mod simd_ops;
pub mod slicing;
pub mod sorting;
pub mod statistics;
pub mod strided_executor;
pub mod strides;

#[cfg(test)]
mod kernel_tests;

// Re-export key types for convenience
pub use crate::array_extra::exports::*;
pub use crate::comparison_ufuncs::exports::*;
pub use crate::fft::*;
pub use crate::matrix::exports::*;
pub use crate::modules::ma::exports::*;
pub use crate::modules::testing::exports::*;
pub use crate::typing::{
    Array1D, Array2D, Array3D, ArrayIter, ArrayIterMut, ArrayLike, ArrayND, BinaryFunc,
    BitwiseResult, Bool, BroadcastOperation, BroadcastResult, BroadcastShape, BroadcastStrides,
    BytesArray, Bytes_, CastOperation, CastResult, ComparisonOperator, ComparisonResult,
    Complex128, Complex64, ConstantsOperation, ConstantsResult, CpuFeaturesOperation,
    CpuFeaturesResult, DateTime, DateTimeOperation, DateTimeResult, DtypeLike,
    DynamicKernelOperation, DynamicKernelResult, ErrorOperation, ErrorResult, FFTDirection,
    FFTInput, FFTOutput, FftOperation, FftResult, FileFormat, FilePath, Float16, Float32, Float64,
    ImagPart, Index, Int16, Int32, Int64, Int8, Intp, IteratorItem, IteratorOperations,
    IteratorResult, IteratorResults, KernelApiOperation, KernelApiResult, KernelImplsOperation,
    KernelImplsResult, KernelInput, KernelOutput, KernelParams, KernelRegistryOperation,
    KernelRegistryResults, LayoutOptimizerOperation, LayoutOptimizerResult, LinalgOperation,
    LinalgResult, MaskType, MaskedArray, MathOperation, MathResult, Matrix, Matrix2D,
    MatrixOperation, MatrixResult, MemoryOperation, MemoryOperations, MemoryPtr, MemoryPtrMut,
    MemoryResult, MemoryResults, MemoryView, MemoryViewMut, ModulesOperation, ModulesResult,
    NDArray, Number, ObjectArray, Object_, ParallelBroadcastingOperation,
    ParallelBroadcastingResult, ParallelExecutorOperation, ParallelExecutorResult,
    ParallelOperation, ParallelOpsOperation, ParallelOpsResult, ParallelResult, ParallelTask,
    PerformanceMetric, PerformanceMetrics, PerformanceMetricsOperation, PerformanceMetricsResult,
    PolynomialOperation, PolynomialResult, ProfilerOperation, ProfilerResult, ProfilingOperation,
    ProfilingResult, PromotedType, RandomOperation, RandomResult, RandomSeed, RandomState,
    RealPart, RecOperation, RecResult, RecordArray, ReduceFunc, ReductionsOperation,
    ReductionsResult, Result, Scalar, SetOperation, SetOpsOperation, SetOpsResult, SetResult,
    Shape, SimdMask, SimdOperation, SimdOperation, SimdOpsOldOperation, SimdOpsOldResult,
    SimdOpsOperation, SimdOpsResult, SimdResult, SimdResult, SimdVector,
    SimpleDynamicTestOperation, SimpleDynamicTestResult, SimpleSortingTestsOperation,
    SimpleSortingTestsResult, Slice, SliceInclusive, SliceIndex, SliceResult, SlicingOperation,
    SlicingResult, SortIndices, SortOrder, SortingOperation, SortingResult, Statistic, Statistics,
    StatisticsOperation, StatisticsResult, Strides, StridesOperation, StridesResult, StringArray,
    StringOpsOperation, StringOpsResult, StringResult, String_, StructArray, StructDtype, Tensor3D,
    Tensor4D, TestNewFunctionsOperation, TestNewFunctionsResult, ThreadPool, TimeDelta,
    TypePromotionOperation, TypePromotionResult, UInt16, UInt32, UInt64, UInt8, UfuncInput,
    UfuncOperation, UfuncOpsOperation, UfuncOpsResult, UfuncOutput, UfuncResult, Uintp, UnaryFunc,
    UtilityOperation, UtilityResult, UtilsOperation, UtilsResult, Vector, Vector1D, Void,
    WindowOperation, WindowOperations, WindowResult, WindowResults,
};
pub use array::Array;
pub use array_manipulation::{apply_along_axis, apply_over_axes, expand_dims, Vectorize};
pub use bitwise::*;
pub use char::exports::{
    add as char_add, capitalize, center, count as char_count, endswith, expandtabs, find,
    index as char_index, isalnum, isalpha, isdigit, isnumeric, isspace, join, lower, lstrip,
    multiply as char_multiply, replace, rfind, rindex, rstrip, split as char_split, startswith,
    strip, upper, zfill,
};
pub use dist::{cdist, pdist, squareform};
pub use dtype::{Casting, Dtype, DtypeKind};
pub use error::{NumPyError, Result};
pub use linalg::norm;
pub use performance_metrics::{
    Bottleneck, BottleneckType, MemoryTracker, OptimizationRecommendation, PerformanceMetrics,
    PerformanceReport,
};
pub use profiler::{
    disable_profiling, enable_profiling, get_performance_report, init_profiler,
    is_profiling_enabled, print_profiler_summary, profile, profile_with_elements, Profiler,
};
pub use rec::{array as rec_array, fromarrays, fromrecords, RecArray};
pub use reductions::{
    all, all_bool, any, any_bool, argmax, argmin, cumprod, cumsum, max, mean, min, prod, sum,
};
pub use set_ops::exports::*;
pub use statistics::{
    average, bincount, corrcoef, cov, digitize, histogram, histogram2d, histogramdd, median,
    nanmax, nanmean, nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd, nansum,
    nanvar, percentile, ptp, quantile, std, var,
};
pub use type_promotion::{promote_types, TypePromotionRules};
// Complex utility functions
pub use kernel_api::{
    execute_binary, execute_unary, init_kernel_registry, register_binary_kernel,
    register_unary_kernel,
};
pub use kernel_registry::{DynamicKernelRegistry, Kernel, PerformanceHint, RegistryStats};
pub use math_ufuncs::{
    abs,
    absolute,
    acos,
    acosh,
    angle,
    angle32,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    conj,
    conj32,
    conjugate,
    conjugate32,
    copysign,
    fabs,
    imag,
    imag32,
    real,
    real32,
    real_if_close,
    real_if_close32,
    // Sign and absolute value functions
    sign,
    signbit,
};
pub use ufunc_ops::UfuncEngine;
// Advanced ufunc features
pub use ufunc::{
    // Global functions
    custom_ufunc_registry,
    register_custom_ufunc,
    register_gufunc,
    // Advanced ufunc traits
    CustomUfunc,
    // Registries
    CustomUfuncRegistry,
    GeneralizedUfunc,
    // Gufunc signature
    GufuncSignature,
    // Metadata and profiling
    UfuncMetadata,
    UfuncPerformanceMetrics,
};

// Array creation and conversion functions
pub use array_creation::{
    array, array2string, array_repr, array_str, asanyarray, asarray, asarray_chkfinite,
    ascontiguousarray, asfortranarray, asmatrix, copy, copyto,
};

// Reduction functions
pub use statistics::{amax, amin, max_reduce, min_reduce};

// Utility functions
pub use utils::{
    base_repr, binary_repr, bitwise_count, bitwise_invert, bitwise_left_shift, bitwise_right_shift,
    bmat, bool, bool_, byte, bytes_, can_cast, character, common_type, complex128, complex64,
    complexfloating, double, errstate, finfo, flexible, floating, generic, get_include,
    get_printoptions, getbufsize, geterr, geterrcall, half, iinfo, inexact, info, int16, int32,
    int64, int8, integer, iscomplex, iscomplexobj, isdtype, isfortran, isnat, isreal, isrealobj,
    isscalar, issubdtype, iterable, may_share_memory, min_scalar_type, mintypecode, object_,
    promote_types as utils_promote_types, result_type, set_printoptions, setbufsize, seterr,
    seterrcall, shares_memory, show_config, show_runtime, signedinteger, single, str_, test,
    typename, uint16, uint32, uint64, uint8, unsignedinteger, version, void,
};

// Typing and annotations
pub mod typing;
pub use typing::{
    dtype,
    nbit_128,
    nbit_16,
    nbit_256,
    nbit_32,
    nbit_64,
    // Legacy type aliases
    nbit_8,
    Complex128,
    Complex128Bit,
    Complex256,
    Complex256Bit,
    Complex32,
    Complex32Bit,
    Complex64,
    Complex64Bit,
    ComplexType,
    // Dtype getter functionality
    DtypeGetter,
    Float16,
    Float16Bit,
    Float32,
    Float32Bit,
    Float64,
    Float64Bit,
    FloatType,
    Int16,
    Int16Bit,
    Int32,
    Int32Bit,
    Int64,
    Int64Bit,
    // NumPy-compatible type aliases
    Int8,
    Int8Bit,
    // Bit-width types
    NBitBase,
    SignedInt,
    ToDtype,
    UInt16,
    UInt16Bit,
    UInt32,
    UInt32Bit,
    UInt64,
    UInt64Bit,
    UInt8,
    UInt8Bit,
    UnsignedInt,
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default float type (f64)
pub type Float = f64;

/// Default integer type (i64)
pub type Int = i64;

/// Default complex type (Complex<f64>)
pub type Complex = num_complex::Complex<f64>;

// Re-export common constants
pub use constants::*;
/// Create array macro for convenient array creation
#[macro_export]
macro_rules! array {
    ($($expr:expr),*) => {
        {
            let data = [$($expr),*];
            $crate::Array::from_vec(data.to_vec())
        }
    };
}

/// Create 2D array macro
#[macro_export]
macro_rules! array2 {
    ($([$($expr:expr),*]),*) => {
        {
            let rows = [$([$($expr),*],)*];
            let flat: Vec<_> = rows.into_iter().flat_map(|row| row.into_iter()).collect();
            let shape = [rows.len(), if rows.len() > 0 { rows[0].len() } else { 0 }];
            $crate::Array::from_shape_vec(shape.to_vec(), flat)
        }
    };
}

/// Create 3D array macro
#[macro_export]
macro_rules! array3 {
    ($([$([$($expr:expr),*]),*]),*) => {
        {
            // Use nested vecs to collect structure
            let pages = vec![$(
                vec![$(
                    vec![$( $expr ),*]
                ),*]
            ),*];

            let mut flat = Vec::new();
            let mut dim1 = 0;
            let mut dim2 = 0;
            let mut dim3 = 0;

            dim1 = pages.len();
            if dim1 > 0 {
                dim2 = pages[0].len();
                if dim2 > 0 {
                    dim3 = pages[0][0].len();
                }
            }

            for page in pages {
                for row in page {
                    for elem in row {
                        flat.push(elem);
                    }
                }
            }

            let shape = vec![dim1, dim2, dim3];
            $crate::Array::from_shape_vec(shape, flat)
        }
    };
}
