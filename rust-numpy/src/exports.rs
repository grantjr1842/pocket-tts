// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Centralized export registry for rust-numpy
//!
//! This module re-exports all public API items from the crate root.
//! All public exports should be listed here to serve as the source of truth.

// ============================================================================
// Array Types
// ============================================================================

pub use crate::array::Array;
pub use crate::dtype::{
    Casting, Dtype, DtypeKind, float32, float64, int16, int32, int64, int8, intp, uint16, uint32,
    uint64, uint8, uintp,
};

// ============================================================================
// Array Creation
// ============================================================================

pub use crate::array_creation::{
    arange, array, array2string, array_repr, array_str, asanyarray, asarray, asarray_chkfinite,
    ascontiguousarray, asfortranarray, asmatrix, copy, copyto, empty, empty_like, eye, full,
    full_like, identity, linspace, logspace, meshgrid, ones, ones_like, zeros, zeros_like,
};

// ============================================================================
// Array Manipulation
// ============================================================================

pub use crate::array_manipulation::{
    append, apply_along_axis, apply_over_axes, atleast_1d, atleast_2d, atleast_3d, delete,
    expand_dims, eye, flatten, flip, insert, moveaxis, pad, ravel, repeat, reshape, roll, rollaxis,
    rot90, squeeze, swapaxes, tile, Vectorize,
};

pub use crate::array_extra::{
    block, choose, column_stack, compress, concatenate, diag, diagonal, diff, dsplit, dstack,
    ediff1d, gradient, hsplit, hstack, place, put, put_along_axis, putmask, round, row_stack,
    split, stack, trim_zeros, tril, triu, vander, vsplit, vstack,
};

// ============================================================================
// Array Methods
// ============================================================================

pub use crate::array_methods::{
    all, any, argmax, argmin, byteswap, clip, conj, conjugate, cumprod, cumsum, diagonal, dot,
    fill, flatten, get, item, max, mean, min, nonzero, prod, ptp, ravel, repeat, reshape, round,
    set, sort, std, sum, swapaxes, take, trace, transpose, var,
};

// ============================================================================
// Bitwise Operations
// ============================================================================

pub use crate::bitwise::{
    bitwise_and, bitwise_not, bitwise_or, bitwise_xor, invert, left_shift, logical_and, logical_not,
    logical_or, logical_xor, right_shift, BitwiseBinaryUfunc, BitwiseOps, BitwiseShiftUfunc,
    BitwiseUnaryUfunc, EnhancedLogicalUfunc,
};

// ============================================================================
// Char Operations
// ============================================================================

pub use crate::char::{
    add as char_add, capitalize, center, count as char_count, endswith, expandtabs, find,
    index as char_index, isalnum, isalpha, isdigit, isnumeric, isspace, join, lower, lstrip,
    multiply as char_multiply, replace, rfind, rindex, rstrip, split as char_split, startswith,
    strip, upper, zfill,
};

// ============================================================================
// Comparison Operations
// ============================================================================

pub use crate::comparison_ufuncs::{
    equal, greater, greater_equal, less, less_equal, not_equal,
};

// ============================================================================
// Constants
// ============================================================================

pub use crate::constants::{
    array::{DEFAULT_ALIGNMENT, MAX_DIMS, MAX_ELEMENTS},
    dtype::{
        INT16_MAX, INT16_MIN, INT32_MAX, INT32_MIN, INT64_MAX, INT64_MIN, INT8_MAX, INT8_MIN,
        UINT16_MAX, UINT32_MAX, UINT64_MAX, UINT8_MAX,
    },
    float::{EPSILON, EPSILON_F32, MAX, MIN, MIN_POSITIVE},
    math::{E, INF, NAN, NEG_INF, PI, TAU},
    ComparisonKind, NEWAXIS, Order, ReductionMode, RoundingMode, SearchMode, ClipMode, SortKind,
};

// ============================================================================
// Data Types
// ============================================================================

pub use crate::dtype::{ByteOrder, Dtype};

// ============================================================================
// FFT
// ============================================================================

pub use crate::fft::{
    fft, fft2, fftfreq, fftn, fftshift, hfft, ifft, ifft2, ifftn, ifftshift, ihfft, irfft, irfft2,
    irfftn, rfft, rfft2, rfftn, rfftfreq, FFTNorm,
};

// ============================================================================
// Linear Algebra
// ============================================================================

pub use crate::linalg::{
    cond, cross, det, dot, eig, inner, kron, matrix_power, matrix_rank, norm, norm_axis, outer,
    qr, slogdet, svd, trace, LinAlgError,
};

// ============================================================================
// Logical Operations
// ============================================================================

// ============================================================================
// Masked Arrays
// ============================================================================

pub use crate::modules::ma;

// ============================================================================
// Mathematical Functions
// ============================================================================

pub use crate::math_ufuncs::{
    abs, absolute, acos, acosh, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, angle,
    angle32, asin, asinh, atan, atan2, atanh, conj, conj32, conjugate, conjugate32, convolve,
    copysign, correlate, cos, cosh, degrees, exp, exp2, expm1, fabs, fix, floor, heaviside, hypot,
    i0, imag, imag32, isfinite, isinf, isnan, isneginf, isposinf, log, log10, log1p, log2,
    logaddexp, logaddexp2, radians, real, real32, real_if_close, real_if_close32, rint, round_,
    sign, signbit, sin, sinc, sinh, sqrt, tan, tanh, trunc, unwrap,
};

// ============================================================================
// Random Number Generation
// ============================================================================

pub use crate::random::{
    default_rng, default_rng_with_seed, beta, binomial, chisquare, dirichlet, exponential, gamma,
    gumbel, geometric, legacy_rng, lognormal, multinomial, normal, normal_float, randint, random,
    shuffle, standard_normal, uniform,
};

pub use crate::random::bit_generator::{BitGenerator, MT19937, PCG64};
pub use crate::random::generator::Generator;

// ============================================================================
// Set Operations
// ============================================================================

pub use crate::set_ops::{in1d, intersect1d, isin, setdiff1d, setxor1d, union1d, unique};

// ============================================================================
// Sorting and Searching
// ============================================================================

pub use crate::sorting::{argsort, sort};

// ============================================================================
// Statistics
// ============================================================================

pub use crate::statistics::{
    amax, amin, average, bincount, corrcoef, cov, digitize, histogram, histogram2d, histogramdd,
    max_reduce, median, min_reduce, nanmax, nanmean, nanmedian, nanmin, nanpercentile, nanprod,
    nanquantile, nanstd, nansum, nanvar, percentile, ptp, quantile, std, var,
};

// ============================================================================
// String Operations
// ============================================================================

// ============================================================================
// Type Promotion
// ============================================================================

pub use crate::type_promotion::{promote_types, TypePromotionRules};

// ============================================================================
// Type Aliases and Protocols
// ============================================================================

pub use crate::typing::{
    dtype,
    prelude::*,
    ArrayLike,
    Boolean,
    Complex128Bit,
    Complex256Bit,
    Complex32Bit,
    Complex64Bit,
    DtypeGetter,
    DtypeLike,
    Float16Bit,
    Float32Bit,
    Float64Bit,
    Int16Bit,
    Int32Bit,
    Int64Bit,
    Int8Bit,
    NDArray,
    ShapeLike,
    SupportsIndex,
    ToDtype,
    UInt16Bit,
    UInt32Bit,
    UInt64Bit,
    UInt8Bit,
    NBitBase, SignedInt, UnsignedInt, nbit_128, nbit_16, nbit_256, nbit_32, nbit_64, nbit_8,
};

// ============================================================================
// Utility Functions
// ============================================================================

pub use crate::utils::{
    base_repr, binary_repr, bitwise_count, bitwise_invert, bitwise_left_shift, bitwise_right_shift,
    bmat, bool, byte, bytes_, can_cast, character, common_type, double, errstate, finfo, flexible,
    get_include, get_printoptions, getbufsize, geterr, geterrcall, half, iinfo, inexact, info,
    iscomplex, iscomplexobj, isdtype, isfortran, isnat, isreal, isrealobj, isscalar, issubdtype,
    iterable, may_share_memory, min_scalar_type, mintypecode, promote_types as utils_promote_types,
    result_type, set_printoptions, setbufsize, seterr, seterrcall, shares_memory, show_config,
    show_runtime, single, test, typename, version,
};

// ============================================================================
// Ufunc Engine
// ============================================================================

pub use crate::ufunc_ops::UfuncEngine;

// ============================================================================
// Advanced Ufunc Features
// ============================================================================

pub use crate::ufunc::{
    custom_ufunc_registry, register_custom_ufunc, register_gufunc, CustomUfunc, CustomUfuncRegistry,
    GeneralizedUfunc, GufuncSignature, UfuncMetadata, UfuncPerformanceMetrics,
};

// ============================================================================
// Kernel Operations
// ============================================================================

pub use crate::kernel_api::{
    execute_binary, execute_unary, init_kernel_registry, register_binary_kernel,
    register_unary_kernel,
};

pub use crate::kernel_registry::Kernel;
pub use crate::kernels::UfuncPerformanceHint as PerformanceHint;

pub use crate::dynamic_kernel_registry::{DynamicKernelRegistry, RegistryStats};

// ============================================================================
// Performance and Profiling
// ============================================================================

pub use crate::performance_metrics::{
    Bottleneck, BottleneckType, MemoryTracker, OptimizationRecommendation, PerformanceMetrics,
    PerformanceReport,
};

pub use crate::profiler::{
    disable_profiling, enable_profiling, get_performance_report, init_profiler,
    is_profiling_enabled, print_profiler_summary, profile, profile_with_elements, Profiler,
};

// ============================================================================
// Matrix Operations
// ============================================================================

pub use crate::matrix::{
    bmat, diag, diagflat, eye, identity, matmul, matrix, tril, triu,
};

// ============================================================================
// Distance Computations
// ============================================================================

pub use crate::dist::{cdist, pdist, squareform};

// ============================================================================
// Polynomial
// ============================================================================

pub use crate::polynomial::{poly, polyadd, polyder, polydiv, polyint, polymul, polysub, polyval};

// ============================================================================
// Window Functions
// ============================================================================

pub use crate::window::{barrett, blackman, blackmanharris, hamming, hann, kaiser};

// ============================================================================
// Testing
// ============================================================================

pub use crate::modules::testing;

// ============================================================================
// Error Types
// ============================================================================

pub use crate::error::{NumPyError, Result};

// ============================================================================
// Default Types
// ============================================================================

/// Default float type (f64)
pub type Float = f64;

/// Default integer type (i64)
pub type Int = i64;

/// Default complex type (Complex<f64>)
pub type Complex = num_complex::Complex<f64>;
