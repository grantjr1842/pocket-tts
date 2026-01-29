//! Centralized Export Registry for NumPy API
//!
//! This module provides a centralized location for all public API exports,
//! replacing scattered `pub use` statements throughout the codebase.
//! This makes the API surface visible at a glance and prevents duplicate exports.

#![allow(unused_imports)]

// ============================================================================
// NumPy Root-Level API - Array Manipulation
// ============================================================================
pub mod array_api {
    pub use crate::array_manipulation::{
        append, apply_along_axis, apply_over_axes, atleast_1d, atleast_2d, atleast_3d,
        broadcast_arrays, broadcast_to, concatenate, delete, expand_dims, flip, hstack,
        insert, moveaxis, pad, ravel, repeat, reshape, roll, rollaxis, rot90, split,
        squeeze, stack, swapaxes, tile, trim_zeros, vstack,
    };
    pub use crate::array_extra::exports::{
        SplitArg, array_split, block, choose, clip, column_stack, diag, diagonal, diff,
        dsplit, dstack, ediff1d, gradient, hsplit, interp, put, put_along_axis, putmask,
        row_stack, tril, triu, vander, vsplit,
    };
}

// ============================================================================
// NumPy Root-Level API - Math Functions (Ufuncs)
// ============================================================================
pub mod math_api {
    pub use crate::math_ufuncs::{
        abs, absolute, acos, acosh, add, angle, asin, asinh, atan, atan2, atanh,
        cbrt, ceil, conj, conjugate, copysign, cos, cosh, deg2rad, degrees, divide,
        equal, exp, exp2, expm1, fabs, floor, floor_divide, fmax, fmin, fmod,
        greater, greater_equal, heaviside, hypot, i0, imag, imag32, isfinite, isinf,
        isnan, isneginf, isposinf, less, less_equal, log, log10, log1p, log2, logaddexp,
        logaddexp2, maximum, minimum, modf, multiply, negative, nextafter, not_equal,
        positive, power, rad2deg, radians, real, real_if_close, reciprocal, remainder,
        rint, round, sign, signbit, sin, sinh, sqrt, square, subtract, tan, tanh, trunc,
    };
}

// ============================================================================
// NumPy Root-Level API - Statistics
// ============================================================================
pub mod stats_api {
    pub use crate::statistics::{
        amax, amin, average, bincount, corrcoef, cov, digitize, histogram, histogram2d,
        histogramdd, max_reduce, median, min_reduce, nanmax, nanmean, nanmedian, nanmin,
        nanpercentile, nanprod, nanquantile, nanstd, nansum, nanvar, percentile, ptp,
        quantile, std, var,
    };
}

// ============================================================================
// NumPy Root-Level API - Linear Algebra
// ============================================================================
pub mod linalg_api {
    pub use crate::linalg::{
        cholesky, cond, cross, det, diagonal as linalg_diagonal, dot, eig, eigh, eigvals,
        eigvalsh, inv, lstsq, matmul, matrix_power, matrix_rank, multi_dot, norm, outer,
        pinv, qr, slogdet, solve, svd, tensorinv, tensorsolve, trace, transpose, vecdot,
        vector_norm,
    };
}

// ============================================================================
// NumPy Root-Level API - Random
// ============================================================================
pub mod random_api {
    pub use crate::random::{
        beta, binomial, chisquare, default_rng, default_rng_with_seed, dirichlet,
        exponential, gamma, geometric, gumbel, legacy_rng, lognormal, multinomial,
        normal, randint, random, standard_normal, uniform,
    };
    pub use crate::random::bit_generator::{BitGenerator, MT19937, PCG64};
    pub use crate::random::generator::Generator;
}

// ============================================================================
// NumPy Root-Level API - FFT
// ============================================================================
pub mod fft_api {
    pub use crate::fft::{
        FFTNorm, fft, fft2, fftfreq, fftn, fftshift, hfft, ifft, ifft2, ifftn, ifftshift,
        ihfft, irfft, irfft2, irfftn, rfft, rfft2, rfftfreq, rfftn,
    };
}

// ============================================================================
// NumPy Root-Level API - Comparison and Logical
// ============================================================================
pub mod comparison_api {
    pub use crate::comparison_ufuncs::exports::{
        allclose, array_equal, array_equiv, isclose, ComparisonOps, ComparisonUfunc,
        ExtremaUfunc, LogicalOps, LogicalUnaryOps, LogicalUnaryUfunc,
    };
}

// ============================================================================
// NumPy Root-Level API - Set Operations
// ============================================================================
pub mod setops_api {
    pub use crate::set_ops::exports::{
        in1d, intersect1d, isin, setdiff1d, setxor1d, union1d, unique, SetElement,
        SetOps, UniqueResult,
    };
}

// ============================================================================
// NumPy Root-Level API - Bitwise Operations
// ============================================================================
pub mod bitwise_api {
    pub use crate::bitwise::{
        bitwise_and, bitwise_not, bitwise_or, bitwise_xor, invert, left_shift,
        logical_and, logical_not, logical_or, logical_xor, right_shift,
        BitwiseBinaryUfunc, BitwiseOps, BitwiseShiftUfunc, BitwiseUnaryUfunc,
        EnhancedLogicalUfunc,
    };
}

// ============================================================================
// NumPy Root-Level API - Constants
// ============================================================================
pub mod constants_api {
    pub use crate::constants::{
        ClipMode, ComparisonKind, Order, ReductionMode, RoundingMode, SearchMode, SortKind,
        DEFAULT_ALIGNMENT, EPSILON, EPSILON_F32, E, INF, MAX, MAX_DIMS, MAX_ELEMENTS,
        MIN, MIN_POSITIVE, NAN, NEG_INF, NEWAXIS, PI, TAU,
    };
}

// ============================================================================
// NumPy Root-Level API - Testing
// ============================================================================
pub mod testing_api {
    pub use crate::testing::{
        assert_allclose, assert_almost_equal, assert_approx_equal, assert_array_almost_equal,
        assert_array_almost_equal_nulp, assert_array_almost_nulp, assert_array_compare,
        assert_array_equal, assert_array_less, assert_array_max_ulp, assert_array_shape_equal,
        assert_equal, assert_no_gc_cycles, assert_no_warnings, assert_raises,
        assert_raises_regex, assert_string_equal, assert_warns,
    };
}

// ============================================================================
// NumPy Root-Level API - Masked Arrays
// ============================================================================
pub mod ma_api {
    pub use crate::ma::{
        allequal, all, any, append, argmax, argmin, argsort, array, average, compress,
        concatenate, copy, count, cumprod, cumsum, empty, filled, full, full_like, getdata,
        getmask, getmaskarray, harden_mask, isclose, is_masked_array, isnan, masked_array,
        masked_equal, masked_greater, masked_greater_equal, masked_inside, masked_less,
        masked_less_equal, masked_not_equal, masked_object, masked_outside, masked_values,
        masked_where, median, ones, place, ptp, put, reshape, resize, set_fill_value,
        shrink_mask, swapaxes, take, transpose, zeros, MaskedArray, NOMASK,
    };
}

// ============================================================================
// Flattened Re-exports for Direct Use
// ============================================================================

// Re-export all APIs at the exports level for convenience
pub use array_api::*;
pub use math_api::*;
pub use stats_api::*;
pub use linalg_api::*;
pub use random_api::*;
pub use fft_api::*;
pub use comparison_api::*;
pub use setops_api::*;
pub use bitwise_api::*;
pub use constants_api::*;
pub use testing_api::*;
pub use ma_api::*;
