# NumPy vs Rust-NumPy Parity Gap Analysis

**Scope:** End-to-end parity audit of NumPy API vs rust-numpy
**Status:** In progress
**Last Updated:** 2026-01-24

## Dataset Catalog (Shared Inputs)

| Dataset ID | Description | Data Summary | Intended Coverage |
| --- | --- | --- | --- |
| DS-ARRAY-1 | Small integer vector | `[0, 1, -2, 3, 4, -5, 6]` | basic array ops, reductions |
| DS-ARRAY-2 | 2D int matrix | `[[1, 2, 3], [4, 5, 6]]` | reshape/transpose/stack |
| DS-ARRAY-3 | Float vector w/ NaN/Inf | `[0.0, -1.5, 2.25, NaN, Inf, -Inf]` | nan-aware stats, comparisons |
| DS-ARRAY-4 | Boolean mask | `[true, false, true, false, true]` | masking, selection |
| DS-ARRAY-5 | String vector | `"alpha", "Beta", "gamma", ""` | string/char ops |
| DS-ARRAY-6 | Datetime sample | `2024-01-01, 2024-06-30, 2025-01-01` | datetime/timedelta |
| DS-ARRAY-7 | Complex vector | `[1+2i, -3+0.5i, 0-1i]` | complex math, fft |
| DS-ARRAY-8 | Sorted vector | `[1, 1, 2, 3, 5, 8, 13]` | sorting/search/unique |
| DS-ARRAY-9 | Set ops pair | A:`[1,2,3,4]` B:`[3,4,5,6]` | union/intersect/diff |
| DS-ARRAY-10 | Linear system | A:`[[3,1],[1,2]]` b:`[9,8]` | linalg solve/inv |
| DS-ARRAY-11 | FFT signal | 64-sample sine wave, 1 Hz | fft/rfft/freq |
| DS-ARRAY-12 | Polynomial coeffs | `p(x)=1 -2x +3x^2` | poly eval/roots |
| DS-RAND-1 | Seeded RNG | seed=1234, size=10 | random distributions |
| DS-IO-1 | Simple CSV | 3x3 numeric matrix | load/save/loadtxt |

## Parity Table

| NumPy Module | NumPy Function | Rust Function | Dataset ID | Parity | Gap Notes | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| array_creation | arange | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: arange |
| array_creation | zeros | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: zeros_vec |
| array_creation | ones | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: ones_vec |
| array_creation | eye | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: eye |
| array_creation | linspace | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: linspace |
| math_ufuncs | log | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: log_vec |
| math_ufuncs | power | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: power_vec |
| statistics | min | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: compute_min |
| statistics | std | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: compute_std |
| statistics | var | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: compute_var |
| array_manipulation | transpose_2d | - | DS-ARRAY-2 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: transpose_2d |
| linalg | dot | - | DS-ARRAY-1 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: dot_vec |
| linalg | matmul | - | DS-ARRAY-2 | ❌ Missing FFI | Function exists in Rust but not exported for FFI | /home/grant/github/pocket-tts/.worktrees/issue-348-gap-analysis-validation/target/release/libnumpy.so: undefined symbol: matmul_2d |
