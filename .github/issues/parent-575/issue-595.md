# [Sub-Issue] Implement Missing FFT Functions (N-D, Real, Hermitian) (#575)

**Issue #595**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/595  
**Created:** 2026-01-29T03:09:09Z  
**Updated:** 2026-01-29T03:24:24Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:fft-io, ralph/numpy:impl  

---

## Parent Issue
Part of #575 - fft and io modules missing many NumPy functions

## Scope
Implement Missing FFT Functions (fft2, fftn, rfft, hfft families)

## Background
rust-numpy currently only has 1-D FFT. NumPy provides N-D and real-input FFTs.

## Tasks

### 1. Implement 2-D FFT Functions
```rust
// src/fft.rs
pub fn fft2<T>(
    input: &Array<T>,
    s: Option<(usize, usize)>,  // shape
    axes: (isize, isize),
    norm: Option<FFTNorm>,
) -> Result<Array<Complex64>, NumPyError>;

pub fn ifft2<T>(
    input: &Array<T>,
    s: Option<(usize, usize)>,
    axes: (isize, isize),
    norm: Option<FFTNorm>,
) -> Result<Array<Complex64>, NumPyError>;
```

### 2. Implement N-D FFT Functions
```rust
pub fn fftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<FFTNorm>,
) -> Result<Array<Complex64>, NumPyError>;

pub fn ifftn<T>(
    input: &Array<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<FFTNorm>,
) -> Result<Array<Complex64>, NumPyError>;
```

### 3. Implement Real FFT Functions
```rust
pub fn rfft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<FFTNorm>,
) -> Result<Array<Complex64>, NumPyError>;
// Note: Returns only positive frequency components

pub fn irfft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<FFTNorm>,
) -> Result<Array<f64>, NumPyError>;
// Note: Returns real array
```

### 4. Implement Real N-D FFT
```rust
pub fn rfft2<T>(...) -> Result<Array<Complex64>, NumPyError>;
pub fn irfft2<T>(...) -> Result<Array<f64>, NumPyError>;
pub fn rfftn<T>(...) -> Result<Array<Complex64>, NumPyError>;
pub fn irfftn<T>(...) -> Result<Array<f64>, NumPyError>;
```

### 5. Implement Hermitian FFT
```rust
pub fn hfft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<FFTNorm>,
) -> Result<Array<f64>, NumPyError>;
// FFT of Hermitian-symmetric signal (real output)

pub fn ihfft<T>(
    input: &Array<T>,
    n: Option<usize>,
    axis: isize,
    norm: Option<FFTNorm>,
) -> Result<Array<Complex64>, NumPyError>;
// Inverse FFT of real signal (Hermitian output)
```

## Implementation Notes
- Use rustfft crate's multi-dimensional support
- Real FFTs use packed format (only store half)
- Handle shape/axes parameters properly

## Files to Modify
- `src/fft.rs` - Add all new functions
- May need to refactor into `src/fft/mod.rs`

## Acceptance Criteria
- [ ] fft2, ifft2 implemented
- [ ] fftn, ifftn implemented
- [ ] rfft, irfft implemented
- [ ] rfft2, irfft2, rfftn, irfftn implemented
- [ ] hfft, ihfft implemented
- [ ] Tests for all functions
- [ ] Match NumPy behavior

## Conflict Avoidance
⚠️ **COORDINATION:**
- May need to restructure fft module
- Coordinate with any existing fft work

