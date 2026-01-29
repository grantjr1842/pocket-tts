# rust-numpy: fft and io modules missing many NumPy functions

**Issue #575**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/575  
**Created:** 2026-01-29T02:47:43Z  
**Updated:** 2026-01-29T03:24:17Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:fft-io, ralph/numpy:audit  

---

## Summary
NumPy's `numpy.fft` and `numpy` I/O functions have specific export structures that need to be verified and completed.

## NumPy fft Module Reference

### Standard FFTs
```python
numpy.fft.fft(a, n=None, axis=-1, norm=None)           # 1-D FFT
numpy.fft.ifft(a, n=None, axis=-1, norm=None)          # 1-D inverse FFT
numpy.fft.fft2(a, s=None, axes=(-2, -1), norm=None)    # 2-D FFT
numpy.fft.ifft2(a, s=None, axes=(-2, -1), norm=None)   # 2-D inverse FFT
numpy.fft.fftn(a, s=None, axes=None, norm=None)        # N-D FFT
numpy.fft.ifftn(a, s=None, axes=None, norm=None)       # N-D inverse FFT
```

### Real FFTs
```python
numpy.fft.rfft(a, n=None, axis=-1, norm=None)          # Real-input 1-D FFT
numpy.fft.irfft(a, n=None, axis=-1, norm=None)         # Inverse of rfft
numpy.fft.rfft2(a, s=None, axes=(-2, -1), norm=None)   # Real-input 2-D FFT
numpy.fft.irfft2(a, s=None, axes=(-2, -1), norm=None)  # Inverse of rfft2
numpy.fft.rfftn(a, s=None, axes=None, norm=None)       # Real-input N-D FFT
numpy.fft.irfftn(a, s=None, axes=None, norm=None)      # Inverse of rfftn
```

### Hermitian FFTs
```python
numpy.fft.hfft(a, n=None, axis=-1, norm=None)          # Hermitian 1-D FFT
numpy.fft.ihfft(a, n=None, axis=-1, norm=None)         # Inverse of hfft
```

### Helper Routines
```python
numpy.fft.fftfreq(n, d=1.0)                            # FFT sample frequencies
numpy.fft.rfftfreq(n, d=1.0)                           # rfft sample frequencies
numpy.fft.fftshift(x, axes=None)                       # Shift zero-frequency to center
numpy.fft.ifftshift(x, axes=None)                      # Inverse of fftshift
```

### Exceptions
```python
numpy.fft.FftTypeError  # (Note: NumPy doesn't have specific FFT exceptions, uses ValueError)
```

## Current rust-numpy fft State

### In `src/fft.rs`
```rust
pub enum FFTNorm { Backward, Ortho, Forward }

pub fn fft<T>(...) -> Result<Array<Complex64>, NumPyError>
pub fn ifft<T>(...) -> Result<Array<Complex64>, NumPyError>
pub fn fftshift<T>(...) -> Array<T>
pub fn ifftshift<T>(...) -> Array<T>
pub fn fftfreq(n: usize, d: f64) -> Array<f64>
pub fn rfftfreq(n: usize, d: f64) -> Array<f64>
```

## Problems Identified

### 1. Missing FFT Functions
- ❌ `fft2`, `ifft2` - 2-D FFT
- ❌ `fftn`, `ifftn` - N-D FFT
- ❌ `rfft`, `irfft` - Real FFT
- ❌ `rfft2`, `irfft2` - Real 2-D FFT
- ❌ `rfftn`, `irfftn` - Real N-D FFT
- ❌ `hfft`, `ihfft` - Hermitian FFT

### 2. Module Structure
Currently all in `src/fft.rs`. Should be:
- `src/fft/mod.rs` - Module with re-exports
- `src/fft/pocketfft.rs` or similar - Implementation

## Required Changes

### 1. Add Missing Functions
```rust
// src/fft/mod.rs
pub fn fft2<T>(input: &Array<T>, s: Option<(usize, usize)>, axes: (isize, isize), norm: Option<FFTNorm>) -> Result<Array<Complex64>, NumPyError>;
pub fn ifft2<T>(...) -> Result<Array<Complex64>, NumPyError>;
pub fn fftn<T>(input: &Array<T>, s: Option<&[usize]>, axes: Option<&[isize]>, norm: Option<FFTNorm>) -> Result<Array<Complex64>, NumPyError>;
pub fn ifftn<T>(...) -> Result<Array<Complex64>, NumPyError>;

pub fn rfft<T>(...) -> Result<Array<Complex64>, NumPyError>;
pub fn irfft<T>(...) -> Result<Array<f64>, NumPyError>;  // Note: returns real
pub fn rfft2<T>(...) -> Result<Array<Complex64>, NumPyError>;
pub fn irfft2<T>(...) -> Result<Array<f64>, NumPyError>;
pub fn rfftn<T>(...) -> Result<Array<Complex64>, NumPyError>;
pub fn irfftn<T>(...) -> Result<Array<f64>, NumPyError>;

pub fn hfft<T>(...) -> Result<Array<f64>, NumPyError>;
pub fn ihfft<T>(...) -> Result<Array<Complex64>, NumPyError>;
```

### 2. Reorganize Module
```rust
// src/fft/mod.rs
pub use self::core::{
    fft, fft2, fftn, ifft, ifft2, ifftn,
    rfft, rfft2, rfftn, irfft, irfft2, irfftn,
    hfft, ihfft,
    fftfreq, rfftfreq, fftshift, ifftshift,
};
pub use self::types::FFTNorm;

mod core;
mod types;
```

---

## NumPy I/O Functions Reference

### NPZ/NPY Format
```python
numpy.save(file, arr, allow_pickle=True, fix_imports=True)      # Save single array
numpy.savez(file, *args, **kwds)                                 # Save multiple arrays uncompressed
numpy.savez_compressed(file, *args, **kwds)                      # Save multiple arrays compressed
numpy.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')  # Load
```

### Text Files
```python
numpy.loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, 
              skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', 
              max_rows=None, quotechar=None, like=None)
numpy.genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0, 
                 skip_footer=0, converters=None, missing_values=None, filling_values=None, 
                 usecols=None, names=None, excludelist=None, deletechars=None, 
                 replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', 
                 unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, 
                 encoding='bytes', like=None)
numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', 
              comments='# ', encoding=None)
numpy.fromregex(file, regexp, dtype, encoding=None)
numpy.fromstring(string, dtype=float, count=-1, sep='')
numpy.ndarray.tofile(fid, sep='', format='%s')
numpy.fromfile(file, dtype=float, count=-1, sep='')
numpy.ndarray.tobytes(order='C')
numpy.frombuffer(buffer, dtype=float, count=-1, offset=0)
```

### String Representations
```python
numpy.array2string(a, max_line_width=None, precision=None, suppress_small=None, 
                   separator=' ', prefix='', style=None, formatter=None, threshold=None, 
                   edgeitems=None, sign=None, floatmode=None, suffix='', legacy=None)
numpy.array_repr(arr, max_line_width=None, precision=None, suppress_small=None)
numpy.array_str(a, max_line_width=None, precision=None, suppress_small=None)
numpy.format_float_positional(x, precision=None, unique=True, fractional=True, 
                               trim='k', sign=False, pad_left=None, pad_right=None, 
                               min_digits=None)
numpy.format_float_scientific(x, precision=None, unique=True, trim='k', sign=False, 
                               pad_left=None, exp_digits=None, min_digits=None)
```

## Current rust-numpy io State

### In `src/io.rs`
```rust
pub fn from_txt<T>(path: &str, delimiter: char) -> Result<Array<T>, NumPyError>
pub fn to_txt<T>(path: &str, arr: &Array<T>, delimiter: char) -> Result<(), NumPyError>
pub fn from_npy<T>(path: &str) -> Result<Array<T>, NumPyError>
pub fn to_npy<T>(path: &str, arr: &Array<T>) -> Result<(), NumPyError>
```

### In `src/lib.rs` (array_creation)
```rust
pub use array_creation::{array, array2string, array_repr, array_str, ...};
```

## Problems Identified

### 1. Missing I/O Functions
- ❌ `save`, `savez`, `savez_compressed`
- ❌ `load` (only `from_npy` exists)
- ❌ `loadtxt` (only `from_txt` exists)
- ❌ `genfromtxt`
- ❌ `savetxt` (only `to_txt` exists)
- ❌ `fromregex`
- ❌ `fromstring`
- ❌ `fromfile`, `ndarray.tofile`
- ❌ `frombuffer`
- ❌ `format_float_positional`, `format_float_scientific`

### 2. Missing Parameters
Most functions missing NumPy parameters:
- `allow_pickle`, `fix_imports`, `encoding` for save/load
- `comments`, `delimiter`, `converters`, `skiprows`, `usecols`, etc. for loadtxt

## Required Changes

### 1. Add Missing Functions
```rust
// src/io.rs
pub fn save(path: &str, arr: &Array<T>, allow_pickle: bool, fix_imports: bool) -> Result<(), NumPyError>;
pub fn savez(path: &str, arrays: &[(&str, &Array<T>)]) -> Result<(), NumPyError>;
pub fn savez_compressed(path: &str, arrays: &[(&str, &Array<T>)]) -> Result<(), NumPyError>;
pub fn load(path: &str, mmap_mode: Option<&str>, allow_pickle: bool, encoding: &str) -> Result<Either<Array<T>, HashMap<String, Array<T>>>, NumPyError>;

pub fn loadtxt(path: &str, dtype: Dtype, comments: &str, delimiter: Option<char>, 
               skiprows: usize, usecols: Option<&[usize]>, unpack: bool, 
               ndmin: usize, encoding: &str) -> Result<Array<T>, NumPyError>;
pub fn genfromtxt(...) -> Result<Array<T>, NumPyError>;  # Many parameters
pub fn savetxt(path: &str, arr: &Array<T>, fmt: &str, delimiter: &str, 
               newline: &str, header: &str, footer: &str, comments: &str, encoding: Option<&str>) -> Result<(), NumPyError>;

pub fn fromstring<T>(string: &str, dtype: Dtype, count: isize, sep: &str) -> Result<Array<T>, NumPyError>;
pub fn fromfile<T>(path: &str, dtype: Dtype, count: isize, sep: Option<&str>) -> Result<Array<T>, NumPyError>;
pub fn frombuffer<T>(buffer: &[u8], dtype: Dtype, count: isize, offset: usize) -> Result<Array<T>, NumPyError>;

pub fn format_float_positional(x: f64, precision: Option<usize>, ...) -> String;
pub fn format_float_scientific(x: f64, precision: Option<usize>, ...) -> String;
```

### 2. Reorganize Exports
```rust
// src/io/mod.rs
pub use self::core::{
    save, savez, savez_compressed, load,
    loadtxt, genfromtxt, savetxt,
    fromstring, fromfile, frombuffer, fromregex,
    format_float_positional, format_float_scientific,
};
```

## Acceptance Criteria
- [ ] All NumPy fft functions exported at `fft` module
- [ ] All NumPy I/O functions exported at root or `io` module
- [ ] All function parameters match NumPy signatures
- [ ] NPZ format support (multiple arrays)
- [ ] Text format support with all options

## Affected Files
- `rust-numpy/src/fft.rs` - Add missing functions (or split to `fft/mod.rs`)
- `rust-numpy/src/io.rs` - Add missing functions
- `rust-numpy/src/lib.rs` - Update re-exports


---

## Sub-Issues (for parallel development)

| Issue | Title | Scope |
|-------|-------|-------|
| #595 | Implement Missing FFT Functions | fft2, fftn, rfft, hfft families |
| #594 | Implement NPZ Format Support | savez, savez_compressed, load |
| #592 | Implement Text File I/O | loadtxt, savetxt, genfromtxt |
| #593 | Implement Memory-Mapped Functions | memmap, frombuffer, fromfile, tofile |

**Coordination Notes:**
- #595: FFT module expansion
- #594, #592, #593: I/O module expansion
- These can all be worked on independently

