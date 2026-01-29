# [Sub-Issue] Implement Text File I/O (loadtxt, savetxt, genfromtxt) (#575)

**Issue #592**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/592  
**Created:** 2026-01-29T03:09:09Z  
**Updated:** 2026-01-29T03:24:19Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:fft-io, ralph/numpy:impl  

---

## Parent Issue
Part of #575 - fft and io modules missing many NumPy functions

## Scope
Implement Text File I/O Functions (loadtxt, savetxt, genfromtxt)

## Background
NumPy can load/save arrays from/to text files. rust-numpy has basic from_txt/to_txt but not full NumPy compatibility.

## Tasks

### 1. Implement loadtxt()
```rust
// src/io.rs
pub fn loadtxt<P: AsRef<Path>, T>(
    path: P,
    dtype: Dtype,
    comments: &str,
    delimiter: Option<&str>,
    converters: Option<HashMap<usize, Box<dyn Fn(&str) -> T>>>,
    skiprows: usize,
    usecols: Option<&[usize]>,
    unpack: bool,
    ndmin: usize,
    encoding: &str,
    max_rows: Option<usize>,
    quotechar: Option<char>,
) -> Result<Array<T>, NumPyError>;
```

### 2. Implement genfromtxt()
```rust
pub fn genfromtxt<P: AsRef<Path>, T>(
    path: P,
    dtype: Dtype,
    comments: &str,
    delimiter: Option<&str>,
    skip_header: usize,
    skip_footer: usize,
    converters: Option<...>,
    missing_values: Option<&[&str]>,
    filling_values: Option<T>,
    usecols: Option<&[usize]>,
    names: Option<Names>,  // true, false, or list of names
    excludelist: Option<&[&str]>,
    deletechars: &str,
    replace_space: &str,
    autostrip: bool,
    case_sensitive: bool,
    defaultfmt: &str,
    unpack: bool,
    usemask: bool,
    loose: bool,
    invalid_raise: bool,
    max_rows: Option<usize>,
    encoding: &str,
) -> Result<Either<Array<T>, MaskedArray<T>>, NumPyError>;
```

### 3. Implement savetxt()
```rust
pub fn savetxt<P: AsRef<Path>, T>(
    path: P,
    arr: &Array<T>,
    fmt: &str,
    delimiter: &str,
    newline: &str,
    header: &str,
    footer: &str,
    comments: &str,
    encoding: Option<&str>,
) -> Result<(), NumPyError>;
```

### 4. Support Structured Arrays (for genfromtxt)
```rust
// May need to enhance dtype to support structured types
pub enum Dtype {
    // ... existing variants
    Structured(Vec<(String, Dtype)>),  // Named fields
}
```

### 5. Handle Missing Values
```rust
// Parse strings like "N/A", "NaN", "missing" as missing
// Use MaskedArray or fill values
```

## Implementation Notes
- Handle various delimiters (space, comma, tab)
- Support comments in files
- Parse headers for column names
- Handle missing/invalid data

## Files to Modify
- `src/io.rs` - Add text I/O functions
- `src/dtype.rs` - May need structured dtype support

## Acceptance Criteria
- [ ] loadtxt() reads various text formats
- [ ] savetxt() writes text files
- [ ] genfromtxt() handles missing data
- [ ] Structured array support for column names
- [ ] Tests with various formats

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #562 (Add missing I/O functions)
- Coordinate with #547 (IO and Serialization Functions)
- May overlap - communicate with owners

