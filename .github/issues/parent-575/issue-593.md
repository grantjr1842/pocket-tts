# [Sub-Issue] Implement Memory-Mapped and Buffer Functions (#575)

**Issue #593**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/593  
**Created:** 2026-01-29T03:09:09Z  
**Updated:** 2026-01-29T03:24:21Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:fft-io, ralph/numpy:impl  

---

## Parent Issue
Part of #575 - fft and io modules missing many NumPy functions

## Scope
Implement Memory-Mapped and Buffer Functions (memmap, frombuffer, fromfile, tofile)

## Background
NumPy supports memory-mapped files for large arrays and buffer protocol for interoperability.

## Tasks

### 1. Implement memmap
```rust
// src/io.rs or src/memmap.rs
pub struct MemmapArray<T> {
    file: File,
    mmap: Mmap,
    shape: Vec<usize>,
    dtype: Dtype,
    phantom: PhantomData<T>,
}

impl<T> MemmapArray<T> {
    /// Open memory-mapped array
    pub fn open<P: AsRef<Path>>(
        path: P,
        mode: &str,  // 'r', 'r+', 'w+', 'c'
        dtype: Dtype,
        shape: Option<&[usize]>,
        order: &str,
    ) -> Result<Self, NumPyError>;
    
    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<(), NumPyError>;
    
    /// Get as Array view
    pub fn as_array(&self) -> Result<ArrayView<T>, NumPyError>;
}

impl<T> Deref for MemmapArray<T> {
    type Target = [T];
    // ...
}
```

### 2. Implement frombuffer
```rust
pub fn frombuffer<T>(
    buffer: &[u8],
    dtype: Dtype,
    count: isize,  // -1 means all
    offset: usize,
) -> Result<Array<T>, NumPyError> {
    // Create array view into existing buffer
    // No copy - shares memory
}
```

### 3. Implement fromfile
```rust
pub fn fromfile<P: AsRef<Path>, T>(
    path: P,
    dtype: Dtype,
    count: isize,
    sep: Option<&str>,  // None for binary, string for text
) -> Result<Array<T>, NumPyError> {
    // If sep is None: binary read
    // If sep is Some: text read with separator
}
```

### 4. Implement tofile (Array method)
```rust
impl<T> Array<T> {
    pub fn tofile<P: AsRef<Path>>(
        &self,
        path: P,
        sep: Option<&str>,
        format: &str,
    ) -> Result<(), NumPyError> {
        // If sep is None: binary write
        // If sep is Some: text write with format
    }
}
```

### 5. Add Dependencies
```toml
[dependencies]
memmap2 = "0.9"  # For memory mapping
```

## Implementation Notes
- Memory mapping uses OS virtual memory
- frombuffer shares memory (no copy)
- Binary format is raw bytes (not .npy format)

## Files to Modify
- NEW: `src/memmap.rs` - Memory-mapped array
- `src/io.rs` - frombuffer, fromfile
- `src/array.rs` - tofile method
- `Cargo.toml` - Add memmap2 dependency

## Acceptance Criteria
- [ ] MemmapArray opens files as memory map
- [ ] frombuffer creates array from byte buffer
- [ ] fromfile reads binary/text files
- [ ] tofile writes binary/text files
- [ ] Tests for memory-mapped arrays
- [ ] Works with large files

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #562 (Add missing I/O functions)
- Coordinate with #547 (IO and Serialization Functions)
- New dependencies - coordinate

