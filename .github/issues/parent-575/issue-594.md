# [Sub-Issue] Implement NPZ Format Support (#575)

**Issue #594**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/594  
**Created:** 2026-01-29T03:09:09Z  
**Updated:** 2026-01-29T03:24:22Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:fft-io, ralph/numpy:impl  

---

## Parent Issue
Part of #575 - fft and io modules missing many NumPy functions

## Scope
Implement NPZ File Format Support (savez, savez_compressed, load)

## Background
NumPy uses .npz format for saving multiple arrays. rust-numpy only has basic .npy support.

## Tasks

### 1. Implement savez()
```rust
// src/io.rs
pub fn savez<P: AsRef<Path>>(
    path: P,
    arrays: &[(&str, &Array<T>)],
) -> Result<(), NumPyError> {
    // Save multiple arrays to uncompressed .npz file
    // .npz is a ZIP archive containing .npy files
}
```

### 2. Implement savez_compressed()
```rust
pub fn savez_compressed<P: AsRef<Path>>(
    path: P,
    arrays: &[(&str, &Array<T>)],
    compression_level: Option<u32>,
) -> Result<(), NumPyError> {
    // Save with ZIP compression
}
```

### 3. Enhance load() for NPZ
```rust
pub fn load<P: AsRef<Path>>(
    path: P,
    mmap_mode: Option<&str>,
    allow_pickle: bool,
    encoding: &str,
) -> Result<Either<Array<T>, HashMap<String, Array<T>>>, NumPyError> {
    // If .npy: return Array<T>
    // If .npz: return HashMap<String, Array<T>>
}
```

### 4. Add Dependencies
```toml
# Cargo.toml
[dependencies]
zip = "2.0"  # For NPZ format
```

### 5. Implement NPZ Iterator
```rust
pub struct NpzArchive {
    // Iterate over arrays in NPZ file
}

impl NpzArchive {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, NumPyError>;
    pub fn names(&self) -> Vec<String>;
    pub fn get<T>(&self, name: &str) -> Result<Option<Array<T>>, NumPyError>;
}
```

## Files to Modify
- `src/io.rs` - Add NPZ functions
- `Cargo.toml` - Add zip dependency

## Acceptance Criteria
- [ ] savez() saves uncompressed .npz
- [ ] savez_compressed() saves compressed .npz
- [ ] load() reads both .npy and .npz
- [ ] Roundtrip test: save -> load works
- [ ] Compatible with NumPy's .npz files

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #562 (Add missing I/O functions)
- Coordinate with #547 (IO and Serialization Functions)
- May need Cargo.toml changes - coordinate

