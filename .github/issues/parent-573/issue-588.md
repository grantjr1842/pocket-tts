# [Sub-Issue] Implement SeedSequence for Spawnable Streams (#573)

**Issue #588**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/588  
**Created:** 2026-01-29T03:06:24Z  
**Updated:** 2026-01-29T03:24:10Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:random, ralph/numpy:impl  

---

## Parent Issue
Part of #573 - random module exports incomplete vs numpy.random

## Scope
Implement SeedSequence for Spawnable Random Streams

## Background
NumPy 2.0 introduced SeedSequence for creating independent random streams, useful for parallel computing.

## Tasks

### 1. Create SeedSequence Struct
```rust
// src/random/seed_sequence.rs
pub struct SeedSequence {
    entropy: u64,
    spawn_key: Vec<u64>,
    n_children_spawned: usize,
}
```

### 2. Implement Core Methods
```rust
impl SeedSequence {
    /// Create new SeedSequence
    pub fn new(entropy: Option<u64>) -> Self;
    
    /// Spawn child SeedSequences
    pub fn spawn(&mut self, n_children: usize) -> Vec<Self>;
    
    /// Generate state for BitGenerator
    pub fn generate_state(&self, n_words: usize, dtype: Dtype) -> Vec<u64>;
    
    /// Create BitGenerator from this sequence
    pub fn spawn_key(&self) -> &[u64];
}
```

### 3. Integrate with BitGenerator
```rust
// Add to BitGenerator trait or implementations
pub trait BitGenerator {
    fn from_seed_sequence(ss: &SeedSequence) -> Self;
    fn spawn_key(&self) -> Option<&[u64]>;
    // ...
}
```

### 4. Update Generator
```rust
impl Generator {
    /// Create Generator with SeedSequence
    pub fn from_seed_sequence(ss: &SeedSequence) -> Self;
    
    /// Spawn independent Generator
    pub fn spawn(&mut self) -> Self;
}
```

### 5. Export
```rust
// src/random/mod.rs
pub use seed_sequence::SeedSequence;
```

## Files to Create/Modify
- NEW: `src/random/seed_sequence.rs`
- `src/random/mod.rs` - Export
- `src/random/bit_generator.rs` - Integrate
- `src/random/generator.rs` - Add spawn support

## Acceptance Criteria
- [ ] SeedSequence struct created
- [ ] spawn() works correctly
- [ ] generate_state() produces valid seeds
- [ ] Integration with BitGenerator works
- [ ] Can create independent random streams
- [ ] Tests verify independence

## Conflict Avoidance
⚠️ **COORDINATION:**
- New file - minimal conflict
- Requires BitGenerator modifications - coordinate

