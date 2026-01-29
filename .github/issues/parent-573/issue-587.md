# [Sub-Issue] Implement Missing BitGenerators (MT19937, Philox, SFC64) (#573)

**Issue #587**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/587  
**Created:** 2026-01-29T03:06:24Z  
**Updated:** 2026-01-29T03:24:08Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:random, ralph/numpy:impl  

---

## Parent Issue
Part of #573 - random module exports incomplete vs numpy.random

## Scope
Implement Missing BitGenerator Implementations

## Background
NumPy's random module supports multiple BitGenerator algorithms. rust-numpy currently only has PCG64.

## Tasks

### 1. Implement MT19937 (Mersenne Twister)
```rust
// src/random/bit_generator.rs or src/random/mt19937.rs
pub struct MT19937 {
    state: [u32; 624],
    index: usize,
}

impl MT19937 {
    pub fn new() -> Self;
    pub fn seed_from_u64(seed: u64) -> Self;
    fn twist(&mut self);
}

impl BitGenerator for MT19937 {
    fn random_raw(&mut self) -> u64;
    fn random_uintegers(&mut self, size: usize, bits: u8) -> Vec<u64>;
    fn state(&self) -> Vec<u8>;
    fn set_state(&mut self, state: &[u8]);
}
```

### 2. Implement Philox (Counter-based)
```rust
pub struct Philox {
    key: [u64; 2],
    counter: [u64; 4],
}

impl BitGenerator for Philox { ... }
```

### 3. Implement SFC64 (Small Fast Chaotic)
```rust
pub struct SFC64 {
    a: u64,
    b: u64,
    c: u64,
    counter: u64,
}

impl BitGenerator for SFC64 { ... }
```

### 4. Implement PCG64DXSM (PCG64 variant)
```rust
pub struct PCG64DXSM {
    state: u128,
    inc: u128,
}

impl BitGenerator for PCG64DXSM { ... }
```

### 5. Export All BitGenerators
```rust
// src/random/mod.rs
pub use bit_generator::{
    BitGenerator, PCG64, PCG64DXSM, MT19937, SFC64, Philox
};
```

## Files to Modify/Create
- `src/random/bit_generator.rs` - Add implementations OR
- `src/random/mt19937.rs` - NEW
- `src/random/philox.rs` - NEW
- `src/random/sfc64.rs` - NEW
- `src/random/pcg64dxsm.rs` - NEW
- `src/random/mod.rs` - Export all

## Acceptance Criteria
- [ ] MT19937 implemented
- [ ] Philox implemented
- [ ] SFC64 implemented
- [ ] PCG64DXSM implemented
- [ ] All pass same statistical tests
- [ ] Exported properly

## Conflict Avoidance
⚠️ **COORDINATION:**
- New files - minimal conflict
- May need to modify BitGenerator trait - coordinate with others

