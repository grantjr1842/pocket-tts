# rust-numpy: random module exports incomplete vs numpy.random

**Issue #573**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/573  
**Created:** 2026-01-29T02:47:37Z  
**Updated:** 2026-01-29T03:24:07Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:random, ralph/numpy:audit  

---

## Summary
NumPy's `numpy.random` module has a complex structure with both legacy (`RandomState`) and modern (`Generator`) APIs. rust-numpy's random module has the structure but is missing some key exports and organization details.

## NumPy random Module Reference

### Modern API (NumPy >= 1.17)
```python
# Primary entry point
numpy.random.default_rng(seed=None) -> Generator

# Generator class - main random number generator
numpy.random.Generator
    .random(size=None, dtype=np.float64)           # [0, 1) uniform
    .integers(low, high=None, size=None, dtype=int)  # Random integers
    .standard_normal(size=None, dtype=np.float64)
    .normal(loc=0.0, scale=1.0, size=None)
    .uniform(low=0.0, high=1.0, size=None)
    .standard_exponential(size=None)
    .exponential(scale=1.0, size=None)
    .standard_gamma(shape, size=None)
    .gamma(shape, scale=1.0, size=None)
    .beta(a, b, size=None)
    .binomial(n, p, size=None)
    .poisson(lam=1.0, size=None)
    .chisquare(df, size=None)
    .f(dfnum, dfden, size=None)
    .standard_t(df, size=None)
    .hypergeometric(ngood, nbad, nsample, size=None)
    .multinomial(n, pvals, size=None)
    .dirichlet(alpha, size=None)
    .multivariate_normal(mean, cov, size=None)
    .standard_cauchy(size=None)
    .standard_exponential(size=None)
    .geometric(p, size=None)
    .negative_binomial(n, p, size=None)
    .lognormal(mean=0.0, sigma=1.0, size=None)
    .pareto(a, size=None)
    .weibull(a, size=None)
    .wald(mean, scale, size=None)
    .rayleigh(scale=1.0, size=None)
    .triangular(left, mode, right, size=None)
    .logseries(p, size=None)
    .zipf(a, size=None)
    .power(a, size=None)
    .vonmises(mu, kappa, size=None)
    .choice(a, size=None, replace=True, p=None, axis=0, shuffle=True)
    .shuffle(x)
    .permutation(x, axis=0)
    .permuted(x, axis=None)
    
# BitGenerator base class
numpy.random.BitGenerator
    .random_raw(size=None)
    .random_uintegers(size=None, bits=64)
    .state  # property
    
# Concrete BitGenerator implementations
numpy.random.PCG64      # Recommended (default)
numpy.random.PCG64DXSM  # PCG64 with DXSM output function
numpy.random.MT19937    # Mersenne Twister (legacy compatibility)
numpy.random.SFC64      # Small Fast Chaotic
numpy.random.Philox     # Counter-based (parallel friendly)
```

### Legacy API (deprecated but maintained)
```python
numpy.random.RandomState
    # Same methods as Generator but different defaults/signatures
    .random_sample(size=None)      # Generator uses .random()
    .randint(low, high=None, size=None, dtype=int)  # Generator uses .integers()
    .random_integers(low, high=None, size=None)  # DEPRECATED
    .rand(d0, d1, ..., dn)        # Convenience for uniform [0,1)
    .randn(d0, d1, ..., dn)       # Convenience for standard normal
    .seed(seed=None)
    .get_state()
    .set_state(state)

# Module-level convenience functions (all use global RandomState)
numpy.random.seed(seed=None)
numpy.random.get_state()
numpy.random.set_state(state)
numpy.random.rand(d0, d1, ..., dn)
numpy.random.randn(d0, d1, ..., dn)
numpy.random.randint(low, high=None, size=None, dtype=int)
numpy.random.random_sample(size=None)
numpy.random.random(size=None)
numpy.random.ranf(size=None)
numpy.random.sample(size=None)
numpy.random.choice(a, size=None, replace=True, p=None)
numpy.random.shuffle(x)
numpy.random.permutation(x)
```

### NumPy 2.0 Additions
```python
numpy.random.Generator.integers()  # Renamed from randint
numpy.random.Generator.random()    # Renamed from random_sample
numpy.random.BitGenerator.spawn_key  # For parallel random streams
numpy.random.SeedSequence          # For spawnable sequences
```

## Current rust-numpy random State

### In `src/random/mod.rs`
```rust
pub mod bit_generator;
pub mod generator;
pub mod random_state;

pub use random_state::RandomState;

// Thread-local default RNGs
// ...

// Module-level functions for modern API
pub fn random<T>(shape: &[usize], dtype: Dtype) -> Result<Array<T>, NumPyError>
pub fn randint<T>(low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
pub fn uniform<T>(low: T, high: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
pub fn normal<T>(mean: T, std: T, shape: &[usize]) -> Result<Array<T>, NumPyError>
// ... many more distributions

// Modern and legacy sub-modules
pub mod modern {
    pub use super::bit_generator::{BitGenerator, PCG64};
    pub use super::generator::Generator;
    pub use super::random_state::RandomState;
    pub use super::{default_rng, default_rng_with_seed};
}

#[deprecated(since = "0.1.0", note = "Use modern Generator API instead")]
pub mod legacy {
    pub use super::RandomState;
    pub use super::{legacy_randint, legacy_random, legacy_rng, seed};
}
```

### In `src/random/generator.rs`
```rust
pub struct Generator { ... }
impl Generator {
    pub fn random<T>(...) -> Result<Array<T>, NumPyError>
    pub fn randint<T>(...) -> Result<Array<T>, NumPyError>
    // ... etc
}
```

### In `src/random/bit_generator.rs`
```rust
pub trait BitGenerator: Send { ... }
pub struct PCG64 { ... }
impl BitGenerator for PCG64 { ... }
```

### In `src/random/random_state.rs`
```rust
pub struct RandomState { ... }
impl RandomState {
    pub fn random<T>(...) -> Result<Array<T>, NumPyError>
    pub fn randint<T>(...) -> Result<Array<T>, NumPyError>
    // ... etc
}
```

## Problems Identified

### 1. Missing BitGenerator Implementations
- ❌ `MT19937` (Mersenne Twister)
- ❌ `SFC64` (Small Fast Chaotic)
- ❌ `Philox` (Counter-based)
- ❌ `PCG64DXSM` variant

### 2. Missing Generator Methods
- ❌ `choice()` - Random sampling
- ❌ `shuffle()` - In-place shuffling
- ❌ `permutation()` - Random permutation
- ❌ `permuted()` - Element-wise random permutation
- ❌ `multivariate_normal()`
- ❌ `standard_t()`
- ❌ `vonmises()` (commented as TODO)
- ❌ `logistic()` (commented as TODO)
- ❌ `power()` (commented as TODO)

### 3. Missing Legacy Functions
- ❌ `rand()` - Convenience for uniform
- ❌ `randn()` - Convenience for normal
- ❌ `random_sample()` (legacy name)
- ❌ `ranf()` (legacy alias)
- ❌ `sample()` (legacy alias)
- ❌ `random_integers()` (deprecated but should exist)

### 4. Missing NumPy 2.0 Features
- ❌ `SeedSequence` for spawnable streams
- ❌ `BitGenerator.spawn_key`
- ❌ `Generator.integers()` (in NumPy 2.0, `randint` deprecated for `integers`)

### 5. Module Organization
Current `modern` and `legacy` submodules are good, but:
- Should `default_rng` be at root or only in `modern`?
- NumPy has all convenience functions at `numpy.random.*`, not in submodules

## Required Changes

### 1. Add Missing BitGenerators
```rust
// src/random/bit_generator.rs
pub struct MT19937 { ... }
pub struct SFC64 { ... }
pub struct Philox { ... }
pub struct PCG64DXSM { ... }

impl BitGenerator for MT19937 { ... }
// etc
```

### 2. Add Missing Generator Methods
```rust
// src/random/generator.rs
impl Generator {
    pub fn choice<T>(&mut self, a: &[T], size: Option<&[usize]>, replace: bool, p: Option<&[f64]>) -> Result<Array<T>, NumPyError>;
    pub fn shuffle<T>(&mut self, x: &mut Array<T>) -> Result<(), NumPyError>;
    pub fn permutation<T>(&mut self, x: Either<usize, &[T]>, axis: isize) -> Result<Array<T>, NumPyError>;
    pub fn permuted<T>(&mut self, x: &Array<T>, axis: Option<isize>) -> Result<Array<T>, NumPyError>;
    pub fn multivariate_normal(&mut self, mean: &[f64], cov: &[&[f64]], size: Option<&[usize]>) -> Result<Array<f64>, NumPyError>;
    pub fn standard_t(&mut self, df: f64, size: Option<&[usize]>) -> Result<Array<f64>, NumPyError>;
    // ... etc
}
```

### 3. Add Legacy Convenience Functions
```rust
// src/random/mod.rs

/// Generate uniform [0, 1) random numbers (legacy convenience)
#[deprecated(note = "Use random() or Generator::random() instead")]
pub fn rand<T>(d0: usize, d1: Option<usize>, ...) -> Result<Array<T>, NumPyError>;

/// Generate standard normal random numbers (legacy convenience)
#[deprecated(note = "Use standard_normal() or Generator::standard_normal() instead")]
pub fn randn<T>(d0: usize, d1: Option<usize>, ...) -> Result<Array<T>, NumPyError>;

/// Legacy alias for random()
#[deprecated(note = "Use random() instead")]
pub fn random_sample<T>(size: &[usize]) -> Result<Array<T>, NumPyError>;

/// Legacy alias
#[deprecated(note = "Use random() instead")]
pub fn ranf<T>(size: &[usize]) -> Result<Array<T>, NumPyError>;

/// Legacy alias
#[deprecated(note = "Use random() instead")]
pub fn sample<T>(size: &[usize]) -> Result<Array<T>, NumPyError>;

/// Legacy random integers (deprecated even in NumPy)
#[deprecated(note = "Use randint() instead")]
pub fn random_integers<T>(low: T, high: Option<T>, size: &[usize]) -> Result<Array<T>, NumPyError>;
```

### 4. Add SeedSequence (NumPy 2.0)
```rust
// src/random/seed_sequence.rs
pub struct SeedSequence {
    entropy: u64,
    spawn_key: Vec<u64>,
    n_children_spawned: usize,
}

impl SeedSequence {
    pub fn new(entropy: Option<u64>) -> Self;
    pub fn spawn(&mut self, n_children: usize) -> Vec<Self>;
    pub fn generate_state(&self, n_words: usize, dtype: Dtype) -> Vec<u64>;
}
```

### 5. Reorganize Exports
```rust
// src/random/mod.rs - Make all convenience functions available at root
// (already mostly done, but verify completeness)

// Re-exports
pub use generator::Generator;
pub use bit_generator::{BitGenerator, PCG64, MT19937, SFC64, Philox, PCG64DXSM};
pub use random_state::RandomState;
pub use seed_sequence::SeedSequence;

// All module-level functions
pub use self::{
    beta, binomial, chisquare, dirichlet, exponential, f, gamma, geometric,
    gumbel, hypergeometric, lognormal, logseries, multinomial, multivariate_normal,
    negative_binomial, normal, pareto, permutation, poisson, power, rand, randint,
    randn, random, rayleigh, standard_cauchy, standard_exponential, standard_gamma,
    standard_normal, standard_t, triangular, uniform, vonmises, wald, weibull, zipf,
    choice, shuffle, 
};

// Keep modern/legacy submodules for organization
pub mod modern { ... }
pub mod legacy { ... }
```

## Acceptance Criteria
- [ ] All BitGenerator implementations (PCG64, PCG64DXSM, MT19937, SFC64, Philox)
- [ ] All Generator methods from NumPy docs
- [ ] All legacy RandomState methods
- [ ] All module-level convenience functions
- [ ] SeedSequence for spawnable streams
- [ ] Proper deprecation attributes on legacy functions
- [ ] All exports match NumPy's structure

## Affected Files
- `rust-numpy/src/random/mod.rs` - Update exports
- `rust-numpy/src/random/generator.rs` - Add missing methods
- `rust-numpy/src/random/bit_generator.rs` - Add missing BitGenerators
- `rust-numpy/src/random/random_state.rs` - Add missing methods
- New: `rust-numpy/src/random/seed_sequence.rs`


---

## Sub-Issues (for parallel development)

| Issue | Title | Scope |
|-------|-------|-------|
| #587 | Implement Missing BitGenerators | MT19937, Philox, SFC64, PCG64DXSM |
| #589 | Implement Generator Methods | choice, shuffle, permutation, permuted |
| #588 | Implement SeedSequence | Spawnable random streams |

**Coordination Notes:**
- #587: New BitGenerator implementations
- #589: Generator class methods
- #588: SeedSequence for parallel streams
- Minimal overlap between these

