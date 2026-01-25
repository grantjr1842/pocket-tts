# Generator Class Documentation

This document describes the enhanced Generator class that wraps BitGenerator instances to provide comprehensive random number generation capabilities matching NumPy's modern random API.

## Overview

The Generator class is the main interface for random number generation in the modern NumPy-compatible API. It wraps a BitGenerator instance and provides all distribution methods, utility functions, and random sampling capabilities.

## Architecture

```
Generator
├── BitGenerator (PCG64, MT19937, Philox, etc.)
├── Core Methods (random, randint, uniform)
├── Statistical Distributions (normal, gamma, beta, etc.)
├── Discrete Distributions (binomial, poisson, bernoulli, etc.)
├── Utility Methods (shuffle, permutation, choice, sample)
└── Random Generation (bytes, floats, booleans, integers)
```

## Creating Generator Instances

### Default Generator (PCG64)

```rust
use rust_numpy::random;

// Create default Generator with PCG64
let mut rng = random::default_rng();

// Create seeded Generator for reproducible results
let mut seeded_rng = random::default_rng_with_seed(42);
```

### Manual Generator Creation

```rust
use rust_numpy::random::{Generator, PCG64};

// Create Generator with specific BitGenerator
let pcg = PCG64::new();
let mut rng = Generator::new(Box::new(pcg));

// Create seeded BitGenerator
let seeded_pcg = PCG64::seed_from_u64(12345);
let mut seeded_rng = Generator::new(Box::new(seeded_pcg));
```

## Core Methods

### random()

Generate random numbers in the half-open interval [0.0, 1.0).

```rust
let mut rng = random::default_rng();
let arr = rng.random::<f64>(&[3, 4], rust_numpy::Dtype::Float64)?;
// Returns: Array<f64> with shape [3, 4] and values in [0.0, 1.0)
```

### randint()

Generate random integers in the range [low, high).

```rust
let mut rng = random::default_rng();
let arr = rng.randint::<i32>(0, 10, &[2, 3])?;
// Returns: Array<i32> with shape [2, 3] and values in [0, 10)
```

### uniform()

Generate samples from a uniform distribution over [low, high).

```rust
let mut rng = random::default_rng();
let arr = rng.uniform::<f64>(0.0, 5.0, &[2, 2])?;
// Returns: Array<f64> with shape [2, 2] and values in [0.0, 5.0)
```

## Statistical Distributions

### Normal Distribution

```rust
// Normal (Gaussian) distribution
let normal_arr = rng.normal::<f64>(0.0, 1.0, &[1000])?;
let std_normal_arr = rng.standard_normal::<f64>(&[1000])?;
```

### Gamma and Beta Distributions

```rust
// Gamma distribution
let gamma_arr = rng.gamma::<f64>(2.0, 2.0, &[1000])?;

// Beta distribution
let beta_arr = rng.beta::<f64>(2.0, 2.0, &[1000])?;
```

### Chi-Square and F-Distribution

```rust
// Chi-square distribution
let chi_arr = rng.chisquare::<f64>(2.0, &[1000])?;

// F-distribution
let f_arr = rng.f::<f64>(2.0, 2.0, &[1000])?;
```

## Discrete Distributions

### Binomial Distribution

```rust
let bin_arr = rng.binomial::<f64>(10, 0.5, &[1000])?;
// n=10 trials, p=0.5 probability of success
```

### Poisson Distribution

```rust
let pois_arr = rng.poisson::<f64>(5.0, &[1000])?;
// λ=5.0 average rate
```

### Bernoulli Distribution

```rust
let bern_arr = rng.bernoulli::<f64>(0.7, &[1000])?;
// p=0.7 probability of success (equivalent to binomial(n=1, p))
```

### Geometric and Negative Binomial

```rust
// Geometric distribution
let geom_arr = rng.geometric::<f64>(0.5, &[1000])?;
// p=0.5 probability of success

// Negative binomial distribution
let neg_bin_arr = rng.negative_binomial::<f64>(5, 0.5, &[1000])?;
// n=5 failures, p=0.5 probability
```

## Multivariate Distributions

### Multinomial Distribution

```rust
let pvals = vec![0.2, 0.3, 0.5];  // Probabilities for each category
let multi_arr = rng.multinomial::<f64>(10, &pvals, None)?;
// n=10 trials, returns counts for each category
```

### Dirichlet Distribution

```rust
let alpha = vec![1.0, 2.0, 3.0];  // Concentration parameters
let dir_arr = rng.dirichlet::<f64>(&alpha, None)?;
// Returns random proportions that sum to 1.0
```

## Utility Methods

### shuffle()

Shuffle an array in-place along the first axis.

```rust
let mut arr = rust_numpy::array![1, 2, 3, 4, 5, 6];
rng.shuffle(&mut arr)?;
// arr is now shuffled
```

### permutation()

Generate a random permutation of integers.

```rust
let perm = rng.permutation(5)?;
// Returns: Array<i64> with values [0, 1, 2, 3, 4] in random order
```

### choice()

Random sampling from an array.

```rust
let choices = vec![1, 2, 3, 4, 5];

// With replacement
let sample_replace = rng.choice(&choices, 3, true)?;
// Returns 3 random elements, possibly with duplicates

// Without replacement
let sample_no_replace = rng.choice(&choices, 3, false)?;
// Returns 3 unique random elements
```

### integers()

Generate random integers from a range.

```rust
let int_arr = rng.integers(0, 100, 10)?;
// Returns 10 random integers in [0, 100)
```

## Random Generation Methods

### Bytes

```rust
let bytes_arr = rng.bytes(16)?;
// Returns 16 random bytes
```

### Floating Point Numbers

```rust
// Random floats in [0.0, 1.0)
let floats_arr = rng.random_floats(10)?;

// Random floats in specified range
let range_arr = rng.random_floats_range(10.0, 20.0, 10)?;
// Returns 10 random floats in [10.0, 20.0)
```

### Boolean Values

```rust
let bool_arr = rng.random_bools(10)?;
// Returns 10 random boolean values
```

## Advanced Distributions

### Log-Normal Distribution

```rust
let lognorm_arr = rng.lognormal::<f64>(0.0, 1.0, &[1000])?;
// μ=0.0, σ=1.0
```

### Logistic Distribution

```rust
let logistic_arr = rng.logistic::<f64>(0.0, 1.0, &[1000])?;
// μ=0.0 (location), s=1.0 (scale)
```

### Gumbel Distribution

```rust
let gumbel_arr = rng.gumbel::<f64>(0.0, 1.0, &[1000])?;
// μ=0.0 (location), β=1.0 (scale)
```

### Weibull Distribution

```rust
let weibull_arr = rng.weibull::<f64>(2.0, &[1000])?;
// k=2.0 (shape parameter)
```

### Triangular Distribution

```rust
let tri_arr = rng.triangular::<f64>(0.0, 0.5, 1.0, &[1000])?;
// left=0.0, mode=0.5, right=1.0
```

### Pareto Distribution

```rust
let pareto_arr = rng.pareto::<f64>(2.0, &[1000])?;
// α=2.0 (shape parameter)
```

### Zipf Distribution

```rust
let zipf_arr = rng.zipf::<f64>(3.0, &[1000])?;
// a=3.0 (exponent)
```

### Special Distributions

```rust
// Standard Cauchy distribution
let cauchy_arr = rng.standard_cauchy::<f64>(&[1000])?;

// Standard exponential distribution
let exp_arr = rng.standard_exponential::<f64>(&[1000])?;

// Standard gamma distribution
let std_gamma_arr = rng.standard_gamma::<f64>(2.0, &[1000])?;
```

## Error Handling

All Generator methods return `Result<Array<T>, NumPyError>` to handle potential errors:

```rust
match rng.normal::<f64>(0.0, -1.0, &[1000]) {
    Ok(arr) => {
        // Use the generated array
        println!("Generated {} values", arr.len());
    }
    Err(e) => {
        // Handle the error
        eprintln!("Error: {}", e);
    }
}
```

Common error types:
- Invalid parameters (negative scale, probability outside [0,1], etc.)
- Invalid ranges (low >= high)
- Empty input arrays
- Memory allocation failures

## Thread Safety

- Generator instances are NOT thread-safe
- Each thread should create its own Generator instance
- For multi-threaded applications, create separate Generators per thread

```rust
use std::thread;

let handles: Vec<_> = (0..4).map(|_| {
    let mut rng = random::default_rng();
    thread::spawn(move || {
        rng.random::<f64>(&[1000], rust_numpy::Dtype::Float64)
    })
}).collect();

for handle in handles {
    let arr = handle.join().unwrap()?;
    // Process arr in main thread
}
```

## Performance Considerations

### BitGenerator Choice

- **PCG64**: Good performance, good statistical properties (default)
- **MT19937**: Excellent statistical properties, slower performance
- **Philox**: Cryptographically secure, slower performance

### Memory Usage

- Generator methods allocate new arrays for each call
- For high-performance scenarios, reuse Generator instances
- Consider pre-allocating arrays when possible

### Reproducibility

- Use seeded BitGenerators for reproducible results
- Different BitGenerators with the same seed produce different sequences
- Thread-local generators are independent

## Compatibility with NumPy

| NumPy Method                  | Rust Method                    | Status       |
|-------------------------------|--------------------------------|--------------|
| `Generator.random()`          | `Generator::random()`          | ✅ Compatible |
| `Generator.randint()`         | `Generator::randint()`         | ✅ Compatible |
| `Generator.normal()`          | `Generator::normal()`          | ✅ Compatible |
| `Generator.standard_normal()` | `Generator::standard_normal()` | ✅ Compatible |
| `Generator.uniform()`         | `Generator::uniform()`         | ✅ Compatible |
| `Generator.shuffle()`         | `Generator::shuffle()`         | ✅ Compatible |
| `Generator.permutation()`     | `Generator::permutation()`     | ✅ Compatible |
| `Generator.choice()`          | `Generator::choice()`          | ✅ Compatible |
| `Generator.integers()`        | `Generator::integers()`        | ✅ Compatible |
| `Generator.bytes()`           | `Generator::bytes()`           | ✅ Compatible |

## Examples

### Basic Usage

```rust
use rust_numpy::random;

let mut rng = random::default_rng();

// Generate random numbers
let random_arr = rng.random::<f64>(&[5, 5], rust_numpy::Dtype::Float64)?;
let int_arr = rng.randint::<i32>(0, 100, &[3, 3])?;
let normal_arr = rng.normal::<f64>(0.0, 1.0, &[1000])?;
```

### Reproducible Results

```rust
let seed = 12345;
let mut rng1 = random::default_rng_with_seed(seed);
let mut rng2 = random::default_rng_with_seed(seed);

let arr1 = rng1.random::<f64>(&[2, 2], rust_numpy::Dtype::Float64)?;
let arr2 = rng2.random::<f64>(&[2, 2], rust_numpy::Dtype::Float64)?;

// arr1 and arr2 have identical sequences
```

### Array Manipulation

```rust
let mut arr = rust_numpy::array![1, 2, 3, 4, 5, 6];
rng.shuffle(&mut arr)?;

let perm = rng.permutation(5)?;
let sample = rng.choice(&[1, 2, 3, 4, 5], 3, false)?;
```

### Statistical Analysis

```rust
// Generate samples from different distributions
let normal_sample = rng.normal::<f64>(0.0, 1.0, &[10000])?;
let gamma_sample = rng.gamma::<f64>(2.0, 2.0, &[10000])?;
let beta_sample = rng.beta::<f64>(2.0, 2.0, &[10000])?;

// Analyze distributions
println!("Normal mean: {:.3}", normal_sample.data.as_slice().iter().sum::<f64>() / 10000.0);
println!("Gamma mean: {:.3}", gamma_sample.data.as_slice().iter().sum::<f64>() / 10000.0);
println!("Beta mean: {:.3}", beta_sample.data.as_slice().iter().sum::<f64>() / 10000.0);
```

## Testing

The Generator class includes comprehensive tests covering:

- All distribution methods
- Error handling for invalid parameters
- Reproducibility with seeded generators
- Thread safety considerations
- Edge cases and boundary conditions
- Compatibility with different array types

Run tests with:
```bash
cargo test generator
```

## Best Practices

1. **Use seeded generators for reproducible results** when needed for testing or debugging
2. **Create separate Generator instances for each thread** in multi-threaded applications
3. **Choose BitGenerator based on requirements**:
   - PCG64 for general use (default)
   - MT19937 for statistical quality
   - Philox for cryptographic security
4. **Handle errors appropriately** using Result types
5. **Reuse Generator instances** for performance-critical code
6. **Validate parameters** before calling distribution methods

## Migration from Legacy API

```rust
// Old (RandomState API)
use rust_numpy::random::legacy;

legacy::seed(42);
let old_arr = legacy::legacy_random::<f64>(&[2, 2], rust_numpy::Dtype::Float64)?;

// New (Generator API) - Recommended
use rust_numpy::random;

let mut rng = random::default_rng_with_seed(42);
let new_arr = rng.random::<f64>(&[2, 2], rust_numpy::Dtype::Float64)?;

// Both produce arrays with same shape, but Generator API is more flexible
```
