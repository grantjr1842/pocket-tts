# Random Module Documentation

This module provides NumPy-compatible random number generation with both modern (Generator/BitGenerator) and legacy (RandomState) APIs.

## Overview

The random module has been restructured to match NumPy's modern API organization while maintaining backward compatibility:

- **Modern API**: Generator/BitGenerator architecture (recommended for new code)
- **Legacy API**: RandomState-based API (maintained for backward compatibility)
- **Module-level functions**: Convenience functions that delegate to modern Generator API

## Modern API (Recommended)

### Creating Random Number Generators

The modern API uses the `default_rng()` function to create Generator instances:

```rust
use rust_numpy::random;

// Create default Generator with PCG64
let mut rng = random::default_rng();

// Create seeded Generator for reproducible results
let mut seeded_rng = random::default_rng_with_seed(42);

// Manual Generator creation
let pcg = random::PCG64::seed_from_u64(123);
let mut manual_rng = random::Generator::new(Box::new(pcg));
```

### Using Generator Methods

```rust
use rust_numpy::random;
use rust_numpy::Dtype;

let mut rng = random::default_rng();

// Generate random numbers
let random_arr = rng.random::<f64>(&[3, 4], Dtype::Float64)?;

// Generate random integers
let int_arr = rng.randint::<i32>(0, 100, &[2, 3])?;

// Generate from specific distributions
let normal_arr = rng.normal::<f64>(0.0, 1.0, &[2, 2])?;
let uniform_arr = rng.uniform::<f64>(0.0, 10.0, &[3, 3])?;
```

### Available BitGenerators

```rust
use rust_numpy::random;

// PCG64 (default, recommended)
let pcg = random::PCG64::new();
let seeded_pcg = random::PCG64::seed_from_u64(42);

// Using BitGenerator trait
let bit_gen: Box<dyn random::BitGenerator> = Box::new(pcg);
```

## Module-Level Convenience Functions

For convenience, the module provides functions that use a thread-local default Generator:

```rust
use rust_numpy::random;
use rust_numpy::Dtype;

// These use the modern Generator API internally
let random_arr = random::random::<f64>(&[2, 3], Dtype::Float64)?;
let int_arr = random::randint::<i32>(0, 10, &[2, 2])?;
let normal_arr = random::normal::<f64>(0.0, 1.0, &[2, 2])?;
```

## Legacy API (Backward Compatibility)

The legacy RandomState API is maintained for existing code:

```rust
use rust_numpy::random;

// Legacy functions (deprecated but functional)
let legacy_arr = random::legacy_random::<f64>(&[2, 2], Dtype::Float64)?;
let legacy_int_arr = random::legacy_randint::<i32>(0, 10, &[2, 2])?;

// Legacy seeding
random::seed(12345);
```

## Sub-Module Organization

### Modern API Sub-Module

```rust
use rust_numpy::random::modern;

// Access modern API components
let rng = modern::default_rng();
let seeded_rng = modern::default_rng_with_seed(42);
let generator = modern::Generator::new(Box::new(modern::PCG64::new()));
```

### Legacy API Sub-Module

```rust
use rust_numpy::random::legacy;

// Access legacy API components
let legacy = legacy::legacy_rng();
legacy::seed(12345);
let arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64)?;
```

## Available Distributions

### Basic Distributions

- `random()` - Uniform random numbers in [0.0, 1.0)
- `randint(low, high)` - Random integers
- `uniform(low, high)` - Uniform distribution over [low, high)

### Normal Distributions

- `normal(mean, std)` - Normal (Gaussian) distribution
- `standard_normal()` - Standard normal distribution (μ=0, σ=1)

### Discrete Distributions

- `binomial(n, p)` - Binomial distribution
- `poisson(lam)` - Poisson distribution
- `bernoulli(p)` - Bernoulli distribution
- `geometric(p)` - Geometric distribution
- `hypergeometric(ngood, nbad, nsample)` - Hypergeometric distribution
- `negative_binomial(n, p)` - Negative binomial distribution
- `logseries(p)` - Logarithmic series distribution

### Continuous Distributions

- `exponential(scale)` - Exponential distribution
- `gamma(shape, scale)` - Gamma distribution
- `beta(a, b)` - Beta distribution
- `chisquare(df)` - Chi-square distribution
- `f(dfnum, dfden)` - F-distribution
- `wald(mean, scale)` - Wald (inverse Gaussian) distribution
- `weibull(a)` - Weibull distribution
- `pareto(a)` - Pareto distribution
- `zipf(a)` - Zipf distribution

### Other Distributions

- `lognormal(mean, sigma)` - Log-normal distribution
- `logistic(loc, scale)` - Logistic distribution
- `gumbel(loc, scale)` - Gumbel distribution
- `triangular(left, mode, right)` - Triangular distribution
- `vonmises(mu, kappa)` - von Mises distribution
- `standard_cauchy()` - Standard Cauchy distribution
- `standard_exponential()` - Standard exponential distribution
- `standard_gamma(shape)` - Standard gamma distribution
- `power(a)` - Power distribution
- `multinomial(n, pvals)` - Multinomial distribution
- `dirichlet(alpha)` - Dirichlet distribution

## Usage Examples

### Basic Random Number Generation

```rust
use rust_numpy::random;
use rust_numpy::Dtype;

// Create a generator
let mut rng = random::default_rng();

// Generate random floats
let floats = rng.random::<f64>(&[3, 4], Dtype::Float64)?;

// Generate random integers
let integers = rng.randint::<i32>(0, 100, &[2, 3])?;
```

### Reproducible Results

```rust
use rust_numpy::random;

// Use seeded generator for reproducible results
let mut rng1 = random::default_rng_with_seed(42);
let mut rng2 = random::default_rng_with_seed(42);

// These will produce the same sequences
let arr1 = rng1.random::<f64>(&[2, 2], Dtype::Float64)?;
let arr2 = rng2.random::<f64>(&[2, 2], Dtype::Float64)?;
```

### Working with Different Distributions

```rust
use rust_numpy::random;

let mut rng = random::default_rng();

// Normal distribution
let normal = rng.normal::<f64>(0.0, 1.0, &[1000])?;

// Binomial distribution
let binomial = rng.binomial::<f64>(10, 0.5, &[1000])?;

// Exponential distribution
let exponential = rng.exponential::<f64>(1.0, &[1000])?;
```

### Module-Level Functions

```rust
use rust_numpy::random;
use rust_numpy::Dtype;

// Quick random generation without creating a generator
let quick_random = random::random::<f64>(&[2, 3], Dtype::Float64)?;
let quick_ints = random::randint::<i32>(0, 10, &[2, 2])?;
let quick_normal = random::normal::<f64>(0.0, 1.0, &[2, 2])?;
```

## Migration from Legacy to Modern API

### Old Code (Legacy)

```rust
use rust_numpy::random;
use rust_numpy::Dtype;

// Legacy approach
random::seed(42);
let arr = random::legacy_random::<f64>(&[2, 3], Dtype::Float64)?;
let ints = random::legacy_randint::<i32>(0, 10, &[2, 2])?;
```

### New Code (Modern)

```rust
use rust_numpy::random;
use rust_numpy::Dtype;

// Modern approach (recommended)
let mut rng = random::default_rng_with_seed(42);
let arr = rng.random::<f64>(&[2, 3], Dtype::Float64)?;
let ints = rng.randint::<i32>(0, 10, &[2, 2])?;

// Or using module-level functions
let arr = random::random::<f64>(&[2, 3], Dtype::Float64)?;
let ints = random::randint::<i32>(0, 10, &[2, 2])?;
```

## Thread Safety

- Each thread has its own thread-local default Generator
- Generator instances are not thread-safe and should not be shared between threads
- Create separate Generator instances for each thread when needed

```rust
use rust_numpy::random;
use std::thread;

// Each thread gets its own default Generator
let handle = thread::spawn(|| {
    let mut rng = random::default_rng();
    rng.random::<f64>(&[2, 2], rust_numpy::Dtype::Float64)
});

let result = handle.join().unwrap()?;
```

## Performance Considerations

- Modern Generator API has minimal overhead compared to legacy API
- PCG64 provides good performance and statistical quality
- Thread-local generators avoid synchronization overhead
- For high-performance scenarios, consider reusing Generator instances

## Compatibility with NumPy

This implementation matches NumPy's random module structure:

| NumPy Function            | Rust Function           | Notes                        |
|---------------------------|-------------------------|------------------------------|
| `np.random.default_rng()` | `random::default_rng()` | Creates Generator with PCG64 |
| `Generator.random()`      | `Generator::random()`   | Same interface               |
| `Generator.randint()`     | `Generator::randint()`  | Same interface               |
| `Generator.normal()`      | `Generator::normal()`   | Same interface               |
| `np.random.random()`      | `random::random()`      | Module-level convenience     |
| `RandomState`             | `RandomState`           | Legacy compatibility         |

## Testing

The module includes comprehensive tests covering:

- Generator creation and functionality
- All distribution functions
- Thread-local behavior
- Legacy API compatibility
- Modern/legacy API separation
- BitGenerator functionality

Run tests with:
```bash
cargo test random
```

## Best Practices

1. **Use `default_rng()` for new code** - It's the recommended modern approach
2. **Use seeded generators for reproducible results** - `default_rng_with_seed(seed)`
3. **Prefer module-level functions for simple use cases** - They're convenient and efficient
4. **Create separate Generator instances for different threads** - Avoid sharing generators
5. **Use legacy API only for existing code** - New code should use modern API

## Error Handling

All random number generation functions return `Result<Array<T>, NumPyError>` to handle potential errors such as:

- Invalid parameters for distributions
- Memory allocation failures
- Invalid dtype specifications

Always handle the result appropriately:

```rust
use rust_numpy::random;

let result = random::random::<f64>(&[2, 3], rust_numpy::Dtype::Float64);
match result {
    Ok(arr) => {
        // Use the generated array
        println!("Generated array shape: {:?}", arr.shape());
    }
    Err(e) => {
        // Handle the error
        eprintln!("Error generating random array: {}", e);
    }
}
```
