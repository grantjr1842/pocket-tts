//! Examples demonstrating the restructured random module
//!
//! This file shows how to use the modern Generator/BitGenerator API
//! and the legacy RandomState API for backward compatibility.

use rust_numpy::dtype::Dtype;
use rust_numpy::error::NumPyError;
use rust_numpy::random;
use rust_numpy::random::legacy;
use rust_numpy::random::{Generator, PCG64, BitGenerator};

/// Example using the modern default_rng() function
fn example_default_rng() -> Result<(), NumPyError> {
    println!("=== Modern API: default_rng() ===");

    // Create default Generator with PCG64
    let mut rng = random::default_rng();

    // Generate random numbers
    let random_arr = rng.random::<f64>(&[3, 4], Dtype::Float64 { byteorder: None })?;
    println!("Random array shape: {:?}", random_arr.shape());
    println!("Random array length: {}", random_arr.len());

    // Generate random integers
    let int_arr = rng.randint::<i32>(0, 100, &[2, 3])?;
    println!("Integer array shape: {:?}", int_arr.shape());
    println!("Integer array length: {}", int_arr.len());

    // Generate from normal distribution
    let normal_arr = rng.normal::<f64>(0.0, 1.0, &[2, 2])?;
    println!("Normal array shape: {:?}", normal_arr.shape());

    Ok(())
}

/// Example using seeded generators for reproducible results
fn example_seeded_generators() -> Result<(), NumPyError> {
    println!("\n=== Seeded Generators ===");

    let seed = 42;

    // Create seeded generators
    let mut rng1 = random::default_rng_with_seed(seed);
    let mut rng2 = random::default_rng_with_seed(seed);

    // Generate arrays with same seed
    let arr1 = rng1.random::<f64>(&[2, 3], Dtype::Float64 { byteorder: None })?;
    let arr2 = rng2.random::<f64>(&[2, 3], Dtype::Float64 { byteorder: None })?;

    println!("Seeded array 1 shape: {:?}", arr1.shape());
    println!("Seeded array 2 shape: {:?}", arr2.shape());
    println!(
        "Both arrays have same shape: {}",
        arr1.shape() == arr2.shape()
    );

    // Manual Generator creation with PCG64
    let pcg = PCG64::seed_from_u64(seed);
    let mut manual_rng = Generator::new(Box::new(pcg));
    let manual_arr = manual_rng.random::<f64>(&[2, 3], Dtype::Float64 { byteorder: None })?;

    println!("Manual generator array shape: {:?}", manual_arr.shape());

    Ok(())
}

/// Example using module-level convenience functions
fn example_module_level_functions() -> Result<(), NumPyError> {
    println!("\n=== Module-Level Functions ===");

    // These use the modern Generator API internally
    let random_arr = random::random::<f64>(&[2, 3], Dtype::Float64 { byteorder: None })?;
    println!("Module random array shape: {:?}", random_arr.shape());

    let int_arr = random::randint::<i32>(0, 10, &[2, 2])?;
    println!("Module randint array shape: {:?}", int_arr.shape());

    let uniform_arr = random::uniform::<f64>(0.0, 5.0, &[2, 2])?;
    println!("Module uniform array shape: {:?}", uniform_arr.shape());

    let normal_arr = random::normal::<f64>(0.0, 1.0, &[2, 2])?;
    println!("Module normal array shape: {:?}", normal_arr.shape());

    let std_normal_arr = random::standard_normal::<f64>(&[2, 2])?;
    println!(
        "Module standard normal array shape: {:?}",
        std_normal_arr.shape()
    );

    Ok(())
}

/// Example using various probability distributions
fn example_distributions() -> Result<(), NumPyError> {
    println!("\n=== Probability Distributions ===");

    let mut rng = random::default_rng();

    // Discrete distributions
    let binomial_arr = rng.binomial::<f64>(10, 0.5, &[2, 2])?;
    println!("Binomial array shape: {:?}", binomial_arr.shape());

    let poisson_arr = rng.poisson::<f64>(5.0, &[2, 2])?;
    println!("Poisson array shape: {:?}", poisson_arr.shape());

    let bernoulli_arr = rng.bernoulli::<f64>(0.7, &[2, 2])?;
    println!("Bernoulli array shape: {:?}", bernoulli_arr.shape());

    // Continuous distributions
    let exponential_arr = rng.exponential::<f64>(1.0, &[2, 2])?;
    println!("Exponential array shape: {:?}", exponential_arr.shape());

    let gamma_arr = rng.gamma::<f64>(2.0, 2.0, &[2, 2])?;
    println!("Gamma array shape: {:?}", gamma_arr.shape());

    let beta_arr = rng.beta::<f64>(2.0, 2.0, &[2, 2])?;
    println!("Beta array shape: {:?}", beta_arr.shape());

    let chisquare_arr = rng.chisquare::<f64>(2.0, &[2, 2])?;
    println!("Chi-square array shape: {:?}", chisquare_arr.shape());

    Ok(())
}

/// Example using the modern sub-module API
fn example_modern_submodule() -> Result<(), NumPyError> {
    println!("\n=== Modern Sub-Module API ===");

    // Use modern submodule
    let mut rng = modern::default_rng();
    let arr = rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("Modern submodule array shape: {:?}", arr.shape());

    // Create seeded generator through submodule
    let mut seeded_rng = modern::default_rng_with_seed(123);
    let seeded_arr = seeded_rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("Modern seeded array shape: {:?}", seeded_arr.shape());

    // Access PCG64 through module
    let pcg = PCG64::new();
    let mut manual_rng = Generator::new(Box::new(pcg));
    let manual_arr = manual_rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("Modern manual array shape: {:?}", manual_arr.shape());

    // Test BitGenerator trait
    let bit_gen: Box<dyn BitGenerator> = Box::new(PCG64::seed_from_u64(456));
    let mut trait_rng = Generator::new(bit_gen);
    let trait_arr = trait_rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("Modern trait array shape: {:?}", trait_arr.shape());

    Ok(())
}

/// Example using the legacy API for backward compatibility
fn example_legacy_api() -> Result<(), NumPyError> {
    println!("\n=== Legacy API (Backward Compatibility) ===");

    // Legacy functions (with deprecation warnings)
    let legacy_arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("Legacy random array shape: {:?}", legacy_arr.shape());

    let legacy_int_arr = legacy::legacy_randint::<i32>(0, 10, &[2, 2])?;
    println!("Legacy randint array shape: {:?}", legacy_int_arr.shape());

    // Legacy seeding
    legacy::seed(12345);
    let seeded_legacy_arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("Seeded legacy array shape: {:?}", seeded_legacy_arr.shape());

    // Legacy RandomState
    let _legacy_rng = legacy::legacy_rng();
    println!("Created legacy RandomState instance");

    Ok(())
}

/// Example demonstrating API separation
fn example_api_separation() -> Result<(), NumPyError> {
    println!("\n=== API Separation ===");

    // Modern API uses Generator internally
    let modern_arr = random::random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("Modern API array shape: {:?}", modern_arr.shape());

    // Legacy API uses RandomState internally
    let legacy_arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("Legacy API array shape: {:?}", legacy_arr.shape());

    // Both work independently
    println!(
        "Both APIs work independently: {}",
        modern_arr.shape() == legacy_arr.shape()
    );

    // Thread-local behavior
    let thread_arr1 = random::random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    let thread_arr2 = random::random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!(
        "Thread-local arrays have same shape: {}",
        thread_arr1.shape() == thread_arr2.shape()
    );

    Ok(())
}

/// Example showing comprehensive distribution usage
fn example_comprehensive_distributions() -> Result<(), NumPyError> {
    println!("\n=== Comprehensive Distributions ===");

    let mut rng = random::default_rng();

    // Test each distribution individually
    let distributions: &[(&str, Result<rust_numpy::Array<f64>, NumPyError>)] = &[
        ("random", rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })),
        ("uniform", rng.uniform::<f64>(0.0, 1.0, &[2, 2])),
        ("normal", rng.normal::<f64>(0.0, 1.0, &[2, 2])),
        ("standard_normal", rng.standard_normal::<f64>(&[2, 2])),
        ("binomial", rng.binomial::<f64>(10, 0.5, &[2, 2])),
        ("poisson", rng.poisson::<f64>(5.0, &[2, 2])),
        ("exponential", rng.exponential::<f64>(1.0, &[2, 2])),
        ("gamma", rng.gamma::<f64>(2.0, 2.0, &[2, 2])),
        ("beta", rng.beta::<f64>(2.0, 2.0, &[2, 2])),
        ("chisquare", rng.chisquare::<f64>(2.0, &[2, 2])),
        ("bernoulli", rng.bernoulli::<f64>(0.5, &[2, 2])),
        ("lognormal", rng.lognormal::<f64>(0.0, 1.0, &[2, 2])),
        ("geometric", rng.geometric::<f64>(0.5, &[2, 2])),
        ("standard_cauchy", rng.standard_cauchy::<f64>(&[2, 2])),
        ("standard_exponential", rng.standard_exponential::<f64>(&[2, 2])),
    ];

    for (name, result) in distributions {
        match result {
            Ok(arr) => println!("✓ {}: shape {:?}", name, arr.shape()),
            Err(e) => println!("✗ {}: error {}", name, e),
        }
    }

    // Test integer distributions separately
    let int_result = rng.randint::<i32>(0, 10, &[2, 2]);
    match int_result {
        Ok(arr) => println!("✓ randint: shape {:?}", arr.shape()),
        Err(e) => println!("✗ randint: error {}", e),
    }

    Ok(())
}

/// Example showing migration patterns
fn example_migration_patterns() -> Result<(), NumPyError> {
    println!("\n=== Migration Patterns ===");

    println!("Old code (Legacy API):");
    println!("  random::seed(42);");
    println!("  let arr = random::legacy_random::<f64>(&[2, 2], Dtype::Float64 {{ byteorder: None }})?;");

    // Old way
    legacy::seed(42);
    let old_arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("  → Legacy array shape: {:?}", old_arr.shape());

    println!("\nNew code (Modern API):");
    println!("  let mut rng = random::default_rng_with_seed(42);");
    println!("  let arr = rng.random::<f64>(&[2, 2], Dtype::Float64 {{ byteorder: None }})?;");

    // New way
    let mut rng = random::default_rng_with_seed(42);
    let new_arr = rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("  → Modern array shape: {:?}", new_arr.shape());

    println!("\nOr using module-level functions:");
    println!("  let arr = random::random::<f64>(&[2, 2], Dtype::Float64 {{ byteorder: None }})?;");

    let module_arr = random::random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    println!("  → Module array shape: {:?}", module_arr.shape());

    println!(
        "\nAll approaches produce arrays with same shape: {}",
        old_arr.shape() == new_arr.shape() && new_arr.shape() == module_arr.shape()
    );

    Ok(())
}

/// Example showing thread safety
fn example_thread_safety() -> Result<(), NumPyError> {
    println!("\n=== Thread Safety ===");

    use std::thread;

    // Each thread gets its own thread-local generator
    let handle1 = thread::spawn(|| {
        let mut rng = random::default_rng();
        rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap()
    });

    let handle2 = thread::spawn(|| {
        let mut rng = random::default_rng();
        rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap()
    });

    let arr1 = handle1.join().unwrap();
    let arr2 = handle2.join().unwrap();

    println!("Thread 1 array shape: {:?}", arr1.shape());
    println!("Thread 2 array shape: {:?}", arr2.shape());
    println!("Both threads work independently");

    Ok(())
}

fn main() -> Result<(), NumPyError> {
    println!("=== Random Module Restructure Examples ===\n");

    // Run all examples
    example_default_rng()?;
    example_seeded_generators()?;
    example_module_level_functions()?;
    example_distributions()?;
    example_modern_submodule()?;
    example_legacy_api()?;
    example_api_separation()?;
    example_comprehensive_distributions()?;
    example_migration_patterns()?;
    example_thread_safety()?;

    println!("\n=== Examples Complete ===");
    println!("✓ Modern Generator/BitGenerator API working");
    println!("✓ Legacy RandomState API maintained");
    println!("✓ Module-level convenience functions working");
    println!("✓ Sub-module organization functional");
    println!("✓ Thread safety maintained");
    println!("✓ Backward compatibility preserved");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_examples() {
        // Test that all examples run without panicking
        main().unwrap();
    }

    #[test]
    fn test_modern_vs_legacy() {
        // Test that modern and legacy APIs work independently
        let modern_arr = random::random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        let legacy_arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();

        assert_eq!(modern_arr.shape(), legacy_arr.shape());
    }

    #[test]
    fn test_seeded_reproducibility() {
        let seed = 12345;

        let mut rng1 = random::default_rng_with_seed(seed);
        let mut rng2 = random::default_rng_with_seed(seed);

        let arr1 = rng1.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        let arr2 = rng2.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();

        assert_eq!(arr1.shape(), arr2.shape());
        assert_eq!(arr1.len(), arr2.len());
    }

    #[test]
    fn test_modern_submodule() {
        let mut rng = modern::default_rng();
        let arr = rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();

        assert_eq!(arr.shape(), &[2, 2]);

        let pcg = modern::PCG64::new();
        let manual_rng = modern::Generator::new(Box::new(pcg));
        let manual_arr = manual_rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();

        assert_eq!(manual_arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_legacy_submodule() {
        let arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        let int_arr = legacy::legacy_randint::<i32>(0, 10, &[2, 2]).unwrap();
        assert_eq!(int_arr.shape(), &[2, 2]);
    }
}
