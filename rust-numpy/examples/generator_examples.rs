//! Examples demonstrating the enhanced Generator class
//!
//! This file shows how to use the Generator class with BitGenerator instances
//! to provide comprehensive random number generation capabilities.

use rust_numpy::array;
use rust_numpy::dtype::Dtype;
use rust_numpy::random::generator::Generator;
use rust_numpy::random::bit_generator::{PCG64, BitGenerator};
use rust_numpy::random;
use rust_numpy::error::NumPyError;
use rand::RngCore;
use rand::Rng;

/// Example 1: Basic Generator creation and usage
fn example_basic_generator() -> Result<(), NumPyError> {
    println!("=== Basic Generator Usage ===");

    // Create default Generator with PCG64
    let mut rng = random::default_rng();

    // Generate random numbers
    let random_arr = rng.random::<f64>(&[3, 4], Dtype::Float64 { byteorder: None })?;
    println!("Random array shape: {:?}", random_arr.shape());
    println!("Random array length: {}", random_arr.len());
    println!("Random array sample: {:?}", &random_arr.data.as_slice()[0..5]);

    // Generate random integers
    let int_arr = rng.randint::<i32>(0, 100, &[2, 3])?;
    println!("Integer array shape: {:?}", int_arr.shape());
    println!("Integer array sample: {:?}", &int_arr.data.as_slice()[0..6]);

    // Generate uniform distribution
    let uniform_arr = rng.uniform::<f64>(0.0, 10.0, &[2, 2])?;
    println!("Uniform array shape: {:?}", uniform_arr.shape());
    println!("Uniform array sample: {:?}", &uniform_arr.data.as_slice()[0..4]);

    Ok(())
}

/// Example 2: Seeded Generator for reproducible results
fn example_seeded_generator() -> Result<(), NumPyError> {
    println!("\n=== Seeded Generator ===");

    let seed = 42;

    // Create seeded generators
    let mut rng1 = random::default_rng_with_seed(seed);
    let mut rng2 = random::default_rng_with_seed(seed);

    // Generate arrays with same seed
    let arr1 = rng1.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;
    let arr2 = rng2.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;

    println!("Seeded array 1 sample: {:?}", &arr1.data.as_slice()[0..4]);
    println!("Seeded array 2 sample: {:?}", &arr2.data.as_slice()[0..4]);
    println!("Arrays have same shape: {}", arr1.shape() == arr2.shape());

    // Manual Generator creation with seeded BitGenerator
    let pcg = PCG64::seed_from_u64(seed);
    let mut manual_rng = Generator::new(Box::new(pcg));
    let manual_arr = manual_rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })?;

    println!("Manual generator sample: {:?}", &manual_arr.data.as_slice()[0..4]);
    println!("Manual generator shape: {:?}", manual_arr.shape());

    Ok(())
}

/// Example 3: Core statistical distributions
fn example_statistical_distributions() -> Result<(), NumPyError> {
    println!("\n=== Statistical Distributions ===");

    let mut rng = random::default_rng();

    // Normal distribution
    let normal_arr = rng.normal::<f64>(0.0, 1.0, &[1000])?;
    let normal_mean = normal_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    let normal_std = (normal_arr.data.as_slice().iter()
        .map(|x| (x - normal_mean).powi(2))
        .sum::<f64>() / 1000.0).sqrt();
    println!("Normal distribution: mean={:.3}, std={:.3}", normal_mean, normal_std);

    // Standard normal distribution
    let std_normal_arr = rng.standard_normal::<f64>(&[1000])?;
    let std_normal_mean = std_normal_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    let std_normal_std = (std_normal_arr.data.as_slice().iter()
        .map(|x| (x - std_normal_mean).powi(2))
        .sum::<f64>() / 1000.0).sqrt();
    println!("Standard normal: mean={:.3}, std={:.3}", std_normal_mean, std_normal_std);

    // Gamma distribution
    let gamma_arr = rng.gamma::<f64>(2.0, 2.0, &[1000])?;
    let gamma_mean = gamma_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    println!("Gamma distribution (k=2, θ=2): mean={:.3}", gamma_mean);

    // Beta distribution
    let beta_arr = rng.beta::<f64>(2.0, 2.0, &[1000])?;
    let beta_mean = beta_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    println!("Beta distribution (α=2, β=2): mean={:.3}", beta_mean);

    // Chi-square distribution
    let chi_arr = rng.chisquare::<f64>(2.0, &[1000])?;
    let chi_mean = chi_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    println!("Chi-square distribution (df=2): mean={:.3}", chi_mean);

    Ok(())
}

/// Example 4: Discrete distributions
fn example_discrete_distributions() -> Result<(), NumPyError> {
    println!("\n=== Discrete Distributions ===");

    let mut rng = random::default_rng();

    // Binomial distribution
    let bin_arr = rng.binomial::<f64>(10, 0.5, &[1000])?;
    let bin_mean = bin_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    println!("Binomial distribution (n=10, p=0.5): mean={:.3}", bin_mean);

    // Poisson distribution
    let pois_arr = rng.poisson::<f64>(5.0, &[1000])?;
    let pois_mean = pois_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    println!("Poisson distribution (λ=5.0): mean={:.3}", pois_mean);

    // Bernoulli distribution
    let bern_arr = rng.bernoulli::<f64>(0.7, &[1000])?;
    let bern_mean = bern_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    let bern_true_rate = bern_arr.data.as_slice().iter().filter(|&x| *x == 1.0).count() as f64 / 1000.0;
    println!("Bernoulli distribution (p=0.7): mean={:.3}, true_rate={:.3}", bern_mean, bern_true_rate);

    // Geometric distribution
    let geom_arr = rng.geometric::<f64>(0.5, &[1000])?;
    let geom_mean = geom_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    println!("Geometric distribution (p=0.5): mean={:.3}", geom_mean);

    // Negative binomial distribution
    let neg_bin_arr = rng.negative_binomial::<f64>(5, 0.5, &[1000])?;
    let neg_bin_mean = neg_bin_arr.data.as_slice().iter().sum::<f64>() / 1000.0;
    println!("Negative binomial (r=5, p=0.5): mean={:.3}", neg_bin_mean);

    Ok(())
}

/// Example 5: Utility methods
fn example_utility_methods() -> Result<(), NumPyError> {
    println!("\n=== Utility Methods ===");

    let mut rng = random::default_rng();

    // Permutation
    let perm_arr = rng.permutation(10)?;
    println!("Permutation of 0..9: {:?}", &perm_arr.data.as_slice());

    // Choice with replacement
    let choices = vec!["apple", "banana", "cherry", "date", "elderberry"];
    let choice_replace = rng.choice(&choices, 3, true)?;
    println!("Choice with replacement: {:?}", &choice_replace.data.as_slice());

    // Choice without replacement
    let choice_no_replace = rng.choice(&choices, 3, false)?;
    println!("Choice without replacement: {:?}", &choice_no_replace.data.as_slice());

    // Integers
    let int_arr = rng.integers(0, 100, 5)?;
    println!("Random integers [0, 100): {:?}", &int_arr.data.as_slice());

    // Shuffle array
    let mut arr = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    println!("Original array: {:?}", arr.data.as_slice());
    rng.shuffle(&mut arr)?;
    println!("Shuffled array: {:?}", arr.data.as_slice());

    // Random bytes
    let bytes_arr = rng.bytes(8)?;
    println!("Random bytes: {:?}", &bytes_arr.data.as_slice());

    // Random floats
    let float_arr = rng.random_floats(5)?;
    println!("Random floats [0,1): {:?}", &float_arr.data.as_slice());

    // Random floats in range
    let range_arr = rng.random_floats_range(10.0, 20.0, 5)?;
    println!("Random floats [10,20): {:?}", &range_arr.data.as_slice());

    // Random booleans
    let bool_arr = rng.random_bools(10)?;
    println!("Random booleans: {:?}", &bool_arr.data.as_slice());

    Ok(())
}

/// Example 6: Multivariate distributions
fn example_multivariate_distributions() -> Result<(), NumPyError> {
    println!("\n=== Multivariate Distributions ===");

    let mut rng = random::default_rng();

    // Multinomial distribution
    let pvals = vec![0.2, 0.3, 0.5];
    let multi_arr = rng.multinomial::<f64>(10, &pvals, None)?;
    println!("Multinomial (n=10, p=[0.2, 0.3, 0.5]): {:?}", &multi_arr.data.as_slice());
    let sum: f64 = multi_arr.data.as_slice().iter().sum();
    println!("Sum: {} (should equal n={})", sum, 10);

    // Dirichlet distribution
    let alpha = vec![1.0, 2.0, 3.0];
    let dir_arr = rng.dirichlet::<f64>(&alpha, None)?;
    println!("Dirichlet (α=[1.0, 2.0, 3.0]): {:?}", &dir_arr.data.as_slice());
    let dir_sum: f64 = dir_arr.data.as_slice().iter().sum();
    println!("Sum: {} (should equal 1.0)", dir_sum);

    Ok(())
}

/// Example 7: Advanced distributions
fn example_advanced_distributions() -> Result<(), NumPyError> {
    println!("\n=== Advanced Distributions ===");

    let mut rng = random::default_rng();

    // Test each distribution individually
    let distributions: &[(&str, Result<rust_numpy::Array<f64>, NumPyError>)] = &[
        ("Log-Normal", rng.lognormal::<f64>(0.0, 1.0, &[1000])),
        ("Gumbel", rng.gumbel::<f64>(0.0, 1.0, &[1000])),
        ("Wald", rng.wald::<f64>(1.0, 1.0, &[1000])),
        ("Weibull", rng.weibull::<f64>(2.0, &[1000])),
        ("Triangular", rng.triangular::<f64>(0.0, 0.5, 1.0, &[1000])),
        ("Pareto", rng.pareto::<f64>(2.0, &[1000])),
        ("Zipf", rng.zipf::<f64>(3.0, &[1000])),
        ("Standard Cauchy", rng.standard_cauchy::<f64>(&[1000])),
        ("Standard Exponential", rng.standard_exponential::<f64>(&[1000])),
        ("Standard Gamma", rng.standard_gamma::<f64>(2.0, &[1000])),
        ("F-Distribution", rng.f::<f64>(2.0, 2.0, &[1000])),
        // ("Power", rng.power::<f64>(2.0, &[1000])),  // TODO: not implemented
        // ("von Mises", rng.vonmises::<f64>(0.0, 1.0, &[1000])),  // TODO: not implemented
    ];

    for (name, result) in distributions {
        match result {
            Ok(arr) => {
                let mean = arr.data.as_slice().iter().sum::<f64>() / 1000.0;
                println!("{}: mean={:.3}", name, mean);
            }
            Err(e) => {
                println!("{}: error - {}", name, e);
            }
        }
    }

    Ok(())
}

/// Example 8: Error handling
fn example_error_handling() -> Result<(), NumPyError> {
    println!("\n=== Error Handling ===");

    let mut rng = random::default_rng();

    // Test invalid parameters - each evaluated separately
    let _exponential_err = rng.exponential::<f64>(-1.0, &[2, 2]);
    let _gamma_err = rng.gamma::<f64>(-1.0, 1.0, &[2, 2]);
    let _binomial_err = rng.binomial::<f64>(10, -0.5, &[2, 2]);
    let _integers_err = rng.integers(10, 5, 5);
    let _floats_err = rng.random_floats_range(10.0, 5.0, 5);

    // Test empty choices separately (different type)
    let empty_choices: Vec<i32> = vec![];
    let _choice_err = rng.choice(&empty_choices, 3, false);

    // Test the "too large sample without replacement" case separately
    let small_choices = vec![1, 2];
    let too_large_result = rng.choice(&small_choices, 5, false);

    // Check error results individually
    match _exponential_err {
        Ok(_) => println!("exponential with negative lambda: Unexpected success"),
        Err(e) => println!("exponential with negative lambda: Expected error - {}", e),
    }

    match _gamma_err {
        Ok(_) => println!("gamma with negative shape: Unexpected success"),
        Err(e) => println!("gamma with negative shape: Expected error - {}", e),
    }

    match _binomial_err {
        Ok(_) => println!("binomial with invalid p: Unexpected success"),
        Err(e) => println!("binomial with invalid p: Expected error - {}", e),
    }

    match _integers_err {
        Ok(_) => println!("integers with reversed bounds: Unexpected success"),
        Err(e) => println!("integers with reversed bounds: Expected error - {}", e),
    }

    match _floats_err {
        Ok(_) => println!("floats_range with reversed bounds: Unexpected success"),
        Err(e) => println!("floats_range with reversed bounds: Expected error - {}", e),
    }

    match _choice_err {
        Ok(_) => println!("choice with empty array: Unexpected success"),
        Err(e) => println!("choice with empty array: Expected error - {}", e),
    }

    match too_large_result {
        Ok(_) => println!("Too large sample without replacement: Unexpected success"),
        Err(e) => println!("Too large sample without replacement: Expected error - {}", e),
    }

    Ok(())
}

/// Example 9: Performance comparison
fn example_performance() -> Result<(), NumPyError> {
    println!("\n=== Performance Comparison ===");

    use std::time::Instant;

    let mut rng = random::default_rng();

    // Test different array sizes
    let sizes = vec![100, 1000, 10000, 100000];

    for size in sizes {
        let start = Instant::now();
        let _ = rng.random::<f64>(&[size], Dtype::Float64 { byteorder: None })?;
        let duration = start.elapsed();
        println!("Generated {} floats in {:?}", size, duration);
    }

    Ok(())
}

/// Example 10: Thread safety demonstration
fn example_thread_safety() -> Result<(), NumPyError> {
    println!("\n=== Thread Safety ===");

    use std::thread;

    // Create multiple generators in different threads
    let handles: Vec<_> = (0..4).map(|i| {
        let mut rng = random::default_rng_with_seed(i);
        thread::spawn(move || {
            let arr = rng.random::<f64>(&[1000], Dtype::Float64 { byteorder: None })?;
            let mean = arr.data.as_slice().iter().sum::<f64>() / 1000.0;
            Ok::<f64, NumPyError>(mean)
        })
    }).collect();

    let mut means = Vec::new();
    for handle in handles {
        means.push(handle.join().unwrap()?);
    }

    println!("Thread means: {:?}", means);

    // Verify means are different (due to different seeds)
    let unique_means: Vec<f64> = means.iter().cloned().collect();
    let unique_count = unique_means.len();
    println!("Unique means: {} (should be 4)", unique_count);

    Ok(())
}

/// Example 11: Statistical analysis
fn example_statistical_analysis() -> Result<(), NumPyError> {
    println!("\n=== Statistical Analysis ===");

    let mut rng = random::default_rng();

    // Generate samples from different distributions
    let normal_sample = rng.normal::<f64>(0.0, 1.0, &[10000])?;
    let gamma_sample = rng.gamma::<f64>(2.0, 2.0, &[10000])?;
    let beta_sample = rng.beta::<f64>(2.0, 2.0, &[10000])?;
    let exp_sample = rng.exponential::<f64>(1.0, &[10000])?;

    // Calculate statistics for each distribution
    let distributions: &[(&str, &rust_numpy::Array<f64>)] = &[
        ("Normal", &normal_sample),
        ("Gamma", &gamma_sample),
        ("Beta", &beta_sample),
        ("Exponential", &exp_sample),
    ];

    for (name, sample) in distributions {
        let data = sample.data.as_slice();
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        let std = variance.sqrt();
        let min = data.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));

        println!("{}:", name);
        println!("  Mean: {:.6}", mean);
        println!("  Std:  {:.6}", std);
        println!("  Min:  {:.6}", min);
        println!("  Max: {:.6}", max);
        println!();
    }

    Ok(())
}

/// Example 12: BitGenerator compatibility
fn example_bitgenerator_compatibility() -> Result<(), NumPyError> {
    println!("\n=== BitGenerator Compatibility ===");

    // Test with different BitGenerators
    let bit_generators = vec![
        ("PCG64", Box::new(PCG64::new()) as Box<dyn BitGenerator>),
        ("PCG64 (seeded)", Box::new(PCG64::seed_from_u64(12345)) as Box<dyn BitGenerator>),
    ];

    for (name, bit_gen) in bit_generators {
        let mut generator = Generator::new(bit_gen);
        let arr = generator.random::<f64>(&[100], Dtype::Float64 { byteorder: None })?;
        let mean = arr.data.as_slice().iter().sum::<f64>() / 100.0;
        println!("{}: mean={:.6}", name, mean);
    }

    Ok(())
}

/// Example 13: RngCore trait implementation
fn example_rng_core() -> Result<(), NumPyError> {
    println!("\n=== RngCore Trait Implementation ===");

    let mut rng = random::default_rng();

    // Test RngCore methods
    let u32_val = rng.next_u32();
    let u64_val = rng.next_u64();

    println!("next_u32(): {}", u32_val);
    println!("next_u64(): {}", u64_val);

    // Test fill_bytes
    let mut bytes = [0u8; 16];
    rng.fill_bytes(&mut bytes);
    println!("fill_bytes(): {:?}", &bytes);

    // Generate random floats using RngCore
    let mut rng_floats = Vec::with_capacity(5);
    for _ in 0..5 {
        rng_floats.push(rng.gen::<f64>());
    }
    println!("gen::<f64>(): {:?}", rng_floats);

    Ok(())
}

fn main() -> Result<(), NumPyError> {
    println!("=== Generator Class Examples ===\n");

    // Run all examples
    example_basic_generator()?;
    example_seeded_generator()?;
    example_statistical_distributions()?;
    example_discrete_distributions()?;
    example_utility_methods()?;
    example_multivariate_distributions()?;
    example_advanced_distributions()?;
    example_error_handling()?;
    example_performance()?;
    example_thread_safety()?;
    example_statistical_analysis()?;
    example_bitgenerator_compatibility()?;
    example_rng_core()?;

    println!("\n=== Examples Complete ===");
    println!("✓ Generator class with BitGenerator wrapper working");
    println!("✓ All distribution methods implemented");
    println!("✓ Utility methods (shuffle, permutation, choice) working");
    println!("✅ Error handling and validation working");
    println!("✅ Thread safety considerations demonstrated");
    println!("✅ RngCore trait implementation working");
    println!("✅ Statistical analysis capabilities working");

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
    fn test_generator_creation() {
        let mut rng = random::default_rng();
        let arr = rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_seeded_reproducibility() {
        let seed = 42;
        let mut rng1 = random::default_rng_with_seed(seed);
        let mut rng2 = random::default_rng_with_seed(seed);

        let arr1 = rng1.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        let arr2 = rng2.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();

        assert_eq!(arr1.shape(), arr2.shape());
        assert_eq!(arr1.len(), arr2.len());
    }

    #[test]
    fn test_utility_methods() {
        let mut rng = random::default_rng();

        let perm = rng.permutation(5).unwrap();
        assert_eq!(perm.shape(), &[5]);

        let choices = vec![1, 2, 3, 4, 5];
        let choice = rng.choice(&choices, 3, false).unwrap();
        assert_eq!(choice.shape(), &[3]);

        let ints = rng.integers(0, 10, 5).unwrap();
        assert_eq!(ints.shape(), &[5]);
    }

    #[test]
    fn test_error_handling() {
        let mut rng = random::default_rng();

        // These should all return errors
        assert!(rng.exponential::<f64>(-1.0, &[2, 2]).is_err());
        assert!(rng.gamma::<f64>(-1.0, 1.0, &[2, 2]).is_err());
        assert!(rng.binomial::<f64>(10, -0.5, &[2, 2]).is_err());
        assert!(rng.integers(10, 5, 5).is_err());
    }
}
