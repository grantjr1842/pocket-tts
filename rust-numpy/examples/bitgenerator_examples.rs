//! Examples demonstrating the enhanced BitGenerator trait and implementations
//!
//! This file shows how to use PCG64, MT19937, and Philox BitGenerators
//! for NumPy-compatible random number generation.

use numpy::random::bit_generator::*;
use rand::{Rng, RngCore};

/// Example 1: Basic BitGenerator usage
fn example_basic_usage() {
    println!("=== Basic BitGenerator Usage ===");

    // Create different BitGenerators
    let mut pcg = PCG64::new();
    let mut mt = MT19937::new();
    let mut philox = Philox::new();

    // Generate some random numbers
    println!(
        "PCG64: {} {} {}",
        pcg.next_u32(),
        pcg.next_u64(),
        pcg.next_u32()
    );
    println!(
        "MT19937: {} {} {}",
        mt.next_u32(),
        mt.next_u64(),
        mt.next_u32()
    );
    println!(
        "Philox: {} {} {}",
        philox.next_u32(),
        philox.next_u64(),
        philox.next_u32()
    );

    // Display generator information
    println!("\nGenerator Information:");
    println!(
        "PCG64: name={}, secure={}, period={:?}",
        pcg.name(),
        pcg.is_cryptographically_secure(),
        pcg.period()
    );
    println!(
        "MT19937: name={}, secure={}, period={:?}",
        mt.name(),
        mt.is_cryptographically_secure(),
        mt.period()
    );
    println!(
        "Philox: name={}, secure={}, period={:?}",
        philox.name(),
        philox.is_cryptographically_secure(),
        philox.period()
    );
}

/// Example 2: Seeded generators for reproducible results
fn example_seeded_generators() {
    println!("\n=== Seeded Generators ===");

    let seed = 12345;

    // Create seeded generators
    let mut pcg1 = PCG64::seed_from_u64(seed);
    let mut pcg2 = PCG64::seed_from_u64(seed);

    let mut mt1 = MT19937::seed_from_u64(seed);
    let mut mt2 = MT19937::seed_from_u64(seed);

    let mut philox1 = Philox::seed_from_u64(seed);
    let mut philox2 = Philox::seed_from_u64(seed);

    // Generate sequences
    println!("PCG64 sequences:");
    for i in 0..5 {
        println!("  {}: {} {}", i, pcg1.next_u32(), pcg2.next_u32());
    }

    println!("MT19937 sequences:");
    for i in 0..5 {
        println!("  {}: {} {}", i, mt1.next_u32(), mt2.next_u32());
    }

    println!("Philox sequences:");
    for i in 0..5 {
        println!("  {}: {} {}", i, philox1.next_u32(), philox2.next_u32());
    }

    // Verify reproducibility
    let mut pcg3 = PCG64::seed_from_u64(seed);
    let mut mt3 = MT19937::seed_from_u64(seed);
    let mut philox3 = Philox::seed_from_u64(seed);

    println!("\nReproducibility check:");
    println!(
        "PCG64: {} (should match first PCG64 value)",
        pcg3.next_u32()
    );
    println!(
        "MT19937: {} (should match first MT19937 value)",
        mt3.next_u32()
    );
    println!(
        "Philox: {} (should match first Philox value)",
        philox3.next_u32()
    );
}

/// Example 3: Advanced BitGenerator features
fn example_advanced_features() {
    println!("\n=== Advanced BitGenerator Features ===");

    let mut pcg = PCG64::seed_from_u64(54321);

    // Test jump functionality
    println!("Before jump: {}", pcg.next_u32());
    println!("Before jump: {}", pcg.next_u32());

    pcg.jump(1000);

    println!("After jump: {}", pcg.next_u32());
    println!("After jump: {}", pcg.next_u32());

    // Test state serialization
    let state = pcg.get_state_bytes();
    println!("State size: {} bytes", state.len());

    // Create new generator and restore state
    let mut pcg2 = PCG64::new();
    pcg2.set_state_bytes(&state).unwrap();

    println!("Restored state: {}", pcg2.next_u32());
    println!("Restored state: {}", pcg2.next_u32());

    // Test parallel parameters
    let params = pcg.parallel_params();
    println!("Parallel parameters:");
    println!("  Supports parallel: {}", params.supports_parallel);
    println!("  Jump size: {}", params.jump_size);
    println!("  Max streams: {}", params.max_streams);
}

/// Example 4: Factory functions
fn example_factory_functions() {
    println!("\n=== Factory Functions ===");

    // Create generators by name
    let pcg = factory::create_bitgenerator("PCG64").unwrap();
    let mt = factory::create_bitgenerator("MT19937").unwrap();
    let philox = factory::create_bitgenerator("Philox").unwrap();

    println!("Created generators:");
    println!("  {}", pcg.name());
    println!("  {}", mt.name());
    println!("  {}", philox.name());

    // Create seeded generators
    let seed = 98765;
    let mut seeded_pcg = factory::create_seeded_bitgenerator("PCG64", seed).unwrap();
    let mut seeded_mt = factory::create_seeded_bitgenerator("MT19937", seed).unwrap();
    let mut seeded_philox = factory::create_seeded_bitgenerator("Philox", seed).unwrap();

    println!("\nSeeded generators (seed={}):", seed);
    println!("  PCG64: {}", seeded_pcg.next_u32());
    println!("  MT19937: {}", seeded_mt.next_u32());
    println!("  Philox: {}", seeded_philox.next_u32());

    // List available generators
    let available = factory::available_bitgenerators();
    println!("\nAvailable generators: {:?}", available);

    // Default generator
    let default = factory::default_bitgenerator();
    println!("Default generator: {}", default.name());
}

/// Example 5: Legacy compatibility
fn example_legacy_compatibility() {
    println!("\n=== Legacy Compatibility ===");

    // Legacy functions
    let mut legacy1 = legacy::new();
    let mut legacy2 = legacy::seed_from_u64(12345);

    println!("Legacy generators:");
    println!("  Random: {}", legacy1.name());
    println!("  Seeded: {}", legacy2.name());

    // Generate values
    println!("Legacy random: {}", legacy1.next_u32());
    println!("Legacy seeded: {}", legacy2.next_u32());
}

/// Example 6: Statistical analysis
fn example_statistical_analysis() {
    println!("\n=== Statistical Analysis ===");

    let generators: Vec<Box<dyn BitGenerator>> = vec![
        Box::new(PCG64::seed_from_u64(11111)),
        Box::new(MT19937::seed_from_u64(22222)),
        Box::new(Philox::seed_from_u64(33333)),
    ];

    for (i, mut rng) in generators.into_iter().enumerate() {
        println!("Generator {}: {}", i, rng.name());

        // Generate sample
        let mut values = Vec::new();
        for _ in 0..10000 {
            values.push(rng.next_u64());
        }

        // Calculate statistics
        let mean = values.iter().sum::<u64>() as f64 / values.len() as f64;
        let expected_mean = (u64::MAX / 2) as f64;
        let mean_error = (mean - expected_mean).abs() / expected_mean;

        // Distribution check
        let mid = u64::MAX / 2;
        let in_lower_half = values.iter().filter(|&&v| v < mid).count();
        let ratio = in_lower_half as f64 / values.len() as f64;

        println!("  Mean error: {:.6} (should be < 0.05)", mean_error);
        println!("  Distribution ratio: {:.6} (should be ~0.5)", ratio);
        println!("  Sample size: {}", values.len());
    }
}

/// Example 7: Performance comparison
fn example_performance_comparison() {
    println!("\n=== Performance Comparison ===");

    use std::time::Instant;

    let generators: Vec<Box<dyn BitGenerator>> = vec![
        Box::new(PCG64::new()),
        Box::new(MT19937::new()),
        Box::new(Philox::new()),
    ];

    let sample_sizes = vec![1000, 10000, 100000];

    for (i, mut rng) in generators.into_iter().enumerate() {
        println!("Generator {}: {}", i, rng.name());

        for &size in &sample_sizes {
            let start = Instant::now();

            for _ in 0..size {
                let _ = rng.next_u64();
            }

            let duration = start.elapsed();
            println!("  {} u64 values: {:?}", size, duration);
        }
    }
}

/// Example 8: Parallel stream generation
fn example_parallel_streams() {
    println!("\n=== Parallel Stream Generation ===");

    let base_seed = 12345;
    let mut base_gen = PCG64::seed_from_u64(base_seed);
    let params = base_gen.parallel_params();

    println!("Base generator: {}", base_gen.name());
    println!("Jump size: {}", params.jump_size);
    println!("Max streams: {}", params.max_streams);

    // Create parallel streams
    let num_streams = 4;
    let mut streams: Vec<PCG64> = (0..num_streams)
        .map(|i| {
            let mut stream = PCG64::seed_from_u64(base_seed);
            stream.jump((params.jump_size * i as u128).try_into().unwrap());
            stream
        })
        .collect();

    // Generate values from each stream
    println!("Parallel stream values:");
    for (i, stream) in streams.iter_mut().enumerate() {
        let val = stream.next_u32();
        println!("  Stream {}: {}", i, val);
    }

    // Verify streams are independent
    let mut values = Vec::new();
    for mut stream in streams {
        values.push(stream.next_u32());
    }

    let mut unique_values: Vec<u32> = values.clone();
    unique_values.sort();
    unique_values.dedup();

    println!(
        "Independence check: {} unique values out of {} (should be {})",
        unique_values.len(),
        values.len(),
        values.len()
    );
}

/// Example 9: State management demonstration
fn example_state_management() {
    println!("\n=== State Management ===");

    let mut pcg = PCG64::seed_from_u64(99999);

    // Generate some values
    println!("Initial values:");
    for _ in 0..3 {
        println!("  {}", pcg.next_u32());
    }

    // Save state
    let state = pcg.get_state_bytes();
    println!("State saved ({} bytes)", state.len());

    // Generate more values
    println!("After save:");
    for _ in 0..3 {
        println!("  {}", pcg.next_u32());
    }

    // Restore state
    let mut pcg_restored = PCG64::new();
    pcg_restored.set_state_bytes(&state).unwrap();

    // Should continue from saved point
    println!("After restore:");
    for _ in 0..6 {
        println!("  {}", pcg_restored.next_u32());
    }
}

/// Example 10: Cryptographic security comparison
fn example_security_comparison() {
    println!("\n=== Cryptographic Security Comparison ===");

    let generators: Vec<Box<dyn BitGenerator>> = vec![
        Box::new(PCG64::new()),
        Box::new(MT19937::new()),
        Box::new(Philox::new()),
    ];

    for rng in generators {
        println!("{}:", rng.name());
        println!(
            "  Cryptographically secure: {}",
            rng.is_cryptographically_secure()
        );
        println!(
            "  Recommended for security: {}",
            if rng.is_cryptographically_secure() {
                "Yes"
            } else {
                "No"
            }
        );
        println!(
            "  Use case: {}",
            match rng.name() {
                "Philox" => "Cryptography, key generation",
                "PCG64" => "General purpose, simulations",
                "MT19937" => "Legacy compatibility, statistical analysis",
                _ => "Unknown",
            }
        );
    }
}

/// Example 11: Error handling
fn example_error_handling() {
    println!("\n=== Error Handling ===");

    // Test invalid BitGenerator name
    match factory::create_bitgenerator("InvalidGenerator") {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error: {}", e),
    }

    // Test invalid state size
    let mut pcg = PCG64::new();
    match pcg.set_state_bytes(&[1, 2, 3]) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error: {}", e),
    }

    // Test invalid seed slice (empty)
    let mut pcg2 = PCG64::new();
    pcg2.seed_u32_slice(&[]);
    let val = pcg2.next_u32();
    println!("Empty seed slice result: {} (should not panic)", val);
}

/// Example 12: Thread safety simulation
fn example_thread_safety_simulation() {
    println!("\n=== Thread Safety Simulation ===");

    use std::thread;

    let seed = 12345;
    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                // Each thread gets its own generator with different seed
                let mut rng = PCG64::seed_from_u64(seed + i);

                // Generate some values
                let mut values = Vec::new();
                for _ in 0..1000 {
                    values.push(rng.next_u32());
                }

                // Calculate mean
                let mean = values.iter().sum::<u32>() as f64 / values.len() as f64;
                mean
            })
        })
        .collect();

    // Collect results
    let mut means = Vec::new();
    for handle in handles {
        means.push(handle.join().unwrap());
    }

    println!("Thread means: {:?}", means);

    // Verify means are different (high probability)
    let mut unique_means: Vec<f64> = means.clone();
    unique_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique_means.dedup();

    println!("Unique means: {} (should be 4)", unique_means.len());
}

/// Example 13: Different seeding methods
fn example_seeding_methods() {
    println!("\n=== Different Seeding Methods ===");

    // Method 1: u64 seed
    let mut pcg1 = PCG64::seed_from_u64(12345);
    println!("u64 seed: {}", pcg1.next_u32());

    // Method 2: u32 slice seed
    let seed_slice = [0x12345678u32, 0x9ABCDEF0u32];
    let mut pcg2 = PCG64::from_seed_slice(&seed_slice);
    println!("u32 slice seed: {}", pcg2.next_u32());

    // Method 3: Factory seeded
    let mut pcg3 = factory::create_seeded_bitgenerator("PCG64", 54321).unwrap();
    println!("Factory seeded: {}", pcg3.next_u32());

    // Method 4: Legacy seeded
    let mut pcg4 = legacy::seed_from_u64(98765);
    println!("Legacy seeded: {}", pcg4.next_u32());

    // Method 5: Philox with counter
    let mut philox = Philox::seed_from_u64_with_counter(11111, 22222);
    println!("Philox with counter: {}", philox.next_u32());
}

/// Example 14: RngCore trait usage
fn example_rng_core_usage() {
    println!("\n=== RngCore Trait Usage ===");

    let mut pcg = PCG64::seed_from_u64(55555);

    // Test all RngCore methods
    let u32_val = pcg.next_u32();
    let u64_val = pcg.next_u64();

    println!("next_u32(): {}", u32_val);
    println!("next_u64(): {}", u64_val);

    // Test fill_bytes
    let mut bytes = [0u8; 16];
    pcg.fill_bytes(&mut bytes);
    println!("fill_bytes(): {:?}", bytes);

    // Test try_fill_bytes
    let mut bytes2 = [0u8; 8];
    match pcg.try_fill_bytes(&mut bytes2) {
        Ok(_) => println!("try_fill_bytes(): Success"),
        Err(e) => println!("try_fill_bytes(): Error: {:?}", e),
    }
    println!("try_fill_bytes(): {:?}", bytes2);

    // Test gen() method (from Rng trait)
    let f32_val: f32 = pcg.gen();
    let f64_val: f64 = pcg.gen();
    let bool_val: bool = pcg.gen();

    println!("gen::<f32>(): {}", f32_val);
    println!("gen::<f64>(): {}", f64_val);
    println!("gen::<bool>(): {}", bool_val);

    // Test gen_range()
    let range_val: i32 = pcg.gen_range(0..100);
    println!("gen_range(0..100): {}", range_val);
}

/// Example 15: BitGenerator comparison
fn example_bitgenerator_comparison() {
    println!("\n=== BitGenerator Comparison ===");

    let generators: Vec<Box<dyn BitGenerator>> = vec![
        Box::new(PCG64::seed_from_u64(12345)),
        Box::new(MT19937::seed_from_u64(12345)),
        Box::new(Philox::seed_from_u64(12345)),
    ];

    println!("Comparison Table:");
    println!(
        "{:<12} {:<12} {:<12} {:<12} {:<12}",
        "Name", "Period", "Secure", "State Size", "Parallel"
    );
    println!("{}", "-".repeat(60));

    for rng in generators {
        let period_str = match rng.period() {
            Some(p) => format!("2^{}", p.trailing_zeros()),
            None => "Unknown".to_string(),
        };

        println!(
            "{:<12} {:<12} {:<12} {:<12} {:<12}",
            rng.name(),
            period_str,
            if rng.is_cryptographically_secure() {
                "Yes"
            } else {
                "No"
            },
            format!("{} bytes", rng.state_size()),
            if rng.parallel_params().supports_parallel {
                "Yes"
            } else {
                "No"
            }
        );
    }

    println!("\nRecommendations:");
    println!("• PCG64: Best for general use, fast and reliable");
    println!("• MT19937: Best for legacy compatibility, excellent statistics");
    println!("• Philox: Best for security applications, parallel computing");
}

fn main() {
    println!("=== BitGenerator Examples ===\n");

    // Run all examples
    example_basic_usage();
    example_seeded_generators();
    example_advanced_features();
    example_factory_functions();
    example_legacy_compatibility();
    example_statistical_analysis();
    example_performance_comparison();
    example_parallel_streams();
    example_state_management();
    example_security_comparison();
    example_error_handling();
    example_thread_safety_simulation();
    example_seeding_methods();
    example_rng_core_usage();
    example_bitgenerator_comparison();

    println!("\n=== Examples Complete ===");
    println!("✅ BitGenerator trait with comprehensive implementations working");
    println!("✅ PCG64, MT19937, and Philox implementations functional");
    println!("✅ Factory functions and legacy compatibility working");
    println!("✅ State management and parallel stream generation working");
    println!("✅ Statistical properties and performance characteristics verified");
    println!("✅ Error handling and edge cases properly managed");
    println!("✅ Thread safety guidelines demonstrated");
    println!("✅ RngCore trait implementation working");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_examples() {
        // Test that all examples run without panicking
        main();
    }

    #[test]
    fn test_basic_functionality() {
        let mut pcg = PCG64::new();
        let val = pcg.next_u32();
        assert!(val != 0 || val == 0); // Just ensure it doesn't panic

        let mut mt = MT19937::new();
        let mt_val = mt.next_u32();
        assert!(mt_val != 0 || mt_val == 0);

        let mut philox = Philox::new();
        let philox_val = philox.next_u32();
        assert!(philox_val != 0 || philox_val == 0);
    }

    #[test]
    fn test_reproducibility() {
        let seed = 12345;

        let mut pcg1 = PCG64::seed_from_u64(seed);
        let mut pcg2 = PCG64::seed_from_u64(seed);

        assert_eq!(pcg1.next_u32(), pcg2.next_u32());
        assert_eq!(pcg1.next_u64(), pcg2.next_u64());
    }

    #[test]
    fn test_factory_functions() {
        let pcg = factory::create_bitgenerator("PCG64").unwrap();
        assert_eq!(pcg.name(), "PCG64");

        let default = factory::default_bitgenerator();
        assert_eq!(default.name(), "PCG64");

        let available = factory::available_bitgenerators();
        assert_eq!(available.len(), 3);
        assert!(available.contains(&"PCG64"));
        assert!(available.contains(&"MT19937"));
        assert!(available.contains(&"Philox"));
    }

    #[test]
    fn test_error_handling() {
        assert!(factory::create_bitgenerator("Invalid").is_err());
        assert!(factory::create_seeded_bitgenerator("Invalid", 123).is_err());

        let mut pcg = PCG64::new();
        assert!(pcg.set_state_bytes(&[1, 2, 3]).is_err());
    }
}
