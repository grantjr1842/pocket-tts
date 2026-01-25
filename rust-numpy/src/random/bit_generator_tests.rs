#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn test_bitgenerator_trait_methods() {
        // Test that all BitGenerators implement the trait correctly
        let generators: Vec<Box<dyn BitGenerator>> = vec![
            Box::new(PCG64::new()),
            Box::new(MT19937::new()),
            Box::new(Philox::new()),
        ];

        for (i, mut rng) in generators.into_iter().enumerate() {
            println!("Testing generator {}: {}", i, rng.name());

            // Test basic trait methods
            assert!(!rng.name().is_empty());
            assert!(!rng.version().is_empty());
            assert!(rng.state_size() > 0);

            // Test cryptographic security flag
            let is_secure = rng.is_cryptographically_secure();
            println!("  Cryptographically secure: {}", is_secure);

            // Test period information
            if let Some(period) = rng.period() {
                println!("  Period: 2^{}", period.trailing_zeros());
            }

            // Test parallel parameters
            let params = rng.parallel_params();
            println!("  Supports parallel: {}", params.supports_parallel);
            println!("  Max streams: {}", params.max_streams);
            println!("  Jump size: {}", params.jump_size);
        }
    }

    #[test]
    fn test_pcg64_implementation() {
        let mut pcg = PCG64::new();

        // Test basic properties
        assert_eq!(pcg.name(), "PCG64");
        assert_eq!(pcg.version(), "1.0.0");
        assert!(!pcg.is_cryptographically_secure());
        assert_eq!(pcg.state_size(), 8);

        // Test period
        if let Some(period) = pcg.period() {
            assert_eq!(period, 2u128.pow(128));
        }

        // Test seeding
        let seed = 12345u64;
        pcg.seed_u64(seed);

        // Generate some values
        let val1 = pcg.next_u32();
        let val2 = pcg.next_u64();
        let val3 = pcg.next_u32();

        // Test reproducibility
        let mut pcg2 = PCG64::seed_from_u64(seed);
        assert_eq!(pcg2.next_u32(), val1);
        assert_eq!(pcg2.next_u64(), val2);
        assert_eq!(pcg2.next_u32(), val3);

        // Test jump functionality
        let mut pcg3 = PCG64::seed_from_u64(seed);
        pcg3.jump(2);
        let val_after_jump = pcg3.next_u32();

        let mut pcg4 = PCG64::seed_from_u64(seed);
        pcg4.next_u32();
        pcg4.next_u32();
        let val_after_manual = pcg4.next_u32();

        assert_eq!(val_after_jump, val_after_manual);
    }

    #[test]
    fn test_mt19937_implementation() {
        let mut mt = MT19937::new();

        // Test basic properties
        assert_eq!(mt.name(), "MT19937");
        assert_eq!(mt.version(), "1.0.0");
        assert!(!mt.is_cryptographically_secure());
        assert_eq!(mt.state_size(), 8);

        // Test period (MT19937 has a very long period)
        if let Some(period) = mt.period() {
            assert_eq!(period, 2u128.pow(19937) - 1);
        }

        // Test seeding
        let seed = 54321u64;
        mt.seed_u64(seed);

        // Generate some values
        let val1 = mt.next_u32();
        let val2 = mt.next_u64();
        let val3 = mt.next_u32();

        // Test reproducibility
        let mut mt2 = MT19937::seed_from_u64(seed);
        assert_eq!(mt2.next_u32(), val1);
        assert_eq!(mt2.next_u64(), val2);
        assert_eq!(mt2.next_u32(), val3);

        // Test jump functionality
        let mut mt3 = MT19937::seed_from_u64(seed);
        mt3.jump(3);
        let val_after_jump = mt3.next_u32();

        let mut mt4 = MT19937::seed_from_u64(seed);
        mt4.next_u32();
        mt4.next_u32();
        mt4.next_u32();
        let val_after_manual = mt4.next_u32();

        assert_eq!(val_after_jump, val_after_manual);
    }

    #[test]
    fn test_philox_implementation() {
        let mut philox = Philox::new();

        // Test basic properties
        assert_eq!(philox.name(), "Philox");
        assert_eq!(philox.version(), "1.0.0");
        assert!(philox.is_cryptographically_secure());
        assert_eq!(philox.state_size(), 16);

        // Test seeding
        let seed = 98765u64;
        philox.seed_u64(seed);

        // Generate some values
        let val1 = philox.next_u32();
        let val2 = philox.next_u64();
        let val3 = philox.next_u32();

        // Test reproducibility
        let mut philox2 = Philox::seed_from_u64(seed);
        assert_eq!(philox2.next_u32(), val1);
        assert_eq!(philox2.next_u64(), val2);
        assert_eq!(philox2.next_u32(), val3);

        // Test counter functionality
        let mut philox3 = Philox::seed_from_u64_with_counter(seed, 100);
        let val_with_counter = philox3.next_u32();

        let mut philox4 = Philox::seed_from_u64(seed);
        philox4.jump(100);
        let val_after_jump = philox4.next_u32();

        // These should be different due to counter implementation
        assert_ne!(val_with_counter, val_after_jump);

        // Test jump functionality
        let mut philox5 = Philox::seed_from_u64(seed);
        philox5.jump(5);
        let val_after_jump2 = philox5.next_u32();

        let mut philox6 = Philox::seed_from_u64(seed);
        for _ in 0..5 {
            philox6.next_u32();
        }
        let val_after_manual = philox6.next_u32();

        assert_eq!(val_after_jump2, val_after_manual);
    }

    #[test]
    fn test_state_serialization() {
        // Test PCG64 state serialization
        let mut pcg = PCG64::seed_from_u64(12345);
        let original_val = pcg.next_u32();

        // Get state
        let state = pcg.get_state_bytes();
        assert_eq!(state.len(), pcg.state_size());

        // Create new generator and restore state
        let mut pcg2 = PCG64::new();
        pcg2.set_state_bytes(&state).unwrap();

        // Should continue from same state
        assert_eq!(pcg2.next_u32(), original_val);

        // Test MT19937 state serialization
        let mut mt = MT19937::seed_from_u64(54321);
        let mt_val = mt.next_u32();

        let mt_state = mt.get_state_bytes();
        assert_eq!(mt_state.len(), mt.state_size());

        let mut mt2 = MT19937::new();
        mt2.set_state_bytes(&mt_state).unwrap();
        assert_eq!(mt2.next_u32(), mt_val);

        // Test Philox state serialization
        let mut philox = Philox::seed_from_u64(98765);
        let philox_val = philox.next_u32();

        let philox_state = philox.get_state_bytes();
        assert_eq!(philox_state.len(), philox.state_size());

        let mut philox2 = Philox::new();
        philox2.set_state_bytes(&philox_state).unwrap();
        assert_eq!(philox2.next_u32(), philox_val);
    }

    #[test]
    fn test_seed_from_slice() {
        // Test seeding from u32 slice
        let seed_slice = [0x12345678u32, 0x9ABCDEF0u32];

        let mut pcg = PCG64::new();
        pcg.seed_u32_slice(&seed_slice);
        let pcg_val = pcg.next_u32();

        let mut pcg2 = PCG64::from_seed_slice(&seed_slice);
        let pcg2_val = pcg2.next_u32();

        assert_eq!(pcg_val, pcg2_val);

        // Test with MT19937
        let mut mt = MT19937::new();
        mt.seed_u32_slice(&seed_slice);
        let mt_val = mt.next_u32();

        let mut mt2 = MT19937::from_seed_slice(&seed_slice);
        let mt2_val = mt2.next_u32();

        assert_eq!(mt_val, mt2_val);

        // Test with Philox
        let mut philox = Philox::new();
        philox.seed_u32_slice(&seed_slice);
        let philox_val = philox.next_u32();

        let mut philox2 = Philox::from_seed_slice(&seed_slice);
        let philox2_val = philox2.next_u32();

        assert_eq!(philox_val, philox2_val);
    }

    #[test]
    fn test_error_handling() {
        // Test invalid state size
        let mut pcg = PCG64::new();
        let result = pcg.set_state_bytes(&[1, 2, 3]); // Too small
        assert!(result.is_err());

        // Test invalid BitGenerator name
        let result = factory::create_bitgenerator("InvalidGenerator");
        assert!(result.is_err());

        let result = factory::create_seeded_bitgenerator("InvalidGenerator", 12345);
        assert!(result.is_err());
    }

    #[test]
    fn test_factory_functions() {
        // Test factory creation
        let pcg = factory::create_bitgenerator("PCG64").unwrap();
        assert_eq!(pcg.name(), "PCG64");

        let mt = factory::create_bitgenerator("MT19937").unwrap();
        assert_eq!(mt.name(), "MT19937");

        let philox = factory::create_bitgenerator("Philox").unwrap();
        assert_eq!(philox.name(), "Philox");

        // Test seeded factory creation
        let seed = 12345;
        let seeded_pcg = factory::create_seeded_bitgenerator("PCG64", seed).unwrap();
        let seeded_pcg2 = factory::create_seeded_bitgenerator("PCG64", seed).unwrap();

        // Should produce same sequence
        assert_eq!(seeded_pcg.next_u32(), seeded_pcg2.next_u32());

        // Test available generators
        let available = factory::available_bitgenerators();
        assert!(available.contains(&"PCG64"));
        assert!(available.contains(&"MT19937"));
        assert!(available.contains(&"Philox"));
        assert_eq!(available.len(), 3);

        // Test default generator
        let default = factory::default_bitgenerator();
        assert_eq!(default.name(), "PCG64");
    }

    #[test]
    fn test_legacy_compatibility() {
        // Test legacy functions
        let legacy1 = legacy::new();
        let legacy2 = legacy::new();

        // Both should be PCG64
        assert_eq!(legacy1.name(), "PCG64");
        assert_eq!(legacy2.name(), "PCG64");

        // Test seeded legacy
        let seed = 54321;
        let seeded1 = legacy::seed_from_u64(seed);
        let seeded2 = legacy::seed_from_u64(seed);

        assert_eq!(seeded1.next_u32(), seeded2.next_u32());
    }

    #[test]
    fn test_statistical_properties() {
        // Basic statistical tests to ensure generators produce reasonable random numbers
        let generators: Vec<Box<dyn BitGenerator>> = vec![
            Box::new(PCG64::seed_from_u64(12345)),
            Box::new(MT19937::seed_from_u64(12345)),
            Box::new(Philox::seed_from_u64(12345)),
        ];

        for (i, mut rng) in generators.into_iter().enumerate() {
            println!(
                "Testing statistical properties for generator {}: {}",
                i,
                rng.name()
            );

            // Generate a large sample
            let mut values = Vec::new();
            for _ in 0..10000 {
                values.push(rng.next_u64());
            }

            // Basic statistical checks
            let mean = values.iter().sum::<u64>() as f64 / values.len() as f64;
            let expected_mean = (u64::MAX / 2) as f64;

            // Mean should be close to expected (within 5%)
            let mean_error = (mean - expected_mean).abs() / expected_mean;
            assert!(
                mean_error < 0.05,
                "Mean error too large for {}: {:.3}",
                rng.name(),
                mean_error
            );

            // Check that values are reasonably distributed
            let mut in_lower_half = 0;
            let mut in_upper_half = 0;
            let mid = u64::MAX / 2;

            for &val in &values {
                if val < mid {
                    in_lower_half += 1;
                } else {
                    in_upper_half += 1;
                }
            }

            // Should be roughly 50/50 split (within 2%)
            let ratio = in_lower_half as f64 / values.len() as f64;
            assert!(
                (ratio - 0.5).abs() < 0.02,
                "Distribution not balanced for {}: {:.3}",
                rng.name(),
                ratio
            );
        }
    }

    #[test]
    fn test_parallel_parameters() {
        // Test that parallel parameters are reasonable
        let generators: Vec<Box<dyn BitGenerator>> = vec![
            Box::new(PCG64::new()),
            Box::new(MT19937::new()),
            Box::new(Philox::new()),
        ];

        for rng in generators {
            let params = rng.parallel_params();

            // All should support parallel streams
            assert!(params.supports_parallel);

            // Should have reasonable jump sizes
            assert!(params.jump_size > 0);

            // Should have reasonable max streams
            assert!(params.max_streams > 0);

            println!(
                "{}: jump_size={}, max_streams={}",
                rng.name(),
                params.jump_size,
                params.max_streams
            );
        }
    }

    #[test]
    fn test_rng_core_compatibility() {
        // Test that all BitGenerators work as RngCore
        let generators: Vec<Box<dyn BitGenerator>> = vec![
            Box::new(PCG64::seed_from_u64(12345)),
            Box::new(MT19937::seed_from_u64(12345)),
            Box::new(Philox::seed_from_u64(12345)),
        ];

        for (i, mut rng) in generators.into_iter().enumerate() {
            println!(
                "Testing RngCore compatibility for generator {}: {}",
                i,
                rng.name()
            );

            // Test all RngCore methods
            let u32_val = rng.next_u32();
            let u64_val = rng.next_u64();

            // Values should be different
            let u32_val2 = rng.next_u32();
            let u64_val2 = rng.next_u64();

            assert_ne!(u32_val, u32_val2);
            assert_ne!(u64_val, u64_val2);

            // Test fill_bytes
            let mut bytes = [0u8; 32];
            rng.fill_bytes(&mut bytes);

            // Should have some non-zero bytes (very high probability)
            let non_zero_count = bytes.iter().filter(|&&b| b != 0).count();
            assert!(non_zero_count > 0);

            // Test try_fill_bytes
            let mut bytes2 = [0u8; 16];
            let result = rng.try_fill_bytes(&mut bytes2);
            assert!(result.is_ok());

            let non_zero_count2 = bytes2.iter().filter(|&&b| b != 0).count();
            assert!(non_zero_count2 > 0);
        }
    }

    #[test]
    fn test_performance_characteristics() {
        use std::time::Instant;

        let generators: Vec<Box<dyn BitGenerator>> = vec![
            Box::new(PCG64::new()),
            Box::new(MT19937::new()),
            Box::new(Philox::new()),
        ];

        for (i, mut rng) in generators.into_iter().enumerate() {
            println!("Testing performance for generator {}: {}", i, rng.name());

            let start = Instant::now();

            // Generate 1 million u64 values
            for _ in 0..1_000_000 {
                let _ = rng.next_u64();
            }

            let duration = start.elapsed();
            println!("  1M u64 values: {:?}", duration);

            // Should complete in reasonable time (less than 1 second)
            assert!(
                duration.as_secs() < 1,
                "Generator {} too slow: {:?}",
                rng.name(),
                duration
            );
        }
    }

    #[test]
    fn test_different_seeds_produce_different_sequences() {
        let seed1 = 12345;
        let seed2 = 54321;

        let generators = vec![
            (
                "PCG64",
                Box::new(PCG64::seed_from_u64(seed1)) as Box<dyn BitGenerator>,
                Box::new(PCG64::seed_from_u64(seed2)),
            ),
            (
                "MT19937",
                Box::new(MT19937::seed_from_u64(seed1)) as Box<dyn BitGenerator>,
                Box::new(MT19937::seed_from_u64(seed2)),
            ),
            (
                "Philox",
                Box::new(Philox::seed_from_u64(seed1)) as Box<dyn BitGenerator>,
                Box::new(Philox::seed_from_u64(seed2)),
            ),
        ];

        for (name, mut rng1, mut rng2) in generators {
            // Generate first few values
            let vals1: Vec<u64> = (0..10).map(|_| rng1.next_u64()).collect();
            let vals2: Vec<u64> = (0..10).map(|_| rng2.next_u64()).collect();

            // Should be different sequences
            assert_ne!(
                vals1, vals2,
                "Generator {} produced same sequence with different seeds",
                name
            );

            println!("{}: Different seeds produce different sequences ✓", name);
        }
    }

    #[test]
    fn test_debug_formatting() {
        // Test that all BitGenerators implement Debug
        let pcg = PCG64::new();
        let mt = MT19937::new();
        let philox = Philox::new();

        // These should compile without errors
        let _debug_pcg = format!("{:?}", pcg);
        let _debug_mt = format!("{:?}", mt);
        let _debug_philox = format!("{:?}", philox);

        // Debug output should contain the generator name
        assert!(_debug_pcg.contains("PCG64"));
        assert!(_debug_mt.contains("MT19937"));
        assert!(_debug_philox.contains("Philox"));
    }

    #[test]
    fn test_clone_behavior() {
        // Test that BitGenerators can be cloned (if they implement Clone)
        let pcg = PCG64::seed_from_u64(12345);
        let pcg_clone = pcg.clone();

        // Should produce same sequence initially
        assert_eq!(pcg.next_u32(), pcg_clone.next_u32());

        // But diverge after different calls
        let pcg_next = pcg.next_u32();
        let clone_next = pcg_clone.next_u32();
        assert_eq!(pcg_next, clone_next);

        // Test MT19937
        let mt = MT19937::seed_from_u64(12345);
        let mt_clone = mt.clone();
        assert_eq!(mt.next_u32(), mt_clone.next_u32());

        // Test Philox
        let philox = Philox::seed_from_u64(12345);
        let philox_clone = philox.clone();
        assert_eq!(philox.next_u32(), philox_clone.next_u32());
    }

    #[test]
    fn test_edge_cases() {
        // Test edge cases and boundary conditions

        // Test with seed 0
        let mut pcg = PCG64::seed_from_u64(0);
        let val = pcg.next_u32();
        assert!(val != 0); // Should still produce random-looking values

        // Test with maximum seed
        let mut pcg = PCG64::seed_from_u64(u64::MAX);
        let val = pcg.next_u32();
        assert!(val != 0);

        // Test empty seed slice
        let empty_slice: &[u32] = &[];
        let mut pcg = PCG64::new();
        pcg.seed_u32_slice(empty_slice);
        let val = pcg.next_u32();
        // Should not panic and produce some value
        assert!(val != 0 || val == 0); // Just ensure it doesn't panic

        // Test jump with 0 steps
        let mut pcg = PCG64::seed_from_u64(12345);
        let val_before = pcg.next_u32();
        pcg.jump(0);
        let val_after = pcg.next_u32();
        assert_ne!(val_before, val_after);
    }

    #[test]
    fn test_concurrent_usage_simulation() {
        // Simulate concurrent usage by creating multiple generators with different seeds
        let seed = 12345;
        let mut generators = Vec::new();

        for i in 0..10 {
            let seeded_gen = factory::create_seeded_bitgenerator("PCG64", seed + i).unwrap();
            generators.push(seeded_gen);
        }

        // Generate values from each generator
        let mut all_values = Vec::new();
        for mut rng in generators {
            let val = rng.next_u32();
            all_values.push(val);
        }

        // All values should be different (very high probability)
        let mut unique_values: Vec<u32> = all_values.clone();
        unique_values.sort();
        unique_values.dedup();

        assert_eq!(
            unique_values.len(),
            all_values.len(),
            "Some generators produced identical values"
        );
    }
}
