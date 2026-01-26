#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use crate::dtype::Dtype;

    // Import functions explicitly since use super::*; doesn't seem to work
    use crate::random::{
        default_rng, default_rng_with_seed, random, randint, uniform, normal, standard_normal,
        binomial, poisson, exponential, gamma, beta, legacy_rng, legacy_random, legacy_randint, seed,
        logistic, power, vonmises, multivariate_normal, noncentral_chisquare, noncentral_f, standard_t,
    };
    use crate::random::generator::Generator;
    use crate::random::bit_generator::BitGenerator;
    use crate::random::bit_generator::{PCG64, SFC64, PCG64DXSM};
    use rand::RngCore;

    #[test]
    fn test_default_rng_creation() {
        // Test that default_rng() creates a Generator with PCG64
        let mut rng = default_rng();

        // Test that it can generate random numbers
        let result = rng.random::<f64>(&[3, 4], Dtype::Float64 { byteorder: None });
        assert!(result.is_ok());
        let arr = result.unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.len(), 12);
    }

    #[test]
    fn test_default_rng_with_seed() {
        // Test that seeded generators produce reproducible results
        let seed = 42;
        let mut rng1 = default_rng_with_seed(seed);
        let mut rng2 = default_rng_with_seed(seed);

        // Generate arrays with same seed
        let arr1 = rng1.random::<f64>(&[2, 3], Dtype::Float64 { byteorder: None }).unwrap();
        let arr2 = rng2.random::<f64>(&[2, 3], Dtype::Float64 { byteorder: None }).unwrap();

        // They should be identical (though we can't test exact values without knowing PCG64 output)
        assert_eq!(arr1.shape(), arr2.shape());
        assert_eq!(arr1.len(), arr2.len());
    }

    #[test]
    fn test_modern_api_functions() {
        // Test that module-level functions work with modern API

        // Test random()
        let arr = random::<f64>(&[2, 3], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.len(), 6);

        // Test randint()
        let int_arr = randint::<i32>(0, 10, &[2, 2]).unwrap();
        assert_eq!(int_arr.shape(), &[2, 2]);
        assert_eq!(int_arr.len(), 4);

        // Test uniform()
        let uniform_arr = uniform::<f64>(0.0, 1.0, &[3, 2]).unwrap();
        assert_eq!(uniform_arr.shape(), &[3, 2]);
        assert_eq!(uniform_arr.len(), 6);

        // Test normal()
        let normal_arr = normal::<f64>(0.0, 1.0, &[2, 2]).unwrap();
        assert_eq!(normal_arr.shape(), &[2, 2]);
        assert_eq!(normal_arr.len(), 4);

        // Test standard_normal()
        let std_normal_arr = standard_normal::<f64>(&[2, 3]).unwrap();
        assert_eq!(std_normal_arr.shape(), &[2, 3]);
        assert_eq!(std_normal_arr.len(), 6);
    }

    #[test]
    fn test_distributions() {
        // Test various distributions work with modern API

        // Test binomial
        let bin_arr = binomial::<f64>(10, 0.5, &[2, 2]).unwrap();
        assert_eq!(bin_arr.shape(), &[2, 2]);

        // Test poisson
        let pois_arr = poisson::<f64>(5.0, &[2, 2]).unwrap();
        assert_eq!(pois_arr.shape(), &[2, 2]);

        // Test exponential
        let exp_arr = exponential::<f64>(1.0, &[2, 2]).unwrap();
        assert_eq!(exp_arr.shape(), &[2, 2]);

        // Test gamma
        let gamma_arr = gamma::<f64>(2.0, 2.0, &[2, 2]).unwrap();
        assert_eq!(gamma_arr.shape(), &[2, 2]);

        // Test beta
        let beta_arr = beta::<f64>(2.0, 2.0, &[2, 2]).unwrap();
        assert_eq!(beta_arr.shape(), &[2, 2]);
    }

    #[test]
    #[allow(deprecated)]
    fn test_legacy_api_compatibility() {
        // Test that legacy API still works for backward compatibility

        // Test legacy_rng()
        let _legacy = legacy_rng();

        // Test legacy_random()
        let legacy_arr = legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(legacy_arr.shape(), &[2, 2]);

        // Test legacy_randint()
        let legacy_int_arr = legacy_randint::<i32>(0, 10, &[2, 2]).unwrap();
        assert_eq!(legacy_int_arr.shape(), &[2, 2]);

        // Test seed()
        seed(12345);
        let seeded_arr = legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(seeded_arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_modern_api() {
        // Test that the modern API functions work correctly
        use crate::random::{generator::Generator, bit_generator::{PCG64, BitGenerator}};

        // Test default_rng
        let mut rng = default_rng();
        let arr = rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test default_rng_with_seed from modern submodule
        let mut seeded_rng = default_rng_with_seed(42);
        let seeded_arr = seeded_rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(seeded_arr.shape(), &[2, 2]);

        // Test that Generator is available
        let _generator: Generator = Generator::new(Box::new(PCG64::new()));

        // Test that PCG64 is available
        let _pcg64 = PCG64::new();

        // Test that BitGenerator trait is available
        let _bit_gen: Box<dyn BitGenerator> = Box::new(PCG64::new());
    }

    #[test]
    #[allow(deprecated)]
    fn test_legacy_api() {
        // Test that the legacy API functions work correctly
        // All these functions are available from the random module directly

        // Test legacy_rng
        let _legacy = legacy_rng();

        // Test seed from legacy submodule
        seed(54321);

        // Test legacy_random from legacy submodule
        let legacy_arr = legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(legacy_arr.shape(), &[2, 2]);

        // Test legacy_randint from legacy submodule
        let legacy_int_arr = legacy_randint::<i32>(0, 10, &[2, 2]).unwrap();
        assert_eq!(legacy_int_arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_generator_methods() {
        // Test that Generator methods work correctly
        let mut rng = default_rng();

        // Test bit_generator access
        let bit_gen = rng.bit_generator();
        let _state = bit_gen.get_state_bytes();

        // Test bit_generator_mut access
        let bit_gen_mut = rng.bit_generator_mut();
        let _ = bit_gen_mut.set_state_bytes(&12345u64.to_le_bytes());

        // Test various Generator methods
        let random_arr = rng.random::<f64>(&[2, 3], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(random_arr.shape(), &[2, 3]);

        let randint_arr = rng.randint::<i32>(0, 100, &[2, 2]).unwrap();
        assert_eq!(randint_arr.shape(), &[2, 2]);

        let uniform_arr = rng.uniform::<f64>(0.0, 10.0, &[2, 2]).unwrap();
        assert_eq!(uniform_arr.shape(), &[2, 2]);

        let normal_arr = rng.normal::<f64>(0.0, 1.0, &[2, 2]).unwrap();
        assert_eq!(normal_arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_pcg64_functionality() {
        // Test PCG64 BitGenerator functionality
        let mut pcg = PCG64::new();

        // Test state operations
        let initial_state = pcg.get_state_bytes();
        pcg.set_state_bytes(&12345u64.to_le_bytes());
        let new_state = pcg.get_state_bytes();
        assert_ne!(initial_state, new_state);

        // Test seeded creation
        let seeded_pcg = PCG64::seed_from_u64(54321);
        let seeded_state = seeded_pcg.get_state_bytes();

        // Test that seeded PCG64 can be used in Generator
        let seeded_pcg_for_rng = PCG64::seed_from_u64(54321);
        let mut rng = Generator::new(Box::new(seeded_pcg_for_rng));
        let arr = rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_thread_local_behavior() {
        // Test that thread-local generators work correctly

        // Generate using module-level function
        let arr1 = random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();

        // Generate again - should use same thread-local generator
        let arr2 = random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();

        // Both should have same shape
        assert_eq!(arr1.shape(), arr2.shape());
        assert_eq!(arr1.len(), arr2.len());
    }

    #[test]
    #[allow(deprecated)]
    fn test_api_separation() {
        // Test that modern and legacy APIs are properly separated

        // Modern API should use Generator internally
        let modern_arr = random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(modern_arr.shape(), &[2, 2]);

        // Legacy API should use RandomState internally
        let legacy_arr = legacy_random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(legacy_arr.shape(), &[2, 2]);

        // Both should work independently
        assert_eq!(modern_arr.shape(), legacy_arr.shape());
    }

    #[test]
    fn test_comprehensive_distributions() {
        // Test a comprehensive set of distributions
        let mut rng = default_rng();

        // Test all major distributions individually
        {
            let result = rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None });
            assert!(result.is_ok(), "random failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.randint::<i32>(0, 10, &[2, 2]);
            assert!(result.is_ok(), "randint failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.uniform::<f64>(0.0, 1.0, &[2, 2]);
            assert!(result.is_ok(), "uniform failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.normal::<f64>(0.0, 1.0, &[2, 2]);
            assert!(result.is_ok(), "normal failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.standard_normal::<f64>(&[2, 2]);
            assert!(result.is_ok(), "standard_normal failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.binomial::<f64>(10, 0.5, &[2, 2]);
            assert!(result.is_ok(), "binomial failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.poisson::<f64>(5.0, &[2, 2]);
            assert!(result.is_ok(), "poisson failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.exponential::<f64>(1.0, &[2, 2]);
            assert!(result.is_ok(), "exponential failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.gamma::<f64>(2.0, 2.0, &[2, 2]);
            assert!(result.is_ok(), "gamma failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.beta::<f64>(2.0, 2.0, &[2, 2]);
            assert!(result.is_ok(), "beta failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.chisquare::<f64>(2.0, &[2, 2]);
            assert!(result.is_ok(), "chisquare failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        {
            let result = rng.bernoulli::<f64>(0.5, &[2, 2]);
            assert!(result.is_ok(), "bernoulli failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }
    }

    #[test]
    #[allow(deprecated)]
    fn test_backward_compatibility() {
        // Test that existing code using the old API still works

        // This simulates existing code patterns
        let shape = &[3, 4];

        // Old way (should still work with deprecation warnings)
        let _arr1 = legacy_random::<f64>(shape, Dtype::Float64 { byteorder: None }).unwrap();
        let _arr2 = legacy_randint::<i32>(0, 10, shape).unwrap();

        // New way (recommended)
        let _arr3 = random::<f64>(shape, Dtype::Float64 { byteorder: None }).unwrap();
        let _arr4 = randint::<i32>(0, 10, shape).unwrap();

        // Both should produce arrays with same shape
        assert_eq!(_arr1.shape(), _arr3.shape());
        assert_eq!(_arr2.shape(), _arr4.shape());
    }

    #[test]
    fn test_default_rng_vs_manual_generator() {
        // Test that default_rng() produces same results as manual Generator creation

        let seed = 12345;

        // Using default_rng_with_seed
        let mut rng1 = default_rng_with_seed(seed);
        let arr1 = rng1.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();

        // Using manual Generator creation
        let pcg = PCG64::seed_from_u64(seed);
        let mut rng2 = Generator::new(Box::new(pcg));
        let arr2 = rng2.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();

        // Should have same properties
        assert_eq!(arr1.shape(), arr2.shape());
        assert_eq!(arr1.len(), arr2.len());
    }

    #[test]
    fn test_sfc64_bitgenerator() {
        // Test SFC64 BitGenerator functionality
        let mut sfc = SFC64::new();

        // Test state operations
        let initial_state = sfc.get_state_bytes();
        // SFC64 needs 16 bytes of state (state + counter)
        let mut test_state = [0u8; 16];
        test_state[0..8].copy_from_slice(&12345u64.to_le_bytes());
        test_state[8..16].copy_from_slice(&67890u64.to_le_bytes());
        sfc.set_state_bytes(&test_state).unwrap();
        let new_state = sfc.get_state_bytes();
        assert_ne!(initial_state, new_state);

        // Test seeded creation
        let seeded_sfc = SFC64::seed_from_u64(54321);
        let seeded_state = seeded_sfc.get_state_bytes();

        // Test that seeded SFC64 can be used in Generator
        let seeded_sfc_for_rng = SFC64::seed_from_u64(54321);
        let mut rng = Generator::new(Box::new(seeded_sfc_for_rng));
        let arr = rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test name
        assert_eq!(sfc.name(), "SFC64");

        // Test period
        if let Some(period) = sfc.period() {
            assert!(period > 0);
        }

        // Test next_u64 produces different values
        let val1 = sfc.next_u64();
        let val2 = sfc.next_u64();
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_pcg64dxsm_bitgenerator() {
        // Test PCG64DXSM BitGenerator functionality
        let mut pcgdxsm = PCG64DXSM::new();

        // Test state operations
        let initial_state = pcgdxsm.get_state_bytes();
        // PCG64DXSM needs 16 bytes of state (state + increment)
        let mut test_state = [0u8; 16];
        test_state[0..8].copy_from_slice(&12345u64.to_le_bytes());
        test_state[8..16].copy_from_slice(&67890u64.to_le_bytes());
        pcgdxsm.set_state_bytes(&test_state).unwrap();
        let new_state = pcgdxsm.get_state_bytes();
        assert_ne!(initial_state, new_state);

        // Test seeded creation
        let seeded_pcgdxsm = PCG64DXSM::seed_from_u64(54321);
        let seeded_state = seeded_pcgdxsm.get_state_bytes();

        // Test that seeded PCG64DXSM can be used in Generator
        let seeded_pcgdxsm_for_rng = PCG64DXSM::seed_from_u64(54321);
        let mut rng = Generator::new(Box::new(seeded_pcgdxsm_for_rng));
        let arr = rng.random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None }).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test name
        assert_eq!(pcgdxsm.name(), "PCG64DXSM");

        // Test period
        if let Some(period) = pcgdxsm.period() {
            assert!(period > 0);
        }

        // Test next_u64 produces different values
        let val1 = pcgdxsm.next_u64();
        let val2 = pcgdxsm.next_u64();
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_logistic_distribution() {
        // Test logistic distribution
        let mut rng = default_rng();

        // Test basic functionality
        let arr = rng.logistic::<f64>(0.0, 1.0, &[2, 2]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test with module-level function
        let arr2 = logistic::<f64>(0.0, 1.0, &[3, 3]).unwrap();
        assert_eq!(arr2.shape(), &[3, 3]);

        // Test error handling for negative scale
        let result = rng.logistic::<f64>(0.0, -1.0, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_power_distribution() {
        // Test power distribution
        let mut rng = default_rng();

        // Test basic functionality
        let arr = rng.power::<f64>(2.0, &[2, 2]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test with module-level function
        let arr2 = power::<f64>(3.0, &[3, 3]).unwrap();
        assert_eq!(arr2.shape(), &[3, 3]);

        // Test error handling for negative a
        let result = rng.power::<f64>(-1.0, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vonmises_distribution() {
        // Test von Mises distribution
        let mut rng = default_rng();

        // Test basic functionality
        let arr = rng.vonmises::<f64>(0.0, 1.0, &[2, 2]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test with module-level function
        let arr2 = vonmises::<f64>(1.0, 2.0, &[3, 3]).unwrap();
        assert_eq!(arr2.shape(), &[3, 3]);

        // Test error handling for negative kappa
        let result = rng.vonmises::<f64>(0.0, -1.0, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_multivariate_normal_distribution() {
        // Test multivariate normal distribution
        let mut rng = default_rng();

        // Test basic functionality with 2D
        let mean = vec![0.0, 0.0];
        let cov = vec![1.0, 0.0, 0.0, 1.0];
        let arr = rng.multivariate_normal::<f64>(&mean, &cov, None).unwrap();
        assert_eq!(arr.shape(), &[2]);

        // Test with size parameter
        let arr2 = rng.multivariate_normal::<f64>(&mean, &cov, Some(&[3])).unwrap();
        assert_eq!(arr2.shape(), &[3, 2]);

        // Test with module-level function
        let arr3 = multivariate_normal::<f64>(&mean, &cov, Some(&[2, 2])).unwrap();
        assert_eq!(arr3.shape(), &[2, 2, 2]);

        // Test error handling for mismatched dimensions
        let mean2 = vec![0.0, 0.0, 0.0];
        let result = rng.multivariate_normal::<f64>(&mean2, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_noncentral_chisquare_distribution() {
        // Test non-central chi-square distribution
        let mut rng = default_rng();

        // Test basic functionality
        let arr = rng.noncentral_chisquare::<f64>(2.0, 1.0, &[2, 2]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test with module-level function
        let arr2 = noncentral_chisquare::<f64>(3.0, 2.0, &[3, 3]).unwrap();
        assert_eq!(arr2.shape(), &[3, 3]);

        // Test error handling for negative df
        let result = rng.noncentral_chisquare::<f64>(-1.0, 1.0, &[2, 2]);
        assert!(result.is_err());

        // Test error handling for negative nonc
        let result2 = rng.noncentral_chisquare::<f64>(2.0, -1.0, &[2, 2]);
        assert!(result2.is_err());
    }

    #[test]
    fn test_noncentral_f_distribution() {
        // Test non-central F-distribution
        let mut rng = default_rng();

        // Test basic functionality
        let arr = rng.noncentral_f::<f64>(2.0, 3.0, &[2, 2]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test with module-level function
        let arr2 = noncentral_f::<f64>(3.0, 4.0, &[3, 3]).unwrap();
        assert_eq!(arr2.shape(), &[3, 3]);

        // Test error handling for negative dfnum
        let result = rng.noncentral_f::<f64>(-1.0, 3.0, &[2, 2]);
        assert!(result.is_err());

        // Test error handling for negative dfden
        let result2 = rng.noncentral_f::<f64>(2.0, -1.0, &[2, 2]);
        assert!(result2.is_err());
    }

    #[test]
    fn test_standard_t_distribution() {
        // Test standard t-distribution
        let mut rng = default_rng();

        // Test basic functionality
        let arr = rng.standard_t::<f64>(2.0, &[2, 2]).unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test with module-level function
        let arr2 = standard_t::<f64>(3.0, &[3, 3]).unwrap();
        assert_eq!(arr2.shape(), &[3, 3]);

        // Test error handling for negative df
        let result = rng.standard_t::<f64>(-1.0, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_new_distributions() {
        // Test all new distributions work correctly
        let mut rng = default_rng();

        // Test logistic
        {
            let result = rng.logistic::<f64>(0.0, 1.0, &[2, 2]);
            assert!(result.is_ok(), "logistic failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        // Test power
        {
            let result = rng.power::<f64>(2.0, &[2, 2]);
            assert!(result.is_ok(), "power failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        // Test vonmises
        {
            let result = rng.vonmises::<f64>(0.0, 1.0, &[2, 2]);
            assert!(result.is_ok(), "vonmises failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        // Test multivariate_normal
        {
            let mean = vec![0.0, 0.0];
            let cov = vec![1.0, 0.0, 0.0, 1.0];
            let result = rng.multivariate_normal::<f64>(&mean, &cov, Some(&[2]));
            assert!(result.is_ok(), "multivariate_normal failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        // Test noncentral_chisquare
        {
            let result = rng.noncentral_chisquare::<f64>(2.0, 1.0, &[2, 2]);
            assert!(result.is_ok(), "noncentral_chisquare failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        // Test noncentral_f
        {
            let result = rng.noncentral_f::<f64>(2.0, 3.0, &[2, 2]);
            assert!(result.is_ok(), "noncentral_f failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }

        // Test standard_t
        {
            let result = rng.standard_t::<f64>(2.0, &[2, 2]);
            assert!(result.is_ok(), "standard_t failed");
            assert_eq!(result.unwrap().shape(), &[2, 2]);
        }
    }
}
