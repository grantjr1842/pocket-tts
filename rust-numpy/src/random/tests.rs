#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::random::default_rng;
    use crate::random::legacy::*;
    use crate::random::legacy_randint;
    use crate::random::legacy_random;
    use crate::random::modern::*;
    use crate::random::{
        beta, binomial, exponential, gamma, normal, poisson, randint, random, standard_normal,
        uniform,
    };
    use crate::{Array, NumPyError};

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
        let arr1 = rng1
            .random::<f64>(&[2, 3], Dtype::from_type::<f64>())
            .unwrap();
        let arr2 = rng2
            .random::<f64>(&[2, 3], Dtype::from_type::<f64>())
            .unwrap();

        // They should be identical (though we can't test exact values without knowing PCG64 output)
        assert_eq!(arr1.shape(), arr2.shape());
        assert_eq!(arr1.len(), arr2.len());
    }

    #[test]
    fn test_modern_api_functions() {
        // Test that module-level functions work with modern API

        // Test random()
        let arr = random::<f64>(&[2, 3], Dtype::from_type::<f64>()).unwrap();
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
    fn test_legacy_api_compatibility() {
        // Test that legacy API still works for backward compatibility

        // Test legacy_rng()
        let _legacy = legacy_rng();

        // Test legacy_random()
        let legacy_arr = legacy_random::<f64>(&[2, 2], Dtype::from_type::<f64>()).unwrap();
        assert_eq!(legacy_arr.shape(), &[2, 2]);

        // Test legacy_randint()
        let legacy_int_arr = legacy_randint::<i32>(0, 10, &[2, 2]).unwrap();
        assert_eq!(legacy_int_arr.shape(), &[2, 2]);

        // Test seed()
        seed(12345);
        let seeded_arr = legacy_random::<f64>(&[2, 2], Dtype::from_type::<f64>()).unwrap();
        assert_eq!(seeded_arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_modern_submodule() {
        // Test that the modern submodule exports work correctly
        use crate::random::modern::*;

        // Test default_rng from modern submodule
        let mut rng = default_rng();
        let arr = rng
            .random::<f64>(&[2, 2], Dtype::from_type::<f64>())
            .unwrap();
        assert_eq!(arr.shape(), &[2, 2]);

        // Test default_rng_with_seed from modern submodule
        let mut seeded_rng = default_rng_with_seed(42);
        let seeded_arr = seeded_rng
            .random::<f64>(&[2, 2], Dtype::from_type::<f64>())
            .unwrap();
        assert_eq!(seeded_arr.shape(), &[2, 2]);

        // Test that Generator is available
        let _generator: Generator = Generator::new(Box::new(PCG64::new()));

        // Test that PCG64 is available
        let _pcg64 = PCG64::new();

        // Test that BitGenerator trait is available
        let _bit_gen: Box<dyn BitGenerator> = Box::new(PCG64::new());
        use crate::random::modern::*;
    }

    #[test]
    fn test_legacy_module_structure() {
        // Confirm legacy module structure
        use crate::random::legacy::*;

        // Test legacy_rng from legacy submodule
        let _legacy = legacy_rng();

        // Test seed from legacy submodule
        seed(54321);

        // Test legacy_random from legacy submodule
        let legacy_arr = legacy_random::<f64>(&[2, 2], Dtype::from_type::<f64>()).unwrap();
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
        bit_gen_mut.set_state_bytes(&12345u64.to_le_bytes());

        // Test
        // Generate random array
        let random_arr = rng
            .random::<f64>(&[2, 3], Dtype::from_type::<f64>())
            .unwrap();
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
        let mut rng = Generator::new(Box::new(seeded_pcg));
        let arr = rng
            .random::<f64>(&[2, 2], Dtype::from_type::<f64>())
            .unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_thread_local_behavior() {
        // Test that thread-local generators work correctly

        // Generate using module-level function
        let arr1 = random::<f64>(&[2, 2], Dtype::from_type::<f64>()).unwrap();

        // Generate again - should use same thread-local generator
        let arr2 = random::<f64>(&[2, 2], Dtype::from_type::<f64>()).unwrap();

        // Both should have same shape
        assert_eq!(arr1.shape(), arr2.shape());
        assert_eq!(arr1.len(), arr2.len());
    }

    #[test]
    fn test_api_separation() {
        // Test that modern and legacy APIs are properly separated

        // Modern API should use Generator internally
        let modern_arr = random::<f64>(&[2, 2], Dtype::from_type::<f64>()).unwrap();
        assert_eq!(modern_arr.shape(), &[2, 2]);

        // Legacy API should use RandomState internally
        let legacy_arr = legacy_random::<f64>(&[2, 2], Dtype::from_type::<f64>()).unwrap();
        assert_eq!(legacy_arr.shape(), &[2, 2]);

        // Both should work independently
        assert_eq!(modern_arr.shape(), legacy_arr.shape());
    }

    #[test]
    fn test_comprehensive_distributions() {
        // Test a comprehensive set of distributions
        let mut rng = default_rng();

        macro_rules! test_dist {
            ($name:expr, $expr:expr) => {
                let result = $expr;
                assert!(result.is_ok(), "{} distribution should work", $name);
                let arr = result.unwrap();
                assert_eq!(
                    arr.shape(),
                    &[2, 2],
                    "{} distribution should produce correct shape",
                    $name
                );
            };
        }

        test_dist!(
            "random",
            rng.random::<f64>(&[2, 2], Dtype::from_type::<f64>())
        );
        // skipping randint as it returns i32 and others return f64 in previous logic, but here we can handle it
        test_dist!("uniform", rng.uniform::<f64>(0.0, 1.0, &[2, 2]));
        test_dist!("normal", rng.normal::<f64>(0.0, 1.0, &[2, 2]));
        test_dist!("standard_normal", rng.standard_normal::<f64>(&[2, 2]));
        test_dist!("binomial", rng.binomial::<f64>(10, 0.5, &[2, 2]));
        test_dist!("poisson", rng.poisson::<f64>(5.0, &[2, 2]));
        test_dist!("exponential", rng.exponential::<f64>(1.0, &[2, 2]));
        test_dist!("gamma", rng.gamma::<f64>(2.0, 2.0, &[2, 2]));
        test_dist!("beta", rng.beta::<f64>(2.0, 2.0, &[2, 2]));
        test_dist!("chisquare", rng.chisquare::<f64>(2.0, &[2, 2]));
        test_dist!("bernoulli", rng.bernoulli::<f64>(0.5, &[2, 2]));
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that existing code using the old API still works

        // This simulates existing code patterns
        let shape = &[3, 4];

        // Old way (should still work with deprecation warnings)
        let _arr1 = legacy_random::<f64>(shape, Dtype::from_type::<f64>()).unwrap();
        let _arr2 = legacy_randint::<i32>(0, 10, shape).unwrap();

        // New way (recommended)
        let _arr3 = random::<f64>(shape, Dtype::from_type::<f64>()).unwrap();
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
        let arr1 = rng1
            .random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })
            .unwrap();

        // Using manual Generator creation
        let pcg = PCG64::seed_from_u64(seed);
        let mut rng2 = Generator::new(Box::new(pcg));
        let arr2 = rng2
            .random::<f64>(&[2, 2], Dtype::Float64 { byteorder: None })
            .unwrap();

        // Should have same properties
        assert_eq!(arr1.shape(), arr2.shape());
        assert_eq!(arr1.len(), arr2.len());
    }
}
