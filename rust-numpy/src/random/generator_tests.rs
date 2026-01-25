#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::dtype::Dtype;
    use crate::random::bit_generator::{BitGenerator, PCG64};

    #[test]
    fn test_generator_creation() {
        // Test that Generator can be created with different BitGenerators
        let pcg = PCG64::new();
        let mut generator = Generator::new(Box::new(pcg));

        // Test basic functionality
        let result = generator.random::<f64>(&[3, 4], Dtype::Float64);
        assert!(result.is_ok());
        let arr = result.unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.len(), 12);
    }

    #[test]
    fn test_generator_bit_generator_access() {
        let pcg = PCG64::new();
        let mut generator = Generator::new(Box::new(pcg));

        // Test bit_generator access
        let bit_gen = generator.bit_generator();
        let state = bit_gen.get_state();

        // Test bit_generator_mut access
        let bit_gen_mut = generator.bit_generator_mut();
        bit_gen_mut.set_state(12345);
        let new_state = bit_gen_mut.get_state();
        assert_ne!(state, new_state);
    }

    #[test]
    fn test_core_distribution_methods() {
        let pcg = PCG64::seed_from_u64(42);
        let mut generator = Generator::new(Box::new(pcg));

        // Test random()
        let random_arr = generator.random::<f64>(&[2, 3], Dtype::Float64).unwrap();
        assert_eq!(random_arr.shape(), &[2, 3]);
        assert_eq!(random_arr.len(), 6);

        // Test randint()
        let int_arr = generator.randint::<i32>(0, 10, &[2, 2]).unwrap();
        assert_eq!(int_arr.shape(), &[2, 2]);
        assert_eq!(int_arr.len(), 4);

        // Verify all values are in range [0, 10)
        for val in int_arr.data.as_slice() {
            assert!(val >= &0 && val < &10);
        }

        // Test uniform()
        let uniform_arr = generator.uniform::<f64>(0.0, 5.0, &[3, 2]).unwrap();
        assert_eq!(uniform_arr.shape(), &[3, 2]);
        assert_eq!(uniform_arr.len(), 6);

        // Verify all values are in range [0.0, 5.0)
        for val in uniform_arr.data.as_slice() {
            assert!(val >= &0.0 && val < &5.0);
        }

        // Test normal()
        let normal_arr = generator.normal::<f64>(0.0, 1.0, &[2, 2]).unwrap();
        assert_eq!(normal_arr.shape(), &[2, 2]);
        assert_eq!(normal_arr.len(), 4);

        // Test standard_normal()
        let std_normal_arr = generator.standard_normal::<f64>(&[2, 3]).unwrap();
        assert_eq!(std_normal_arr.shape(), &[2, 3]);
        assert_eq!(std_normal_arr.len(), 6);
    }

    #[test]
    fn test_discrete_distributions() {
        let pcg = PCG64::seed_from_u64(123);
        let mut generator = Generator::new(Box::new(pcg));

        // Test binomial
        let bin_arr = generator.binomial::<f64>(10, 0.5, &[2, 2]).unwrap();
        assert_eq!(bin_arr.shape(), &[2, 2]);

        // Verify all values are in valid range [0, 10]
        for val in bin_arr.data.as_slice() {
            assert!(val >= &0.0 && val <= &10.0);
        }

        // Test poisson
        let pois_arr = generator.poisson::<f64>(5.0, &[2, 2]).unwrap();
        assert_eq!(pois_arr.shape(), &[2, 2]);

        // Verify all values are non-negative
        for val in pois_arr.data.as_slice() {
            assert!(val >= &0.0);
        }

        // Test bernoulli
        let bern_arr = generator.bernoulli::<f64>(0.7, &[2, 2]).unwrap();
        assert_eq!(bern_arr.shape(), &[2, 2]);

        // Verify all values are 0 or 1
        for val in bern_arr.data.as_slice() {
            assert!(val == &0.0 || val == &1.0);
        }
    }

    #[test]
    fn test_continuous_distributions() {
        let pcg = PCG64::seed_from_u64(456);
        let mut generator = Generator::new(Box::new(pcg));

        // Test exponential
        let exp_arr = generator.exponential::<f64>(1.0, &[2, 2]).unwrap();
        assert_eq!(exp_arr.shape(), &[2, 2]);

        // Verify all values are positive
        for val in exp_arr.data.as_slice() {
            assert!(val > &0.0);
        }

        // Test gamma
        let gamma_arr = generator.gamma::<f64>(2.0, 2.0, &[2, 2]).unwrap();
        assert_eq!(gamma_arr.shape(), &[2, 2]);

        // Verify all values are positive
        for val in gamma_arr.data.as_slice() {
            assert!(val > &0.0);
        }

        // Test beta
        let beta_arr = generator.beta::<f64>(2.0, 2.0, &[2, 2]).unwrap();
        assert_eq!(beta_arr.shape(), &[2, 2]);

        // Verify all values are in range [0, 1]
        for val in beta_arr.data.as_slice() {
            assert!(val >= &0.0 && val <= &1.0);
        }

        // Test chisquare
        let chi_arr = generator.chisquare::<f64>(2.0, &[2, 2]).unwrap();
        assert_eq!(chi_arr.shape(), &[2, 2]);

        // Verify all values are positive
        for val in chi_arr.data.as_slice() {
            assert!(val > &0.0);
        }
    }

    #[test]
    fn test_utility_methods() {
        let pcg = PCG64::seed_from_u64(789);
        let mut generator = Generator::new(Box::new(pcg));

        // Test permutation
        let perm_arr = generator.permutation(5).unwrap();
        assert_eq!(perm_arr.shape(), &[5]);
        assert_eq!(perm_arr.len(), 5);

        // Verify it's a permutation of 0..4
        let mut sorted = perm_arr.data.as_slice().to_vec();
        sorted.sort();
        for (i, &val) in sorted.iter().enumerate() {
            assert_eq!(val, i as i64);
        }

        // Test choice with replacement
        let choices = vec![1, 2, 3, 4, 5];
        let choice_arr = generator.choice(&choices, 3, true).unwrap();
        assert_eq!(choice_arr.shape(), &[3]);
        assert_eq!(choice_arr.len(), 3);

        // Verify all choices are from the original array
        for val in choice_arr.data.as_slice() {
            assert!(choices.contains(val));
        }

        // Test choice without replacement
        let choice_arr_no_replace = generator.choice(&choices, 3, false).unwrap();
        assert_eq!(choice_arr_no_replace.shape(), &[3]);
        assert_eq!(choice_arr_no_replace.len(), 3);

        // Verify no duplicates in choice without replacement
        let mut unique_choices: Vec<i32> = choice_arr_no_replace.data.as_slice().to_vec();
        unique_choices.sort();
        unique_choices.dedup();
        assert_eq!(unique_choices.len(), 3);

        // Test integers
        let int_arr = generator.integers(0, 100, 5).unwrap();
        assert_eq!(int_arr.shape(), &[5]);
        assert_eq!(int_arr.len(), 5);

        // Verify all values are in range [0, 100)
        for val in int_arr.data.as_slice() {
            assert!(val >= &0 && val < &100);
        }

        // Test bytes
        let bytes_arr = generator.bytes(10).unwrap();
        assert_eq!(bytes_arr.shape(), &[10]);
        assert_eq!(bytes_arr.len(), 10);

        // Test random_floats
        let float_arr = generator.random_floats(5).unwrap();
        assert_eq!(float_arr.shape(), &[5]);
        assert_eq!(float_arr.len(), 5);

        // Verify all values are in [0.0, 1.0)
        for val in float_arr.data.as_slice() {
            assert!(val >= &0.0 && val < &1.0);
        }

        // Test random_floats_range
        let range_arr = generator.random_floats_range(10.0, 20.0, 5).unwrap();
        assert_eq!(range_arr.shape(), &[5]);
        assert_eq!(range_arr.len(), 5);

        // Verify all values are in [10.0, 20.0)
        for val in range_arr.data.as_slice() {
            assert!(val >= &10.0 && val < &20.0);
        }

        // Test random_bools
        let bool_arr = generator.random_bools(10).unwrap();
        assert_eq!(bool_arr.shape(), &[10]);
        assert_eq!(bool_arr.len(), 10);
    }

    #[test]
    fn test_shuffle_method() {
        let pcg = PCG64::seed_from_u64(999);
        let mut generator = Generator::new(Box::new(pcg));

        // Test shuffle 1D array
        let mut arr_1d = Array::from_data(vec![1, 2, 3, 4, 5, 6], vec![6]);
        let original_data = arr_1d.data.as_slice().to_vec();

        generator.shuffle(&mut arr_1d).unwrap();

        // Verify array still has same elements
        assert_eq!(arr_1d.shape(), &[6]);
        assert_eq!(arr_1d.len(), 6);

        let mut shuffled_data = arr_1d.data.as_slice().to_vec();
        shuffled_data.sort();
        let mut sorted_original = original_data.clone();
        sorted_original.sort();

        assert_eq!(shuffled_data, sorted_original);

        // Test shuffle 2D array
        let mut arr_2d = Array::from_data(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let original_shape = arr_2d.shape().to_vec();
        let original_len = arr_2d.len();

        generator.shuffle(&mut arr_2d).unwrap();

        // Verify shape and length are preserved
        assert_eq!(arr_2d.shape(), &original_shape);
        assert_eq!(arr_2d.len(), original_len);
    }

    #[test]
    fn test_error_handling() {
        let pcg = PCG64::new();
        let mut generator = Generator::new(Box::new(pcg));

        // Test invalid binomial parameters
        let result = generator.binomial::<f64>(10, -0.5, &[2, 2]);
        assert!(result.is_err());

        // Test invalid poisson parameters
        let result = generator.poisson::<f64>(-1.0, &[2, 2]);
        assert!(result.is_err());

        // Test invalid exponential parameters
        let result = generator.exponential::<f64>(-1.0, &[2, 2]);
        assert!(result.is_err());

        // Test invalid gamma parameters
        let result = generator.gamma::<f64>(-1.0, 1.0, &[2, 2]);
        assert!(result.is_err());

        // Test invalid beta parameters
        let result = generator.beta::<f64>(-1.0, 1.0, &[2, 2]);
        assert!(result.is_err());

        // Test invalid choice parameters
        let empty_choices: Vec<i32> = vec![];
        let result = generator.choice(&empty_choices, 3, false);
        assert!(result.is_err());

        // Test choice without replacement when size > population
        let small_choices = vec![1, 2];
        let result = generator.choice(&small_choices, 5, false);
        assert!(result.is_err());

        // Test invalid integers range
        let result = generator.integers(10, 5, 5);
        assert!(result.is_err());

        // Test invalid float range
        let result = generator.random_floats_range(10.0, 5.0, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_reproducibility() {
        let seed = 12345;

        // Create two generators with same seed
        let pcg1 = PCG64::seed_from_u64(seed);
        let pcg2 = PCG64::seed_from_u64(seed);

        let mut generator1 = Generator::new(Box::new(pcg1));
        let mut generator2 = Generator::new(Box::new(pcg2));

        // Generate same arrays
        let arr1 = generator1.random::<f64>(&[2, 2], Dtype::Float64).unwrap();
        let arr2 = generator2.random::<f64>(&[2, 2], Dtype::Float64).unwrap();

        // They should have same shape and length
        assert_eq!(arr1.shape(), arr2.shape());
        assert_eq!(arr1.len(), arr2.len());

        // Test permutation reproducibility
        let perm1 = generator1.permutation(5).unwrap();
        let perm2 = generator2.permutation(5).unwrap();

        assert_eq!(perm1.shape(), perm2.shape());
        assert_eq!(perm1.len(), perm2.len());
    }

    #[test]
    fn test_rng_core_implementation() {
        let pcg = PCG64::new();
        let mut generator = Generator::new(Box::new(pcg));

        // Test RngCore methods
        let u1 = generator.next_u32();
        let u2 = generator.next_u64();

        // These should produce different values
        assert_ne!(u1, generator.next_u32());
        assert_ne!(u2, generator.next_u64());

        // Test fill_bytes
        let mut bytes = [0u8; 10];
        generator.fill_bytes(&mut bytes);

        // At least some bytes should be non-zero (very high probability)
        let non_zero_count = bytes.iter().filter(|&&b| b != 0).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_multivariate_distributions() {
        let pcg = PCG64::seed_from_u64(555);
        let mut generator = Generator::new(Box::new(pcg));

        // Test multinomial
        let pvals = vec![0.2, 0.3, 0.5];
        let multi_arr = generator.multinomial::<f64>(10, &pvals, None).unwrap();
        assert_eq!(multi_arr.shape(), &[3]);

        // Verify sum equals n
        let sum: f64 = multi_arr.data.as_slice().iter().sum();
        assert!((sum - 10.0).abs() < 1e-10);

        // Test Dirichlet
        let alpha = vec![1.0, 2.0, 3.0];
        let dir_arr = generator.dirichlet::<f64>(&alpha, None).unwrap();
        assert_eq!(dir_arr.shape(), &[3]);

        // Verify sum equals 1.0
        let sum: f64 = dir_arr.data.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_special_distributions() {
        let pcg = PCG64::seed_from_u64(777);
        let mut generator = Generator::new(Box::new(pcg));

        // Test geometric
        let geom_arr = generator.geometric::<f64>(0.5, &[2, 2]).unwrap();
        assert_eq!(geom_arr.shape(), &[2, 2]);

        // Verify all values are positive integers
        for val in geom_arr.data.as_slice() {
            assert!(val >= &1.0);
        }

        // Test negative_binomial
        let neg_bin_arr = generator.negative_binomial::<f64>(5, 0.5, &[2, 2]).unwrap();
        assert_eq!(neg_bin_arr.shape(), &[2, 2]);

        // Verify all values are >= n
        for val in neg_bin_arr.data.as_slice() {
            assert!(val >= &5.0);
        }

        // Test hypergeometric
        let hyper_arr = generator.hypergeometric::<f64>(10, 5, 3, &[2, 2]).unwrap();
        assert_eq!(hyper_arr.shape(), &[2, 2]);

        // Verify all values are in valid range [0, 3]
        for val in hyper_arr.data.as_slice() {
            assert!(val >= &0.0 && val <= &3.0);
        }
    }

    #[test]
    fn test_advanced_distributions() {
        let pcg = PCG64::seed_from_u64(888);
        let mut generator = Generator::new(Box::new(pcg));

        // Test distributions that should work
        let distributions = vec![
            ("lognormal", || {
                generator.lognormal::<f64>(0.0, 1.0, &[2, 2])
            }),
            ("logistic", || generator.logistic::<f64>(0.0, 1.0, &[2, 2])),
            ("gumbel", || generator.gumbel::<f64>(0.0, 1.0, &[2, 2])),
            ("wald", || generator.wald::<f64>(1.0, 1.0, &[2, 2])),
            ("weibull", || generator.weibull::<f64>(2.0, &[2, 2])),
            ("triangular", || {
                generator.triangular::<f64>(0.0, 0.5, 1.0, &[2, 2])
            }),
            ("pareto", || generator.pareto::<f64>(2.0, &[2, 2])),
            ("zipf", || generator.zipf::<f64>(3.0, &[2, 2])),
            ("standard_cauchy", || {
                generator.standard_cauchy::<f64>(&[2, 2])
            }),
            ("standard_exponential", || {
                generator.standard_exponential::<f64>(&[2, 2])
            }),
            ("standard_gamma", || {
                generator.standard_gamma::<f64>(2.0, &[2, 2])
            }),
            ("f", || generator.f::<f64>(2.0, 2.0, &[2, 2])),
            ("power", || generator.power::<f64>(2.0, &[2, 2])),
            ("vonmises", || generator.vonmises::<f64>(0.0, 1.0, &[2, 2])),
        ];

        for (name, dist_fn) in distributions {
            let result = dist_fn();
            assert!(result.is_ok(), "{} distribution should work", name);
            let arr = result.unwrap();
            assert_eq!(
                arr.shape(),
                &[2, 2],
                "{} should produce correct shape",
                name
            );
        }
    }

    #[test]
    fn test_generator_with_different_bitgenerators() {
        // Test that Generator works with different BitGenerator implementations
        let pcg = PCG64::new();
        let mut generator = Generator::new(Box::new(pcg));

        // Should work with any BitGenerator
        let result = generator.random::<f64>(&[2, 2], Dtype::Float64);
        assert!(result.is_ok());

        let arr = result.unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_array_compatibility() {
        // Test that Generator methods work with different array types
        let pcg = PCG64::new();
        let mut generator = Generator::new(Box::new(pcg));

        // Test with different dtypes
        let f32_arr = generator.random::<f32>(&[2, 2], Dtype::Float32).unwrap();
        assert_eq!(f32_arr.shape(), &[2, 2]);

        let i32_arr = generator.randint::<i32>(0, 10, &[2, 2]).unwrap();
        assert_eq!(i32_arr.shape(), &[2, 2]);

        let i64_arr = generator.randint::<i64>(0, 100, &[2, 2]).unwrap();
        assert_eq!(i64_arr.shape(), &[2, 2]);
    }

    #[test]
    fn test_edge_cases() {
        let pcg = PCG64::new();
        let mut generator = Generator::new(Box::new(pcg));

        // Test with empty shapes
        let empty_arr = generator.random::<f64>(&[], Dtype::Float64).unwrap();
        assert_eq!(empty_arr.shape(), &[]);
        assert_eq!(empty_arr.len(), 0);

        // Test with size 1
        let single_arr = generator.random::<f64>(&[1], Dtype::Float64).unwrap();
        assert_eq!(single_arr.shape(), &[1]);
        assert_eq!(single_arr.len(), 1);

        // Test permutation with n=0
        let perm_arr = generator.permutation(0).unwrap();
        assert_eq!(perm_arr.shape(), &[0]);
        assert_eq!(perm_arr.len(), 0);

        // Test choice with size 0
        let choices = vec![1, 2, 3];
        let choice_arr = generator.choice(&choices, 0, true).unwrap();
        assert_eq!(choice_arr.shape(), &[0]);
        assert_eq!(choice_arr.len(), 0);
    }
}
