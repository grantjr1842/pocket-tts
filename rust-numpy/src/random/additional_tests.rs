//! Additional tests for the new random distribution functions

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;

    fn create_test_rng() -> generator::Generator {
        generator::Generator::new(Box::new(bit_generator::PCG64::seed_from_u64(42)))
    }

    #[test]
    fn test_geometric_distribution() {
        let mut rng = create_test_rng();
        let result = rng.geometric::<f64>(0.5, &[100]).unwrap();
        
        // Geometric should produce positive integers
        for val in result.iter() {
            assert!(*val >= 0.0);
        }
        
        // Test with invalid probability
        assert!(rng.geometric::<f64>(0.0, &[10]).is_err());
        assert!(rng.geometric::<f64>(1.5, &[10]).is_err());
    }

    #[test]
    fn test_cauchy_distribution() {
        let mut rng = create_test_rng();
        let result = rng.cauchy::<f64>(0.0, 1.0, &[100]).unwrap();
        
        // Cauchy should produce a wide range of values
        let mut sum = 0.0;
        let mut count = 0;
        for val in result.iter() {
            sum += *val;
            count += 1;
        }
        
        // Should have some reasonable values (not all zeros or infinities)
        assert!(count > 0);
        assert!(sum.is_finite());
        
        // Test with invalid scale
        assert!(rng.cauchy::<f64>(0.0, 0.0, &[10]).is_err());
        assert!(rng.cauchy::<f64>(0.0, -1.0, &[10]).is_err());
    }

    #[test]
    fn test_pareto_distribution() {
        let mut rng = create_test_rng();
        let result = rng.pareto::<f64>(1.0, &[100]).unwrap();
        
        // Pareto should produce values >= 1.0
        for val in result.iter() {
            assert!(*val >= 1.0);
        }
        
        // Test with invalid shape parameter
        assert!(rng.pareto::<f64>(0.0, &[10]).is_err());
        assert!(rng.pareto::<f64>(-1.0, &[10]).is_err());
    }

    #[test]
    fn test_power_distribution() {
        let mut rng = create_test_rng();
        let result = rng.power::<f64>(2.0, &[100]).unwrap();
        
        // Power distribution should produce values in [0, 1]
        for val in result.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
        
        // Test with invalid shape parameter
        assert!(rng.power::<f64>(0.0, &[10]).is_err());
        assert!(rng.power::<f64>(-1.0, &[10]).is_err());
    }

    #[test]
    fn test_rayleigh_distribution() {
        let mut rng = create_test_rng();
        let result = rng.rayleigh::<f64>(1.0, &[100]).unwrap();
        
        // Rayleigh should produce positive values
        for val in result.iter() {
            assert!(*val >= 0.0);
        }
        
        // Test with invalid scale
        assert!(rng.rayleigh::<f64>(0.0, &[10]).is_err());
        assert!(rng.rayleigh::<f64>(-1.0, &[10]).is_err());
    }

    #[test]
    fn test_triangular_distribution() {
        let mut rng = create_test_rng();
        let result = rng.triangular::<f64>(0.0, 0.5, 1.0, &[100]).unwrap();
        
        // Triangular should produce values in [left, right]
        for val in result.iter() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
        
        // Test with invalid parameters
        assert!(rng.triangular::<f64>(1.0, 0.5, 0.0, &[10]).is_err());
        assert!(rng.triangular::<f64>(0.0, 1.5, 1.0, &[10]).is_err());
        assert!(rng.triangular::<f64>(0.0, -0.5, 1.0, &[10]).is_err());
    }

    #[test]
    fn test_weibull_distribution() {
        let mut rng = create_test_rng();
        let result = rng.weibull::<f64>(2.0, &[100]).unwrap();
        
        // Weibull should produce positive values
        for val in result.iter() {
            assert!(*val >= 0.0);
        }
        
        // Test with invalid shape parameter
        assert!(rng.weibull::<f64>(0.0, &[10]).is_err());
        assert!(rng.weibull::<f64>(-1.0, &[10]).is_err());
    }

    #[test]
    fn test_choice_function() {
        let mut rng = create_test_rng();
        let arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        
        // Test with replacement
        let result = rng.choice(&arr, 10, true).unwrap();
        assert_eq!(result.size(), 10);
        
        // Test without replacement
        let result = rng.choice(&arr, 3, false).unwrap();
        assert_eq!(result.size(), 3);
        
        // Test error cases
        let empty_arr = Array::<i32>::from_vec(vec![]);
        assert!(rng.choice(&empty_arr, 5, true).is_err());
        
        // Too many samples without replacement
        assert!(rng.choice(&arr, 10, false).is_err());
    }

    #[test]
    fn test_bytes_function() {
        let mut rng = create_test_rng();
        let result = rng.bytes(100).unwrap();
        
        assert_eq!(result.size(), 100);
        // Bytes should be in range [0, 255]
        for byte in result.iter() {
            assert!(*byte <= 255);
        }
    }

    #[test]
    fn test_permutation_function() {
        let mut rng = create_test_rng();
        let result = rng.permutation(5).unwrap();
        
        assert_eq!(result.size(), 5);
        
        // Should contain all numbers from 0 to 4 exactly once
        let mut values = vec![false; 5];
        for val in result.iter() {
            assert!(*val < 5);
            values[*val] = true;
        }
        
        for val in values {
            assert!(val);
        }
    }

    #[test]
    fn test_shuffle_function() {
        let mut rng = create_test_rng();
        let mut arr = Array::from_vec(vec![1, 2, 3, 4, 5]);
        let original = arr.clone();
        
        let result = rng.shuffle(&mut arr);
        assert!(result.is_ok());
        
        // Should still have same elements
        assert_eq!(arr.size(), 5);
        
        // Elements should be the same (just reordered)
        let mut original_sorted = original.iter().cloned().collect::<Vec<_>>();
        original_sorted.sort();
        
        let mut shuffled_sorted = arr.iter().cloned().collect::<Vec<_>>();
        shuffled_sorted.sort();
        
        assert_eq!(original_sorted, shuffled_sorted);
    }

    #[test]
    fn test_module_level_functions() {
        // Test that module-level functions work
        let result = geometric::<f64>(0.5, &[10]).unwrap();
        assert_eq!(result.size(), 10);
        
        let result = cauchy::<f64>(0.0, 1.0, &[10]).unwrap();
        assert_eq!(result.size(), 10);
        
        let result = bytes(10).unwrap();
        assert_eq!(result.size(), 10);
        
        let result = permutation(5).unwrap();
        assert_eq!(result.size(), 5);
    }

    #[test]
    fn test_distributions_statistical_properties() {
        let mut rng = create_test_rng();
        let size = 10000;
        
        // Test normal distribution mean and std
        let normal_result = rng.normal::<f64>(0.0, 1.0, &[size]).unwrap();
        let mean: f64 = normal_result.iter().sum();
        let mean = mean / size as f64;
        
        // Mean should be close to 0 (within reasonable tolerance)
        assert!((mean - 0.0).abs() < 0.1);
        
        // Test uniform distribution range
        let uniform_result = rng.uniform::<f64>(0.0, 1.0, &[size]).unwrap();
        for val in uniform_result.iter() {
            assert!(*val >= 0.0 && *val < 1.0);
        }
    }

    #[test]
    fn test_error_handling() {
        let mut rng = create_test_rng();
        
        // Test various error conditions
        assert!(rng.geometric::<f64>(-0.1, &[10]).is_err());
        assert!(rng.cauchy::<f64>(0.0, -1.0, &[10]).is_err());
        assert!(rng.pareto::<f64>(0.0, &[10]).is_err());
        assert!(rng.power::<f64>(-1.0, &[10]).is_err());
        assert!(rng.rayleigh::<f64>(-1.0, &[10]).is_err());
        assert!(rng.weibull::<f64>(0.0, &[10]).is_err());
    }
}
