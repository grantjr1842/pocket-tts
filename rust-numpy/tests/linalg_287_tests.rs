#[cfg(test)]
mod tests {
    use rust_numpy::*;

    #[test]
    fn test_sign_functions() {
        // Test sign function with various values
        let x = Array::from_data(vec![-2.0f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0f64], vec![7]);
        let result = sign(&x).unwrap();
        let expected = vec![-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0];
        assert_eq!(result.size(), 7);
        for i in 0..7 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }

    #[test]
    fn test_signbit_function() {
        // Test signbit function with various values
        // Note: use -0.0f64 explicitly to avoid optimization to 0.0
        let x = Array::from_data(
            vec![-2.0f64, -1.0f64, -0.0f64, 0.0f64, 0.5f64, 1.0f64, 2.0f64],
            vec![7],
        );
        let result = signbit(&x).unwrap();
        // expected: [-2.0 (T), -1.0 (T), -0.0 (T), 0.0 (F), 0.5 (F), 1.0 (F), 2.0 (F)]
        let expected = vec![true, true, true, false, false, false, false];
        assert_eq!(result.size(), 7);
        for i in 0..7 {
            assert_eq!(result.get(i), Some(&expected[i]), "Failed at index {}", i);
        }
    }

    #[test]
    fn test_copysign_function() {
        // Test copysign function: result has magnitude of x1 and sign of x2
        let x1 = Array::from_data(vec![1.0, -2.0, 3.0, -4.0f64], vec![4]);
        let x2 = Array::from_data(vec![1.0, -1.0, 1.0, -1.0f64], vec![4]);
        let result = copysign(&x1, &x2).unwrap();
        // x1: [ 1, -2,  3, -4]
        // x2: [ 1, -1,  1, -1]
        // res: [ 1, -2,  3, -4]
        let expected = vec![1.0, -2.0, 3.0, -4.0f64];
        assert_eq!(result.size(), 4);
        for i in 0..4 {
            assert_eq!(result.get(i), Some(&expected[i]), "Failed at index {}", i);
        }
    }

    #[test]
    fn test_fabs_function() {
        // Test fabs function (float only)
        let x = Array::from_data(
            vec![-2.5f32, -1.0f32, -0.5f32, 0.0f32, 0.5f32, 1.0f32, 2.0f32],
            vec![7],
        );
        let result = fabs(&x).unwrap();
        let expected = vec![2.5, 1.0, 0.5, 0.0, 0.5, 1.0, 2.0f32];
        assert_eq!(result.size(), 7);
        for i in 0..7 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }

    #[test]
    fn test_absolute_function() {
        // Test absolute function (alias for abs)
        let x = Array::from_data(vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], vec![7]);
        let result = absolute(&x).unwrap();
        let expected = vec![2.0, 1.0, 0.5, 0.0, 0.5, 1.0, 2.0];
        assert_eq!(result.size(), 7);
        for i in 0..7 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }

    #[test]
    fn test_sign_edge_cases() {
        let x = Array::from_data(vec![-0.0, 0.0f64], vec![2]);
        let result = sign(&x).unwrap();
        assert_eq!(result.size(), 2);
        assert_eq!(result.get(0), Some(&0.0f64));
        assert_eq!(result.get(1), Some(&0.0f64));
    }

    #[test]
    fn test_sign_with_integers() {
        // Test sign with integer types
        let x = Array::from_data(vec![-5i64, -1i64, 0i64, 1i64, 5i64], vec![5]);
        let result = sign(&x).unwrap();
        let expected = vec![-1i64, -1i64, 0i64, 1i64, 1i64];
        assert_eq!(result.size(), 5);
        for i in 0..5 {
            assert_eq!(result.get(i), Some(&expected[i]));
        }
    }
}
