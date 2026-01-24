//! Simple tests that don't require external test data
//! These demonstrate the basic functionality of the Rust NumPy implementation

use rust_numpy_validator::*;

#[test]
fn test_basic_array_creation() {
    // Test basic array creation
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let arr = array(data).unwrap();

    assert_eq!(arr.shape(), &[5]);
    assert_eq!(arr.len(), 5);
    assert_eq!(arr.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

    println!("✅ Basic array creation test passed");
}

#[test]
fn test_array_operations() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = Array::from_vec(data);

    // Test reshape
    let arr_2d = arr.reshape(&[2, 3]).unwrap();
    assert_eq!(arr_2d.shape(), &[2, 3]);

    // Test transpose
    let transposed = arr_2d.transpose().unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);

    // Test flatten
    let flattened = transposed.flatten();
    assert_eq!(flattened.shape(), &[6]);

    println!("✅ Array operations test passed");
}

#[test]
fn test_array_creation_functions() {
    // Test zeros
    let zeros = Array::zeros(&[3, 4]);
    assert_eq!(zeros.shape(), &[3, 4]);
    assert_eq!(zeros.sum(), 0.0);

    // Test ones
    let ones = Array::ones(&[2, 3]);
    assert_eq!(ones.shape(), &[2, 3]);
    assert_eq!(ones.sum(), 6.0);

    // Test eye
    let eye = Array::eye(3);
    assert_eq!(eye.shape(), &[3, 3]);
    assert_eq!(eye.sum(), 3.0);

    // Test arange
    let arange = Array::arange(0, 10, 2);
    assert_eq!(arange.shape(), &[5]);
    assert_eq!(arange.data(), &[0.0, 2.0, 4.0, 6.0, 8.0]);

    // Test linspace
    let linspace = Array::linspace(0.0, 1.0, 5);
    assert_eq!(linspace.shape(), &[5]);
    assert!(linspace.all_close(&Array::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]), 1e-10));

    println!("✅ Array creation functions test passed");
}

#[test]
fn test_arithmetic_functions() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    // Test add
    let add_result = rust_numpy_validator::functions::add(&a, &b).unwrap();
    assert_eq!(add_result, vec![6.0, 8.0, 10.0, 12.0]);

    // Test subtract
    let sub_result = rust_numpy_validator::functions::subtract(&b, &a).unwrap();
    assert_eq!(sub_result, vec![4.0, 4.0, 4.0, 4.0]);

    // Test multiply
    let mul_result = rust_numpy_validator::functions::multiply(&a, &b).unwrap();
    assert_eq!(mul_result, vec![5.0, 12.0, 21.0, 32.0]);

    // Test divide
    let div_result = rust_numpy_validator::functions::divide(&b, &a).unwrap();
    assert_eq!(div_result, vec![5.0, 3.0, 2.3333333333333335, 2.0]);

    // Test power
    let pow_result = rust_numpy_validator::functions::power(&a, 2.0).unwrap();
    assert_eq!(pow_result, vec![1.0, 4.0, 9.0, 16.0]);

    println!("✅ Arithmetic functions test passed");
}

#[test]
fn test_mathematical_functions() {
    let a = vec![1.0, 4.0, 9.0, 16.0];

    // Test sqrt
    let sqrt_result = rust_numpy_validator::functions::sqrt(&a).unwrap();
    assert_eq!(sqrt_result, vec![1.0, 2.0, 3.0, 4.0]);

    // Test abs
    let abs_data = vec![-1.0, -2.0, 3.0, -4.0];
    let abs_result = rust_numpy_validator::functions::abs(&abs_data).unwrap();
    assert_eq!(abs_result, vec![1.0, 2.0, 3.0, 4.0]);

    // Test sin
    let sin_data = vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI];
    let sin_result = rust_numpy_validator::functions::sin(&sin_data).unwrap();
    assert!((sin_result[0] - 0.0).abs() < 1e-10);
    assert!((sin_result[1] - 1.0).abs() < 1e-10);
    assert!((sin_result[2] - 0.0).abs() < 1e-10);

    // Test cos
    let cos_result = rust_numpy_validator::functions::cos(&sin_data).unwrap();
    assert!((cos_result[0] - 1.0).abs() < 1e-10);
    assert!((cos_result[1] - 0.0).abs() < 1e-10);
    assert!((cos_result[2] + 1.0).abs() < 1e-10);

    // Test exp
    let exp_data = vec![0.0, 1.0, 2.0];
    let exp_result = rust_numpy_validator::functions::exp(&exp_data).unwrap();
    assert!((exp_result[0] - 1.0).abs() < 1e-10);
    assert!((exp_result[1] - std::f64::consts::E).abs() < 1e-10);
    assert!((exp_result[2] - std::f64::consts::E * std::f64::consts::E).abs() < 1e-10);

    // Test log
    let log_data = vec![
        1.0,
        std::f64::consts::E,
        std::f64::consts::E * std::f64::consts::E,
    ];
    let log_result = rust_numpy_validator::functions::log(&log_data).unwrap();
    assert!((log_result[0] - 0.0).abs() < 1e-10);
    assert!((log_result[1] - 1.0).abs() < 1e-10);
    assert!((log_result[2] - 2.0).abs() < 1e-10);

    println!("✅ Mathematical functions test passed");
}

#[test]
fn test_statistical_functions() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let arr = Array::from_vec(data);

    // Test statistical operations
    assert_eq!(arr.sum(), 55.0);
    assert_eq!(arr.mean(), 5.5);
    assert_eq!(arr.min(), 1.0);
    assert_eq!(arr.max(), 10.0);
    assert_eq!(arr.argmin(), 0);
    assert_eq!(arr.argmax(), 9);

    println!("✅ Statistical functions test passed");
}

#[test]
fn test_edge_cases() {
    // Test empty array
    let empty_arr = Array::from_vec(vec![]);
    assert_eq!(empty_arr.shape(), &[0]);
    assert_eq!(empty_arr.len(), 0);
    assert!(empty_arr.is_empty());

    // Test single element
    let single_arr = Array::from_vec(vec![42.0]);
    assert_eq!(single_arr.shape(), &[1]);
    assert_eq!(single_arr.len(), 1);
    assert_eq!(single_arr.sum(), 42.0);

    // Test special values
    let special_arr = Array::from_vec(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
    assert!(special_arr.sum().is_nan()); // NaN + anything = NaN

    println!("✅ Edge cases test passed");
}

#[test]
fn test_numpy_compatibility_examples() {
    // These examples mirror the ones from the Python usage examples

    // Example 1: Basic array creation (like np.array([1, 2, 3, 4, 5]))
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = array(input).unwrap();
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(result.data(), &expected);

    // Example 2: Mathematical operations (like np.sqrt([1, 4, 9, 16]))
    let math_input = vec![1.0, 4.0, 9.0, 16.0];
    let sqrt_result = rust_numpy_validator::functions::sqrt(&math_input).unwrap();
    let sqrt_expected = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(sqrt_result, sqrt_expected);

    // Example 3: Arithmetic operations (like np.add([1, 2], [3, 4]))
    let x = vec![1.0, 2.0];
    let y = vec![3.0, 4.0];
    let add_result = rust_numpy_validator::functions::add(&x, &y).unwrap();
    let add_expected = vec![4.0, 6.0];
    assert_eq!(add_result, add_expected);

    println!("✅ NumPy compatibility examples test passed");
}

#[test]
fn test_performance_basics() {
    use std::time::Instant;

    let data = vec![1.0; 1000]; // 1000 elements
    let arr = Array::from_vec(data);

    // Benchmark sum operation
    let start = Instant::now();
    let _sum = arr.sum();
    let sum_time = start.elapsed();

    // Benchmark mean operation
    let start = Instant::now();
    let _mean = arr.mean();
    let mean_time = start.elapsed();

    // Benchmark min/max operations
    let start = Instant::now();
    let _min = arr.min();
    let _max = arr.max();
    let minmax_time = start.elapsed();

    println!("✅ Performance basics test completed");
    println!("  Sum:   {:.2} μs", sum_time.as_micros());
    println!("  Mean:  {:.2} μs", mean_time.as_micros());
    println!("  Min/Max: {:.2} μs", minmax_time.as_micros());

    // Ensure operations are reasonable (should be very fast)
    assert!(sum_time.as_micros() < 1000, "Sum operation too slow");
    assert!(mean_time.as_micros() < 1000, "Mean operation too slow");
    assert!(
        minmax_time.as_micros() < 1000,
        "Min/Max operations too slow"
    );
}
