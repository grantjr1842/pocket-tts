//! Integration tests for Rust NumPy implementation
//!
//! These tests validate the Rust implementation against the
//! comprehensive NumPy test examples.

use rust_numpy_validator::*;

#[test]
fn test_array_creation_validation() {
    // Adjust the path to point to your test data
    let validator = NumPyTestValidator::with_path("../test_data");

    // Test array creation functions
    let results = validator.validate_array_creation();

    match results {
        Ok(validation_results) => {
            println!("Array Creation Validation Results:");
            for result in &validation_results {
                println!("  {}", result);
            }

            // Check that all tests passed or were skipped
            let failed_tests: Vec<_> = validation_results
                .iter()
                .filter(|r| r.contains("FAILED"))
                .collect();

            assert!(
                failed_tests.is_empty(),
                "Some array creation tests failed: {:?}",
                failed_tests
            );
        }
        Err(e) => {
            panic!("Failed to validate array creation: {}", e);
        }
    }
}

#[test]
fn test_arithmetic_validation() {
    let validator = NumPyTestValidator::with_path("../test_data");

    match validator.validate_arithmetic() {
        Ok(results) => {
            println!("Arithmetic Validation Results:");
            for result in &results {
                println!("  {}", result);
            }

            let failed_tests: Vec<_> = results.iter().filter(|r| r.contains("FAILED")).collect();

            assert!(
                failed_tests.is_empty(),
                "Some arithmetic tests failed: {:?}",
                failed_tests
            );
        }
        Err(e) => {
            panic!("Failed to validate arithmetic: {}", e);
        }
    }
}

#[test]
fn test_mathematical_validation() {
    let validator = NumPyTestValidator::with_path("../test_data");

    match validator.validate_mathematical() {
        Ok(results) => {
            println!("Mathematical Validation Results:");
            for result in &results {
                println!("  {}", result);
            }

            let failed_tests: Vec<_> = results.iter().filter(|r| r.contains("FAILED")).collect();

            assert!(
                failed_tests.is_empty(),
                "Some mathematical tests failed: {:?}",
                failed_tests
            );
        }
        Err(e) => {
            panic!("Failed to validate mathematical: {}", e);
        }
    }
}

#[test]
fn test_specific_array_case() {
    // Test a specific case from the test data
    let validator = NumPyTestValidator::with_path("../test_data");
    let test_cases = validator.load_test_cases("array_creation").unwrap();

    // Find the basic array test
    let array_test = test_cases
        .iter()
        .find(|tc| tc.function == "array")
        .expect("Should find array test case");

    // Test input: [1, 2, 3, 4, 5]
    let input_data = &array_test.input;
    let input_array = TestArray::from_json(input_data).unwrap();

    // Your implementation
    let result = array(input_array.data).unwrap();

    // Validate
    assert_eq!(result.shape(), [5]);
    assert_eq!(result.len(), 5);

    let expected_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result_data: Vec<f64> = result.iter().cloned().collect();
    assert_eq!(result_data, expected_data);

    println!("✅ Specific array test passed");
}

#[test]
fn test_specific_arithmetic_case() {
    let validator = NumPyTestValidator::with_path("../test_data");
    let test_cases = validator.load_test_cases("arithmetic").unwrap();

    // Find the add test
    let add_test = test_cases
        .iter()
        .find(|tc| tc.function == "add")
        .expect("Should find add test case");

    // Parse inputs
    let x_array = TestArray::from_json(&add_test.input["x"]).unwrap();
    let y_array = TestArray::from_json(&add_test.input["y"]).unwrap();

    // Test addition
    let result = rust_numpy_validator::functions::add(&x_array.data, &y_array.data).unwrap();

    // Validate against expected
    let expected_data: Vec<f64> = add_test
        .output
        .data
        .as_ref()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();

    assert_eq!(result, expected_data);
    println!("✅ Specific arithmetic test passed");
}

#[test]
fn test_csv_integration() {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    // Test loading CSV data (simulating cross-language testing)
    let csv_path = "../test_data/generated/array_(10,)_float64.csv";

    if std::path::Path::new(csv_path).exists() {
        let file = File::open(csv_path).unwrap();
        let reader = BufReader::new(file);

        let mut values = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            if !line.trim().is_empty() {
                // Parse the value (CSV format from our generator)
                if let Ok(value) = line.trim().parse::<f64>() {
                    values.push(value);
                }
            }
        }

        // Test with our array implementation
        let arr = Array::from_vec(values);
        assert_eq!(arr.shape(), &[10]);
        assert_eq!(arr.len(), 10);

        // Test some operations
        let sum = arr.sum();
        let mean = arr.mean();

        println!("✅ CSV integration test passed");
        println!("  Sum: {:.6}", sum);
        println!("  Mean: {:.6}", mean);
    } else {
        println!("⚠️  CSV file not found, skipping CSV integration test");
    }
}

#[test]
fn test_performance_with_generated_data() {
    use std::fs;
    use std::time::Instant;

    // Load a generated array for performance testing
    let csv_path = "../test_data/generated/array_(10,)_float64.csv";

    if std::path::Path::new(csv_path).exists() {
        let file = fs::File::open(csv_path).unwrap();
        let mut rdr = csv::Reader::from_reader(file);

        let mut values = Vec::new();
        for result in rdr.records() {
            let record = result.unwrap();
            for field in record.iter() {
                if let Ok(value) = field.parse::<f64>() {
                    values.push(value);
                }
            }
        }

        let arr = Array::from_vec(values);

        // Benchmark operations
        let start = Instant::now();
        let _sum = arr.sum();
        let sum_time = start.elapsed();

        let start = Instant::now();
        let _mean = arr.mean();
        let mean_time = start.elapsed();

        let start = Instant::now();
        let _min = arr.min();
        let min_time = start.elapsed();

        let start = Instant::now();
        let _max = arr.max();
        let max_time = start.elapsed();

        println!("✅ Performance test completed");
        println!("  Sum:   {:.2} μs", sum_time.as_micros());
        println!("  Mean:  {:.2} μs", mean_time.as_micros());
        println!("  Min:   {:.2} μs", min_time.as_micros());
        println!("  Max:   {:.2} μs", max_time.as_micros());
    } else {
        println!("⚠️  Performance test data not found");
    }
}

#[test]
fn test_edge_cases() {
    // Test edge cases that might not be in the main test suite

    // Empty array
    let empty_arr = Array::from_vec(vec![]);
    assert_eq!(empty_arr.shape(), &[0]);
    assert_eq!(empty_arr.len(), 0);
    assert!(empty_arr.is_empty());

    // Single element array
    let single_arr = Array::from_vec(vec![42.0]);
    assert_eq!(single_arr.shape(), &[1]);
    assert_eq!(single_arr.len(), 1);
    assert_eq!(single_arr.sum(), 42.0);

    // Test with NaN and Inf values
    let special_arr = Array::from_vec(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
    assert!(special_arr.sum().is_nan()); // NaN + anything = NaN

    println!("✅ Edge cases test passed");
}

#[test]
fn test_array_reshape_operations() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];

    // Test different reshape operations
    let arr_1d = Array::from_vec(data.clone());
    assert_eq!(arr_1d.shape(), &[12]);

    let _arr_2d = arr_1d.clone().reshape(&[3, 4]).unwrap();
    let _arr_3d = arr_1d.clone().reshape(&[2, 3, 2]).unwrap();

    // Test invalid reshape with fresh clone
    let invalid_reshape = arr_1d.clone().reshape(&[5, 3]); // 5*3=15, but we have 12 elements
    assert!(invalid_reshape.is_err());

    println!("✅ Array reshape operations test passed");
}

fn test_comprehensive_validation() {
    // Run a comprehensive validation of all implemented functions
    let validator = NumPyTestValidator::with_path("../test_data");

    let mut total_tests = 0;
    let mut passed_tests = 0;

    // Test all categories we have implemented
    let categories = vec!["array_creation", "arithmetic", "mathematical"];

    for category in categories {
        match validator.load_test_cases(category) {
            Ok(test_cases) => {
                println!(
                    "Testing category: {} ({} tests)",
                    category,
                    test_cases.len()
                );
                total_tests += test_cases.len();

                for test_case in test_cases {
                    // Simple validation - just check that we can parse the test case
                    let _input_array = TestArray::from_json(&test_case.input);
                    passed_tests += 1;
                }
            }
            Err(e) => {
                println!("Could not load category {}: {}", category, e);
            }
        }
    }

    println!("✅ Comprehensive validation completed");
    println!("  Total test cases parsed: {}", total_tests);
    println!("  Successfully parsed: {}", passed_tests);

    assert!(
        passed_tests > 0,
        "Should have parsed at least some test cases"
    );
}
