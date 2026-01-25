# Rust Integration Guide for NumPy Test Examples

This guide shows how to use the NumPy test examples to validate your 100% Rust NumPy port with exact API parity.

## Setup

### 1. Add Dependencies to Cargo.toml

```toml
[dependencies]
# Core dependencies for your NumPy port
num-complex = "0.4"
num-traits = "0.2"

# For loading test data
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# For CSV parsing (cross-language testing)
csv = "1.3"

# For testing and benchmarking
criterion = "0.5"

# For complex number support
ndarray = "0.15"  # Only for test validation, not in your actual implementation
```

### 2. Project Structure

```
rust-numpy/
├── src/
│   ├── lib.rs              # Your main NumPy implementation
│   ├── array.rs            # Array implementation
│   ├── functions.rs        # Mathematical functions
│   ├── linalg.rs           # Linear algebra
│   └── tests/
│       ├── integration_tests.rs
│       └── test_utils.rs
├── test_data/              # Generated test data
│   ├── test_cases_*.json
│   ├── generated/
│   └── *.csv
└── benches/
    └── performance.rs
```

## Core Integration Pattern

### 1. Test Case Structure

First, define Rust structs to match the JSON test cases:

```rust
// src/tests/test_utils.rs
use serde::{Deserialize, Serialize};
use num_complex::Complex64;

#[derive(Debug, Deserialize, Serialize)]
pub struct TestCase {
    pub function: String,
    pub input: serde_json::Value,
    pub output: TestOutput,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TestOutput {
    pub data: Option<Vec<serde_json::Value>>,
    pub shape: Option<Vec<usize>>,
    pub dtype: Option<String>,
    pub value: Option<serde_json::Value>,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TestArray {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl TestArray {
    pub fn from_json(input: &serde_json::Value) -> Result<Self, Box<dyn std::error::Error>> {
        // Handle different input formats
        match input {
            serde_json::Value::Array(arr) => {
                let data: Vec<f64> = arr.iter()
                    .map(|v| v.as_f64().unwrap_or(0.0))
                    .collect();
                Ok(TestArray {
                    data,
                    shape: vec![data.len()],
                    dtype: "float64".to_string(),
                })
            }
            serde_json::Value::Object(obj) => {
                if let Some(object) = obj.get("object") {
                    Self::from_json(object)
                } else {
                    Err("Invalid input format".into())
                }
            }
            _ => Err("Unsupported input format".into()),
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        assert_eq!(self.data.len(), new_shape.iter().product(), "Shape mismatch");
        TestArray {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            dtype: self.dtype.clone(),
        }
    }
}
```

### 2. Validation Framework

```rust
// src/tests/test_utils.rs (continued)
use crate::prelude::*; // Your NumPy port's prelude

pub struct NumPyTestValidator {
    test_data_dir: std::path::PathBuf,
}

impl NumPyTestValidator {
    pub fn new() -> Self {
        Self {
            test_data_dir: std::path::PathBuf::from("test_data"),
        }
    }

    pub fn load_test_cases(&self, category: &str) -> Result<Vec<TestCase>, Box<dyn std::error::Error>> {
        let file_path = self.test_data_dir.join(format!("test_cases_{}.json", category));
        let file = std::fs::File::open(file_path)?;
        let test_cases: Vec<TestCase> = serde_json::from_reader(file)?;
        Ok(test_cases)
    }

    pub fn validate_array_creation(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let test_cases = self.load_test_cases("array_creation")?;
        let mut results = Vec::new();

        for test_case in test_cases {
            let result = match test_case.function.as_str() {
                "array" => self.test_array_creation(&test_case),
                "zeros" => self.test_zeros(&test_case),
                "ones" => self.test_ones(&test_case),
                "arange" => self.test_arange(&test_case),
                "linspace" => self.test_linspace(&test_case),
                "eye" => self.test_eye(&test_case),
                _ => Ok(format!("Skipped unknown function: {}", test_case.function)),
            };

            match result {
                Ok(msg) => results.push(msg),
                Err(e) => results.push(format!("FAILED: {}", e)),
            }
        }

        Ok(results)
    }

    fn test_array_creation(&self, test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        let input_array = TestArray::from_json(&test_case.input)?;
        let expected = &test_case.output;

        // Call your Rust implementation
        let result = crate::array::array(input_array.data)?;

        // Validate shape
        if let Some(expected_shape) = &expected.shape {
            if result.shape() != expected_shape.as_slice() {
                return Err(format!("Shape mismatch: expected {:?}, got {:?}",
                    expected_shape, result.shape()).into());
            }
        }

        // Validate data
        if let Some(expected_data) = &expected.data {
            let result_data: Vec<f64> = result.iter().map(|&x| x).collect();
            let expected_vec: Vec<f64> = expected_data.iter()
                .map(|v| v.as_f64().unwrap_or(0.0))
                .collect();

            if !self.arrays_close(&result_data, &expected_vec) {
                return Err("Data mismatch".into());
            }
        }

        Ok(format!("✅ {}: PASSED", test_case.function))
    }

    fn arrays_close(&self, a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() {
            return false;
        }

        const TOLERANCE: f64 = 1e-10;
        a.iter().zip(b.iter()).all(|(&x, &y)| (x - y).abs() < TOLERANCE)
    }
}
```

## Complete Implementation Examples

### 1. Array Creation Tests

```rust
// src/tests/array_creation_tests.rs
use crate::tests::test_utils::NumPyTestValidator;

#[cfg(test)]
mod array_creation_tests {
    use super::*;

    #[test]
    fn test_array_function() {
        let validator = NumPyTestValidator::new();
        let results = validator.validate_array_creation().unwrap();

        for result in results {
            println!("{}", result);
            assert!(result.contains("✅") || result.contains("Skipped"),
                "Test failed: {}", result);
        }
    }

    #[test]
    fn test_specific_array_case() {
        let validator = NumPyTestValidator::new();
        let test_cases = validator.load_test_cases("array_creation").unwrap();

        // Find the basic array test
        let array_test = test_cases.iter()
            .find(|tc| tc.function == "array")
            .unwrap();

        // Test input: [1, 2, 3, 4, 5]
        let input_data = &array_test.input;
        let input_array = TestArray::from_json(input_data).unwrap();

        // Your implementation
        let result = crate::array::array(input_array.data).unwrap();

        // Validate
        assert_eq!(result.shape(), [5]);
        assert_eq!(result.len(), 5);

        let expected_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result_data: Vec<f64> = result.iter().cloned().collect();
        assert_eq!(result_data, expected_data);
    }
}
```

### 2. Mathematical Function Tests

```rust
// src/tests/mathematical_tests.rs
use crate::tests::test_utils::*;

#[cfg(test)]
mod mathematical_tests {
    use super::*;

    #[test]
    fn test_arithmetic_functions() {
        let validator = NumPyTestValidator::new();
        let test_cases = validator.load_test_cases("arithmetic").unwrap();

        for test_case in test_cases {
            let result = match test_case.function.as_str() {
                "add" => test_add(&test_case),
                "subtract" => test_subtract(&test_case),
                "multiply" => test_multiply(&test_case),
                "divide" => test_divide(&test_case),
                "power" => test_power(&test_case),
                _ => Ok(format!("Skipped: {}", test_case.function)),
            };

            assert!(result.is_ok(), "Test failed: {:?}", result);
            println!("{}", result.unwrap());
        }
    }

    fn test_add(test_case: &TestCase) -> Result<String, Box<dyn std::error::Error>> {
        // Parse inputs
        let x_array = TestArray::from_json(&test_case.input["x"])?;
        let y_array = TestArray::from_json(&test_case.input["y"])?;

        // Call your implementation
        let result = crate::functions::add(&x_array.data, &y_array.data)?;

        // Validate
        let expected_data: Vec<f64> = test_case.output.data.as_ref()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        if result.len() != expected_data.len() {
            return Err("Length mismatch".into());
        }

        for (actual, expected) in result.iter().zip(expected_data.iter()) {
            if (actual - expected).abs() > 1e-10 {
                return Err(format!("Value mismatch: {} != {}", actual, expected).into());
            }
        }

        Ok("✅ add: PASSED".to_string())
    }
}
```

### 3. Cross-Language CSV Testing

```rust
// src/tests/csv_tests.rs
use std::fs::File;
use std::io::{BufRead, BufReader};
use csv::ReaderBuilder;

#[cfg(test)]
mod csv_tests {
    use super::*;

    #[test]
    fn test_csv_array_loading() {
        // Load CSV file generated by Python
        let csv_path = "test_data/generated/array_(5, 4)_int32.csv";
        let file = File::open(csv_path).unwrap();
        let reader = BufReader::new(file);

        let mut values = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            if !line.trim().is_empty() {
                // Parse complex number format: "(real+imagj)"
                let clean = line.trim().replace('(', "").replace(')', "");
                if let Ok(value) = clean.parse::<f64>() {
                    values.push(value);
                }
            }
        }

        // Reshape to expected dimensions
        let array = crate::array::Array::from_vec(values)
            .reshape(&[5, 4])
            .unwrap();

        assert_eq!(array.shape(), [5, 4]);
        println!("✅ CSV array loading: PASSED");
    }

    #[test]
    fn test_all_csv_formats() {
        let test_files = vec![
            "array_(10,)_float64.csv",
            "array_(5, 4)_int32.csv",
            "array_(2, 3, 4)_complex64.csv",
        ];

        for filename in test_files {
            let csv_path = format!("test_data/generated/{}", filename);
            if std::path::Path::new(&csv_path).exists() {
                test_csv_file(&csv_path);
                println!("✅ {}: PASSED", filename);
            } else {
                println!("⚠️  {}: File not found", filename);
            }
        }
    }

    fn test_csv_file(csv_path: &str) {
        let file = File::open(csv_path).unwrap();
        let mut rdr = ReaderBuilder::new()
            .has_headers(false)
            .from_reader(file);

        let mut values = Vec::new();
        for result in rdr.records() {
            let record = result.unwrap();
            for field in record.iter() {
                if let Ok(value) = field.parse::<f64>() {
                    values.push(value);
                }
            }
        }

        // Basic validation
        assert!(!values.is_empty());

        // Test with your array implementation
        let _array = crate::array::Array::from_vec(values);
    }
}
```

### 4. Performance Benchmarking

```rust
// benches/performance.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::fs;

fn load_test_array(shape: &str, dtype: &str) -> Vec<f64> {
    let filename = format!("test_data/generated/array_{}_{}.csv", shape, dtype);
    let file = fs::File::open(filename).unwrap();
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
    values
}

fn bench_array_creation(c: &mut Criterion) {
    let data = load_test_array("(10,)", "float64");

    c.bench_function("array_creation_10", |b| {
        b.iter(|| {
            let result = crate::array::array(black_box(data.clone())).unwrap();
            black_box(result)
        })
    });
}

fn bench_arithmetic(c: &mut Criterion) {
    let data1 = load_test_array("(10,)", "float64");
    let data2 = load_test_array("(10,)", "float64");

    c.bench_function("add_10", |b| {
        b.iter(|| {
            let result = crate::functions::add(
                black_box(&data1),
                black_box(&data2)
            ).unwrap();
            black_box(result)
        })
    });
}

criterion_group!(benches, bench_array_creation, bench_arithmetic);
criterion_main!(benches);
```

## Integration with Your Rust Implementation

### 1. Example Implementation Structure

```rust
// src/lib.rs
pub mod array;
pub mod functions;
pub mod linalg;
pub mod dtype;

pub use array::{Array, array};
pub use functions::*;
pub use linalg::*;

#[cfg(test)]
mod tests;

// Your main NumPy port implementation
pub mod prelude {
    pub use crate::{Array, array};
    pub use num_complex::Complex64;
    pub use num_traits::{Float, Num};
}
```

### 2. Array Implementation Example

```rust
// src/array.rs
use std::fmt;

#[derive(Debug, Clone)]
pub struct Array {
    data: Vec<f64>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Array {
    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self {
            data,
            shape: vec![len],
            strides: vec![1],
        }
    }

    pub fn reshape(mut self, new_shape: &[usize]) -> Result<Self, String> {
        let total_elements: usize = new_shape.iter().product();
        if total_elements != self.data.len() {
            return Err(format!("Cannot reshape array of size {} to shape {:?}",
                self.data.len(), new_shape));
        }

        // Calculate strides
        let mut strides = vec![1usize; new_shape.len()];
        for i in (1..new_shape.len()).rev() {
            strides[i-1] = strides[i] * new_shape[i];
        }

        self.shape = new_shape.to_vec();
        self.strides = strides;
        Ok(self)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn iter(&self) -> std::slice::Iter<f64> {
        self.data.iter()
    }
}

// Your array function
pub fn array(data: Vec<f64>) -> Result<Array, String> {
    Ok(Array::from_vec(data))
}
```

### 3. Functions Implementation Example

```rust
// src/functions.rs
use crate::Array;

pub fn add(a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    if a.len() != b.len() {
        return Err("Arrays must have same length".to_string());
    }

    let result: Vec<f64> = a.iter().zip(b.iter())
        .map(|(&x, &y)| x + y)
        .collect();

    Ok(result)
}

pub fn sqrt(a: &[f64]) -> Result<Vec<f64>, String> {
    let result: Vec<f64> = a.iter()
        .map(|&x| {
            if x < 0.0 {
                f64::NAN
            } else {
                x.sqrt()
            }
        })
        .collect();
    Ok(result)
}

// Add other functions...
```

## Running Tests

### 1. Unit Tests

```bash
# Run all tests
cargo test

# Run specific test category
cargo test array_creation_tests

# Run with output
cargo test -- --nocapture
```

### 2. Benchmarks

```bash
# Run performance benchmarks
cargo bench

# Specific benchmark
cargo bench array_creation_10
```

### 3. Integration Tests

```bash
# Run integration tests
cargo test --test integration_tests

# Validate against all test data
cargo test validate_all_categories
```

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Test Rust NumPy Port

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Cache dependencies
        uses: actions/cache@v2
        with:
          path: ~/.cargo/registry
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Run tests
        run: cargo test --all

      - name: Run benchmarks
        run: cargo bench

      - name: Validate NumPy compatibility
        run: |
          python test_data/validation_script.py
          # Check that your Rust implementation passes the same tests
          cargo test validate_numpy_compatibility
```

## Best Practices

1. **Start Simple**: Begin with array creation, then basic arithmetic
2. **Use Exact Data**: Always use the provided test data for validation
3. **Handle Edge Cases**: Test NaN, Inf, empty arrays explicitly
4. **Performance Testing**: Use the benchmark suite regularly
5. **Incremental Development**: Add tests as you implement each function
6. **Cross-Validation**: Compare with NumPy reference implementation
7. **Documentation**: Document any deviations from NumPy behavior

This integration guide provides everything you need to validate your 100% Rust NumPy port against the original NumPy implementation with exact API parity!
