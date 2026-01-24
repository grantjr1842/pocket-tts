# How to Use the NumPy Test Examples

This guide shows you practical ways to use the comprehensive NumPy test examples we've generated from the original NumPy source code.

## Quick Start

### 1. Basic Validation
```python
import numpy as np
import json

# Load test cases
with open('test_data/test_cases_array_creation.json', 'r') as f:
    test_cases = json.load(f)

# Test array creation
test_case = test_cases[0]
result = np.array(test_case['input']['object'])
expected = np.array(test_case['output']['data'])

assert np.array_equal(result, expected), "Test failed!"
print("✅ Array creation test passed!")
```

### 2. Load Generated Test Arrays
```python
import numpy as np

# Load a pre-generated test array
arr = np.load('test_data/generated/array_(10,)_float64.npy')
print(f"Array shape: {arr.shape}")
print(f"Array values: {arr}")

# Use for testing mathematical functions
sqrt_result = np.sqrt(np.abs(arr))
print(f"Square root results: {sqrt_result}")
```

## Detailed Usage Patterns

### Pattern 1: Function Validation
Use the JSON test cases to validate your implementation:

```python
def validate_numpy_function(test_case_file, function_name):
    """Validate a specific NumPy function against test cases"""
    with open(test_case_file, 'r') as f:
        test_cases = json.load(f)

    for test_case in test_cases:
        if test_case['function'] == function_name:
            # Execute test
            input_data = test_case['input']
            expected = test_case['output']

            # Your implementation here
            result = your_implementation(input_data)

            # Validate
            assert np.allclose(result, expected['data']), f"Failed on {input_data}"
            print(f"✅ {function_name} passed")

# Usage
validate_numpy_function('test_data/test_cases_arithmetic.json', 'add')
```

### Pattern 2: Cross-Language Testing
Use CSV files for testing in other languages (Rust, C++, etc.):

```python
# Python side - generate test data
import numpy as np

# Create test array
test_array = np.array([1.5, -2.3, 4.7, 0.0, -1.1])
np.savetxt('test_data.csv', test_array, delimiter=',')

# Rust side - read and test
/*
// Rust code would read the CSV and validate
let mut rdr = csv::Reader::from_path("test_data.csv")?;
let test_data: Vec<f64> = rdr.deserialize().collect::<Result<_, _>>()?;

// Use test_data to validate your Rust NumPy implementation
let result = rust_numpy::array(test_data);
assert_eq!(result.shape(), [5]);
*/
```

### Pattern 3: Performance Benchmarking
Use the generated arrays to benchmark performance:

```python
import time
import numpy as np

def benchmark_function(func, arr, iterations=1000):
    """Benchmark a function with test data"""
    start_time = time.time()
    for _ in range(iterations):
        result = func(arr)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations * 1000  # ms
    return avg_time

# Load different sized arrays
sizes = ['(10,)', '(5, 4)', '(2, 3, 4)']
for size in sizes:
    arr = np.load(f'test_data/generated/array_{size}_float64.npy')

    # Benchmark matrix multiplication
    if arr.ndim == 2:
        time_ms = benchmark_function(lambda x: np.dot(x, x.T), arr)
        print(f"Size {size}: {time_ms:.2f} ms")
```

### Pattern 4: Edge Case Testing
Test special values and edge cases:

```python
def test_edge_cases():
    """Test NaN, Inf, and other edge cases"""
    # Load edge case tests
    with open('test_data/test_cases_edge_cases.json', 'r') as f:
        edge_cases = json.load(f)

    for test_case in edge_cases:
        func_name = test_case['function']
        input_data = test_case['input']
        expected = test_case['output']

        if func_name == 'isnan':
            x = np.array(input_data['x'])
            result = np.isnan(x)
            expected_bool = np.array(expected['data'])

            assert np.array_equal(result, expected_bool), "NaN test failed"
            print(f"✅ {func_name} edge case passed")

test_edge_cases()
```

### Pattern 5: Rust Integration Workflow
Simulate how Rust would use the test data:

```python
# This is what your Rust validation code would do:
def rust_validation_simulation():
    """Simulate Rust using the test data for validation"""

    # Step 1: Load test specification
    with open('test_data/test_cases_array_creation.json', 'r') as f:
        test_cases = json.load(f)

    test_case = test_cases[1]  # 2D array test

    # Step 2: Parse input (as Rust would)
    input_object = test_case['input']['object']
    print(f"Rust would parse: {input_object}")
    print(f"Target shape: {test_case['output']['shape']}")

    # Step 3: Validation criteria for Rust
    print("Rust implementation must:")
    print("1. Parse nested list into 2D array")
    print("2. Verify shape is [2, 3]")
    print("3. Verify dtype matches 'int64'")
    print("4. Verify all data values are correct")

rust_validation_simulation()
```

## File-by-File Guide

### JSON Test Case Files
Each file contains test cases for a specific category:

- `test_cases_array_creation.json` - Array creation functions
- `test_cases_arithmetic.json` - Basic arithmetic operations
- `test_cases_mathematical.json` - Mathematical functions
- `test_cases_statistical.json` - Statistical functions
- `test_cases_manipulation.json` - Array manipulation
- `test_cases_comparison.json` - Comparison operations
- `test_cases_linear_algebra.json` - Linear algebra functions
- `test_cases_edge_cases.json` - Edge cases and special values

### Generated Array Files
In the `generated/` directory:

- **NumPy format (.npy)** - Load with `np.load()`
- **CSV format (.csv)** - Read as text for any language
- **Naming convention**: `array_(shape)_(dtype).format`

### Example Test Case Structure
```json
{
  "function": "add",
  "input": {
    "x": [1, 2, 3, 4],
    "y": [5, 6, 7, 8]
  },
  "output": {
    "data": [6, 8, 10, 12],
    "shape": [4],
    "dtype": "int64"
  }
}
```

## Integration Examples

### For Rust NumPy Implementation
```rust
// Rust pseudocode showing how to use test data
use serde_json;
use numpy_rust as np;

fn validate_array_creation() -> Result<(), Box<dyn std::error::Error>> {
    // Load test case
    let test_file = std::fs::File::open("test_data/test_cases_array_creation.json")?;
    let test_cases: serde_json::Value = serde_json::from_reader(test_file)?;

    // Get first test case
    let test_case = &test_cases[0];
    let input_data = &test_case["input"]["object"];
    let expected_output = &test_case["output"];

    // Convert JSON to Rust arrays
    let input_vec: Vec<i64> = serde_json::from_value(input_data.clone())?;
    let result = np::array(input_vec)?;

    // Validate
    assert_eq!(result.shape(), expected_output["shape"]);
    assert_eq!(result.dtype(), expected_output["dtype"]);

    Ok(())
}
```

### For Continuous Integration
```yaml
# GitHub Actions example
- name: Validate NumPy Implementation
  run: |
    python test_data/validation_script.py

    # Check validation results
    python -c "
    import json
    with open('test_data/validation_report.json') as f:
        report = json.load(f)
    assert report['summary']['success_rate'] > 90, 'Too many tests failed'
    "
```

### For Development Testing
```python
# Quick development validation
def quick_validate():
    """Quick validation during development"""
    import json
    import numpy as np

    # Test a few key functions
    key_tests = [
        ('test_cases_array_creation.json', 'array'),
        ('test_cases_arithmetic.json', 'add'),
        ('test_cases_mathematical.json', 'sqrt'),
    ]

    for test_file, func_name in key_tests:
        with open(f'test_data/{test_file}') as f:
            test_cases = json.load(f)

        # Find test case for this function
        test_case = next(tc for tc in test_cases if tc['function'] == func_name)

        # Quick validation
        if func_name == 'array':
            result = np.array(test_case['input']['object'])
        elif func_name == 'add':
            x, y = map(np.array, [test_case['input']['x'], test_case['input']['y']])
            result = np.add(x, y)

        print(f"✅ {func_name} quick validation passed")

quick_validate()
```

## Best Practices

1. **Start Simple**: Begin with basic array creation tests
2. **Use Exact Data**: Use the provided exact input/output pairs
3. **Validate Edge Cases**: Don't forget NaN, Inf, and empty arrays
4. **Cross-Language**: Use CSV files for non-Python languages
5. **Automate**: Integrate validation into your CI/CD pipeline
6. **Performance**: Use generated arrays for benchmarking
7. **Documentation**: Keep track of which tests your implementation passes

## Troubleshooting

### Common Issues
- **Shape mismatches**: Check array dimensions carefully
- **Type differences**: Ensure dtypes match exactly
- **Floating-point precision**: Use `np.allclose()` for float comparisons
- **File paths**: Use absolute paths or ensure correct working directory

### Debug Tips
```python
# Detailed test debugging
def debug_test_case(test_case):
    """Debug a failing test case"""
    print(f"Function: {test_case['function']}")
    print(f"Input: {test_case['input']}")
    print(f"Expected: {test_case['output']}")

    # Execute with NumPy reference
    if test_case['function'] == 'add':
        result = np.add(
            np.array(test_case['input']['x']),
            np.array(test_case['input']['y'])
        )

    print(f"NumPy result: {result}")
    print(f"Expected data: {test_case['output']['data']}")
    print(f"Match: {np.allclose(result, test_case['output']['data'])}")
```

## Next Steps

1. **Run the examples**: Execute `usage_examples.py` to see everything in action
2. **Pick a category**: Start with one function category (e.g., array creation)
3. **Implement and validate**: Build your implementation and test against the data
4. **Expand coverage**: Add more test categories as you implement more functions
5. **Automate**: Set up continuous validation in your development workflow

The test examples provide a solid foundation for ensuring your NumPy implementation has 100% API parity with the original!
