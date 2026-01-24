#!/usr/bin/env python3
"""
Usage Examples for NumPy Test Data
Demonstrates how to use the generated test examples for various purposes
"""

import numpy as np
import json
import os
from pathlib import Path


class NumPyTestDataUsage:
    """Examples of how to use the generated NumPy test data"""

    def __init__(self):
        self.test_data_dir = Path("/home/grant/github/pocket-tts/test_data")

    def example_1_basic_validation(self):
        """Example 1: Basic function validation using JSON test cases"""
        print("=== Example 1: Basic Function Validation ===")

        # Load array creation test cases
        with open(self.test_data_dir / "test_cases_array_creation.json", "r") as f:
            test_cases = json.load(f)

        # Test the 'array' function
        array_test = test_cases[0]  # First test case should be array creation
        print(f"Testing function: {array_test['function']}")
        print(f"Input: {array_test['input']}")

        # Execute the function with NumPy
        result = np.array(array_test["input"]["object"])
        expected = np.array(array_test["output"]["data"])

        # Validate
        if np.array_equal(result, expected):
            print("✅ PASSED: NumPy array creation works correctly")
        else:
            print("❌ FAILED: Results don't match")
            print(f"Expected: {expected}")
            print(f"Got: {result}")

        print()

    def example_2_load_generated_arrays(self):
        """Example 2: Loading and using generated test arrays"""
        print("=== Example 2: Loading Generated Test Arrays ===")

        # Load a generated array
        array_path = self.test_data_dir / "generated" / "array_(10,)_float64.npy"
        arr = np.load(array_path)

        print(f"Loaded array shape: {arr.shape}")
        print(f"Array dtype: {arr.dtype}")
        print(f"Array values: {arr}")

        # Use the array for testing mathematical functions
        sqrt_result = np.sqrt(np.abs(arr))  # Take sqrt of absolute values
        print(f"Square root of absolute values: {sqrt_result}")

        # Test statistical functions
        print(f"Mean: {np.mean(arr):.6f}")
        print(f"Std: {np.std(arr):.6f}")
        print(f"Min: {np.min(arr):.6f}")
        print(f"Max: {np.max(arr):.6f}")

        print()

    def example_3_cross_language_testing(self):
        """Example 3: Cross-language testing using CSV format"""
        print("=== Example 3: Cross-Language Testing with CSV ===")

        # Load CSV data (simulating how another language would read it)
        csv_path = self.test_data_dir / "generated" / "array_(5, 4)_int32.csv"

        # Read CSV as text (simulating file reading in any language)
        with open(csv_path, "r") as f:
            lines = f.readlines()

        # Parse CSV values (this is what you'd do in Rust, C++, etc.)
        values = []
        for line in lines:
            # Remove parentheses and convert to float
            clean_line = line.strip().replace("(", "").replace(")", "")
            if clean_line:
                try:
                    values.append(float(clean_line))
                except ValueError:
                    continue

        # Reshape to expected dimensions (5x4 = 20 values needed)
        if len(values) >= 20:
            arr = np.array(values[:20]).reshape(5, 4)
            print(f"Loaded CSV data as {arr.shape} array")
            print(f"Data type: {arr.dtype}")
            print(f"Sample values:\n{arr[:2, :2]}")  # Show 2x2 subset
        else:
            print(f"Not enough values in CSV: got {len(values)}, need 20")
            # Use a smaller array for demonstration
            arr = np.array(values).reshape(len(values), 1)
            print(f"Using smaller array: {arr.shape}")

        print()

    def example_4_comprehensive_function_testing(self):
        """Example 4: Comprehensive testing of multiple functions"""
        print("=== Example 4: Comprehensive Function Testing ===")

        # Load arithmetic test cases
        with open(self.test_data_dir / "test_cases_arithmetic.json", "r") as f:
            arithmetic_tests = json.load(f)

        # Test all arithmetic functions
        for test_case in arithmetic_tests:
            func_name = test_case["function"]
            input_data = test_case["input"]
            expected = test_case["output"]["data"]

            # Execute the function
            x = np.array(input_data["x"])
            y = np.array(input_data["y"])

            if func_name == "add":
                result = np.add(x, y)
            elif func_name == "subtract":
                result = np.subtract(x, y)
            elif func_name == "multiply":
                result = np.multiply(x, y)
            elif func_name == "divide":
                result = np.divide(x, y)
            elif func_name == "power":
                result = np.power(x, input_data["y"])
            else:
                continue

            # Validate
            if np.allclose(result, expected):
                print(f"✅ {func_name}: PASSED")
            else:
                print(f"❌ {func_name}: FAILED")
                print(f"   Expected: {expected}")
                print(f"   Got: {result.tolist()}")

        print()

    def example_5_edge_case_testing(self):
        """Example 5: Testing edge cases and special values"""
        print("=== Example 5: Edge Case Testing ===")

        # Load edge case test cases
        with open(self.test_data_dir / "test_cases_edge_cases.json", "r") as f:
            edge_cases = json.load(f)

        for test_case in edge_cases:
            func_name = test_case["function"]
            input_data = test_case["input"]
            expected = test_case["output"]

            if func_name == "isnan":
                x = np.array(input_data["x"])
                result = np.isnan(x)
                expected_bool = np.array(expected["data"])

                if np.array_equal(result, expected_bool):
                    print(f"✅ {func_name}: Correctly identified NaN values")
                else:
                    print(f"❌ {func_name}: Failed to identify NaN")
                    print(f"   Input: {x}")
                    print(f"   Expected: {expected_bool}")
                    print(f"   Got: {result}")

            elif func_name == "isinf":
                x = np.array(input_data["x"])
                result = np.isinf(x)
                expected_bool = np.array(expected["data"])

                if np.array_equal(result, expected_bool):
                    print(f"✅ {func_name}: Correctly identified Inf values")
                else:
                    print(f"❌ {func_name}: Failed to identify Inf")

        print()

    def example_6_performance_benchmarking(self):
        """Example 6: Using test data for performance benchmarking"""
        print("=== Example 6: Performance Benchmarking ===")

        import time

        # Load different sized arrays for benchmarking
        sizes = ["(10,)", "(5, 4)", "(2, 3, 4)"]

        for size_str in sizes:
            # Load array
            filename = f"array_{size_str}_float64.npy"
            array_path = self.test_data_dir / "generated" / filename
            arr = np.load(array_path)

            # Benchmark matrix multiplication (if 2D)
            if arr.ndim == 2:
                start_time = time.time()
                result = np.dot(arr, arr.T)  # Matrix multiplication with transpose
                end_time = time.time()

                print(
                    f"Size {size_str}: Matrix multiplication took {(end_time - start_time) * 1000:.2f} ms"
                )
                print(f"  Result shape: {result.shape}")

            # Benchmark statistical operations
            start_time = time.time()
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            end_time = time.time()

            print(
                f"Size {size_str}: Statistical ops took {(end_time - start_time) * 1000:.2f} ms"
            )
            print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}")

        print()

    def example_7_rust_integration_simulation(self):
        """Example 7: Simulating Rust integration workflow"""
        print("=== Example 7: Rust Integration Simulation ===")

        # This simulates how a Rust implementation would use the test data

        # Step 1: Load test case definition
        with open(self.test_data_dir / "test_cases_array_creation.json", "r") as f:
            test_cases = json.load(f)

        test_case = test_cases[1]  # 2D array creation
        print(f"Rust Test Case: {test_case['function']}")
        print(f"Input JSON: {json.dumps(test_case['input'], indent=2)}")

        # Step 2: Parse input data (as Rust would)
        input_object = test_case["input"]["object"]
        print(f"Parsed input shape: ({len(input_object)}, {len(input_object[0])})")

        # Step 3: Expected output specification
        expected_output = test_case["output"]
        print(f"Expected output specification:")
        print(f"  Shape: {expected_output['shape']}")
        print(f"  Dtype: {expected_output['dtype']}")
        print(f"  Data preview: {expected_output['data'][:6]}...")

        # Step 4: Validation criteria
        print("Validation criteria for Rust implementation:")
        print("  1. Create array from nested list")
        print("  2. Verify shape matches [2, 3]")
        print("  3. Verify dtype is 'int64'")
        print("  4. Verify data matches expected values exactly")

        print()

    def example_8_custom_test_generation(self):
        """Example 8: Generating custom test cases"""
        print("=== Example 8: Custom Test Case Generation ===")

        # Create custom test case for a specific scenario
        custom_test = {
            "function": "custom_reshape_test",
            "description": "Test reshaping with different dimensions",
            "input": {
                "array": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "new_shape": (3, 4),
            },
            "output": {
                "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "shape": [3, 4],
                "dtype": "int64",
            },
        }

        # Execute the test
        arr = np.array(custom_test["input"]["array"])
        reshaped = np.reshape(arr, custom_test["input"]["new_shape"])

        print(f"Custom test: {custom_test['description']}")
        print(f"Input array: {custom_test['input']['array']}")
        print(f"Target shape: {custom_test['input']['new_shape']}")
        print(f"Result shape: {reshaped.shape}")
        print(f"Result data:\n{reshaped}")

        # Validate against expected
        expected_data = np.array(custom_test["output"]["data"]).reshape(3, 4)
        if np.array_equal(reshaped, expected_data):
            print("✅ Custom test PASSED")
        else:
            print("❌ Custom test FAILED")

        print()

    def run_all_examples(self):
        """Run all usage examples"""
        print("NumPy Test Data Usage Examples")
        print("=" * 50)
        print()

        self.example_1_basic_validation()
        self.example_2_load_generated_arrays()
        self.example_3_cross_language_testing()
        self.example_4_comprehensive_function_testing()
        self.example_5_edge_case_testing()
        self.example_6_performance_benchmarking()
        self.example_7_rust_integration_simulation()
        self.example_8_custom_test_generation()

        print("All examples completed!")


def main():
    """Main function to run all usage examples"""
    usage = NumPyTestDataUsage()
    usage.run_all_examples()


if __name__ == "__main__":
    main()
