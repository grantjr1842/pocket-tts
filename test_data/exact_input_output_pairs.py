#!/usr/bin/env python3
"""
Exact Input/Output Pairs for NumPy Functions
Generated from original NumPy source tests
Provides precise test cases for Rust implementation validation
"""

import numpy as np
import json


class ExactInputOutputPairs:
    """Generate exact input/output pairs for NumPy function testing"""

    def __init__(self):
        self.test_cases = {}

    def generate_basic_array_tests(self):
        """Generate exact input/output pairs for basic array operations"""
        test_cases = {}

        # Test array creation with exact values
        test_cases["array_creation"] = [
            {
                "function": "array",
                "input": {"object": [1, 2, 3, 4, 5]},
                "output": {"data": [1, 2, 3, 4, 5], "shape": [5], "dtype": "int64"},
            },
            {
                "function": "array",
                "input": {"object": [[1, 2, 3], [4, 5, 6]]},
                "output": {
                    "data": [1, 2, 3, 4, 5, 6],
                    "shape": [2, 3],
                    "dtype": "int64",
                },
            },
            {
                "function": "zeros",
                "input": {"shape": (3, 4)},
                "output": {"shape": [3, 4], "dtype": "float64", "all_zeros": True},
            },
            {
                "function": "ones",
                "input": {"shape": (2, 3)},
                "output": {"shape": [2, 3], "dtype": "float64", "all_ones": True},
            },
            {
                "function": "arange",
                "input": {"start": 0, "stop": 10, "step": 2},
                "output": {"data": [0, 2, 4, 6, 8], "shape": [5], "dtype": "int64"},
            },
            {
                "function": "linspace",
                "input": {"start": 0, "stop": 1, "num": 5},
                "output": {
                    "data": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "shape": [5],
                    "dtype": "float64",
                },
            },
            {
                "function": "eye",
                "input": {"N": 3},
                "output": {
                    "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "shape": [3, 3],
                    "dtype": "float64",
                },
            },
        ]

        return test_cases

    def generate_mathematical_tests(self):
        """Generate exact input/output pairs for mathematical functions"""
        test_cases = {}

        # Basic arithmetic
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])

        test_cases["arithmetic"] = [
            {
                "function": "add",
                "input": {"x": a.tolist(), "y": b.tolist()},
                "output": {"data": [6, 8, 10, 12], "shape": [4], "dtype": "int64"},
            },
            {
                "function": "subtract",
                "input": {"x": b.tolist(), "y": a.tolist()},
                "output": {"data": [4, 4, 4, 4], "shape": [4], "dtype": "int64"},
            },
            {
                "function": "multiply",
                "input": {"x": a.tolist(), "y": b.tolist()},
                "output": {"data": [5, 12, 21, 32], "shape": [4], "dtype": "int64"},
            },
            {
                "function": "divide",
                "input": {"x": b.tolist(), "y": a.tolist()},
                "output": {
                    "data": [5.0, 3.0, 2.3333333333333335, 2.0],
                    "shape": [4],
                    "dtype": "float64",
                },
            },
            {
                "function": "power",
                "input": {"x": a.tolist(), "y": 2},
                "output": {"data": [1, 4, 9, 16], "shape": [4], "dtype": "int64"},
            },
        ]

        # Mathematical functions
        test_cases["mathematical"] = [
            {
                "function": "sqrt",
                "input": {"x": [1, 4, 9, 16]},
                "output": {
                    "data": [1.0, 2.0, 3.0, 4.0],
                    "shape": [4],
                    "dtype": "float64",
                },
            },
            {
                "function": "abs",
                "input": {"x": [-1, -2, 3, -4]},
                "output": {"data": [1, 2, 3, 4], "shape": [4], "dtype": "int64"},
            },
            {
                "function": "sin",
                "input": {"x": [0, np.pi / 2, np.pi]},
                "output": {"data": [0.0, 1.0, 0.0], "shape": [3], "dtype": "float64"},
            },
            {
                "function": "cos",
                "input": {"x": [0, np.pi / 2, np.pi]},
                "output": {"data": [1.0, 0.0, -1.0], "shape": [3], "dtype": "float64"},
            },
            {
                "function": "exp",
                "input": {"x": [0, 1, 2]},
                "output": {
                    "data": [1.0, 2.718281828459045, 7.38905609893065],
                    "shape": [3],
                    "dtype": "float64",
                },
            },
            {
                "function": "log",
                "input": {"x": [1, np.e, np.e**2]},
                "output": {"data": [0.0, 1.0, 2.0], "shape": [3], "dtype": "float64"},
            },
        ]

        return test_cases

    def generate_statistical_tests(self):
        """Generate exact input/output pairs for statistical functions"""
        test_cases = {}

        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        test_cases["statistical"] = [
            {"function": "sum", "input": {"a": arr.tolist()}, "output": {"value": 55}},
            {
                "function": "mean",
                "input": {"a": arr.tolist()},
                "output": {"value": 5.5},
            },
            {
                "function": "median",
                "input": {"a": arr.tolist()},
                "output": {"value": 5.5},
            },
            {
                "function": "std",
                "input": {"a": arr.tolist()},
                "output": {"value": 3.0276503540974917},
            },
            {
                "function": "var",
                "input": {"a": arr.tolist()},
                "output": {"value": 9.166666666666666},
            },
            {"function": "min", "input": {"a": arr.tolist()}, "output": {"value": 1}},
            {"function": "max", "input": {"a": arr.tolist()}, "output": {"value": 10}},
            {
                "function": "argmin",
                "input": {"a": arr.tolist()},
                "output": {"value": 0},
            },
            {
                "function": "argmax",
                "input": {"a": arr.tolist()},
                "output": {"value": 9},
            },
        ]

        return test_cases

    def generate_manipulation_tests(self):
        """Generate exact input/output pairs for array manipulation"""
        test_cases = {}

        arr = np.array([[1, 2, 3], [4, 5, 6]])

        test_cases["manipulation"] = [
            {
                "function": "reshape",
                "input": {"a": arr.tolist(), "newshape": (3, 2)},
                "output": {
                    "data": [1, 2, 3, 4, 5, 6],
                    "shape": [3, 2],
                    "dtype": "int64",
                },
            },
            {
                "function": "transpose",
                "input": {"a": arr.tolist()},
                "output": {
                    "data": [1, 4, 2, 5, 3, 6],
                    "shape": [3, 2],
                    "dtype": "int64",
                },
            },
            {
                "function": "flatten",
                "input": {"a": arr.tolist()},
                "output": {"data": [1, 2, 3, 4, 5, 6], "shape": [6], "dtype": "int64"},
            },
            {
                "function": "concatenate",
                "input": {"arrays": [[1, 2], [3, 4]]},
                "output": {"data": [1, 2, 3, 4], "shape": [4], "dtype": "int64"},
            },
            {
                "function": "stack",
                "input": {"arrays": [[1, 2], [3, 4]]},
                "output": {"data": [1, 2, 3, 4], "shape": [2, 2], "dtype": "int64"},
            },
        ]

        return test_cases

    def generate_comparison_tests(self):
        """Generate exact input/output pairs for comparison operations"""
        test_cases = {}

        a = np.array([1, 2, 3, 4])
        b = np.array([2, 2, 2, 2])

        test_cases["comparison"] = [
            {
                "function": "equal",
                "input": {"x": a.tolist(), "y": b.tolist()},
                "output": {
                    "data": [False, True, False, False],
                    "shape": [4],
                    "dtype": "bool",
                },
            },
            {
                "function": "not_equal",
                "input": {"x": a.tolist(), "y": b.tolist()},
                "output": {
                    "data": [True, False, True, True],
                    "shape": [4],
                    "dtype": "bool",
                },
            },
            {
                "function": "greater",
                "input": {"x": a.tolist(), "y": b.tolist()},
                "output": {
                    "data": [False, False, True, True],
                    "shape": [4],
                    "dtype": "bool",
                },
            },
            {
                "function": "less",
                "input": {"x": a.tolist(), "y": b.tolist()},
                "output": {
                    "data": [True, False, False, False],
                    "shape": [4],
                    "dtype": "bool",
                },
            },
        ]

        return test_cases

    def generate_linear_algebra_tests(self):
        """Generate exact input/output pairs for linear algebra functions"""
        test_cases = {}

        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([1, 2])

        test_cases["linear_algebra"] = [
            {
                "function": "dot",
                "input": {"a": vector.tolist(), "b": vector.tolist()},
                "output": {"value": 5},
            },
            {
                "function": "matmul",
                "input": {"a": matrix.tolist(), "b": vector.tolist()},
                "output": {"data": [5, 11], "shape": [2], "dtype": "int64"},
            },
            {
                "function": "det",
                "input": {"a": matrix.tolist()},
                "output": {"value": -2.0},
            },
            {
                "function": "inv",
                "input": {"a": matrix.tolist()},
                "output": {
                    "data": [-2.0, 1.0, 1.5, -0.5],
                    "shape": [2, 2],
                    "dtype": "float64",
                },
            },
        ]

        return test_cases

    def generate_edge_case_tests(self):
        """Generate exact input/output pairs for edge cases"""
        test_cases = {}

        test_cases["edge_cases"] = [
            {
                "function": "isnan",
                "input": {"x": [1, 2, np.nan, 4]},
                "output": {
                    "data": [False, False, True, False],
                    "shape": [4],
                    "dtype": "bool",
                },
            },
            {
                "function": "isinf",
                "input": {"x": [1, np.inf, -np.inf, 4]},
                "output": {
                    "data": [False, True, True, False],
                    "shape": [4],
                    "dtype": "bool",
                },
            },
            {
                "function": "empty_array",
                "input": {},
                "output": {"data": [], "shape": [0]},
            },
        ]

        return test_cases

    def generate_all_test_cases(self):
        """Generate all test cases"""
        all_tests = {}

        all_tests.update(self.generate_basic_array_tests())
        all_tests.update(self.generate_mathematical_tests())
        all_tests.update(self.generate_statistical_tests())
        all_tests.update(self.generate_manipulation_tests())
        all_tests.update(self.generate_comparison_tests())
        all_tests.update(self.generate_linear_algebra_tests())
        all_tests.update(self.generate_edge_case_tests())

        return all_tests

    def save_test_cases(self, filename):
        """Save test cases to JSON file"""
        test_cases = self.generate_all_test_cases()

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        converted_cases = convert_numpy_types(test_cases)

        with open(filename, "w") as f:
            json.dump(converted_cases, f, indent=2)

        return converted_cases


def main():
    """Main function to generate exact input/output pairs"""
    print("Generating exact input/output pairs for NumPy functions...")

    generator = ExactInputOutputPairs()

    # Generate and save test cases
    test_cases = generator.save_test_cases(
        "/home/grant/github/pocket-tts/test_data/exact_test_cases.json"
    )

    print(f"Generated test cases for {len(test_cases)} categories")

    # Print summary
    total_functions = sum(
        len(category) if isinstance(category, list) else 1
        for category in test_cases.values()
    )
    print(f"Total test functions: {total_functions}")

    # Save individual category files for easier testing
    for category, cases in test_cases.items():
        filename = f"/home/grant/github/pocket-tts/test_data/test_cases_{category}.json"
        with open(filename, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Saved {category} test cases to {filename}")

    return test_cases


if __name__ == "__main__":
    main()
