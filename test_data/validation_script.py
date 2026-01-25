#!/usr/bin/env python3
"""
Validation Script for NumPy Test Examples
Validates generated test cases against NumPy reference implementation
"""

import numpy as np
import json
import traceback


class NumPyTestValidator:
    """Validate test cases against NumPy reference implementation"""

    def __init__(self):
        self.validation_results = {}
        self.passed = 0
        self.failed = 0

    def load_test_cases(self, filepath):
        """Load test cases from JSON file"""
        with open(filepath, "r") as f:
            return json.load(f)

    def validate_array_creation(self, test_cases):
        """Validate array creation test cases"""
        results = []

        for case in test_cases:
            function = case["function"]
            input_data = case["input"]
            expected = case["output"]

            try:
                if function == "array":
                    result = np.array(input_data["object"])
                elif function == "zeros":
                    result = np.zeros(input_data["shape"])
                elif function == "ones":
                    result = np.ones(input_data["shape"])
                elif function == "arange":
                    if "step" in input_data:
                        result = np.arange(
                            input_data["start"], input_data["stop"], input_data["step"]
                        )
                    else:
                        result = np.arange(input_data["start"], input_data["stop"])
                elif function == "linspace":
                    result = np.linspace(
                        input_data["start"], input_data["stop"], input_data["num"]
                    )
                elif function == "eye":
                    result = np.eye(input_data["N"])
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "skipped",
                            "reason": f"Unknown function: {function}",
                        }
                    )
                    continue

                # Validate result
                if "data" in expected:
                    if np.allclose(result.flatten(), expected["data"]):
                        results.append(
                            {
                                "function": function,
                                "status": "passed",
                                "input": input_data,
                                "expected_shape": expected["shape"],
                                "actual_shape": list(result.shape),
                                "expected_dtype": expected["dtype"],
                                "actual_dtype": str(result.dtype),
                            }
                        )
                        self.passed += 1
                    else:
                        results.append(
                            {
                                "function": function,
                                "status": "failed",
                                "input": input_data,
                                "expected": expected["data"],
                                "actual": result.flatten().tolist(),
                                "error": "Data mismatch",
                            }
                        )
                        self.failed += 1
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "passed",
                            "input": input_data,
                            "expected_shape": expected["shape"],
                            "actual_shape": list(result.shape),
                        }
                    )
                    self.passed += 1

            except Exception as e:
                results.append(
                    {
                        "function": function,
                        "status": "error",
                        "input": input_data,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
                self.failed += 1

        return results

    def validate_arithmetic(self, test_cases):
        """Validate arithmetic test cases"""
        results = []

        for case in test_cases:
            function = case["function"]
            input_data = case["input"]
            expected = case["output"]

            try:
                x = np.array(input_data["x"])
                y = np.array(input_data["y"]) if "y" in input_data else None

                if function == "add":
                    result = np.add(x, y)
                elif function == "subtract":
                    result = np.subtract(x, y)
                elif function == "multiply":
                    result = np.multiply(x, y)
                elif function == "divide":
                    result = np.divide(x, y)
                elif function == "power":
                    result = np.power(x, input_data["y"])
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "skipped",
                            "reason": f"Unknown function: {function}",
                        }
                    )
                    continue

                if np.allclose(result.flatten(), expected["data"]):
                    results.append(
                        {
                            "function": function,
                            "status": "passed",
                            "input": input_data,
                            "expected_shape": expected["shape"],
                            "actual_shape": list(result.shape),
                            "expected_dtype": expected["dtype"],
                            "actual_dtype": str(result.dtype),
                        }
                    )
                    self.passed += 1
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "failed",
                            "input": input_data,
                            "expected": expected["data"],
                            "actual": result.flatten().tolist(),
                            "error": "Data mismatch",
                        }
                    )
                    self.failed += 1

            except Exception as e:
                results.append(
                    {
                        "function": function,
                        "status": "error",
                        "input": input_data,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
                self.failed += 1

        return results

    def validate_mathematical(self, test_cases):
        """Validate mathematical function test cases"""
        results = []

        for case in test_cases:
            function = case["function"]
            input_data = case["input"]
            expected = case["output"]

            try:
                x = np.array(input_data["x"])

                if function == "sqrt":
                    result = np.sqrt(x)
                elif function == "abs":
                    result = np.abs(x)
                elif function == "sin":
                    result = np.sin(x)
                elif function == "cos":
                    result = np.cos(x)
                elif function == "exp":
                    result = np.exp(x)
                elif function == "log":
                    result = np.log(x)
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "skipped",
                            "reason": f"Unknown function: {function}",
                        }
                    )
                    continue

                if np.allclose(result.flatten(), expected["data"], rtol=1e-10):
                    results.append(
                        {
                            "function": function,
                            "status": "passed",
                            "input": input_data,
                            "expected_shape": expected["shape"],
                            "actual_shape": list(result.shape),
                            "expected_dtype": expected["dtype"],
                            "actual_dtype": str(result.dtype),
                        }
                    )
                    self.passed += 1
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "failed",
                            "input": input_data,
                            "expected": expected["data"],
                            "actual": result.flatten().tolist(),
                            "error": "Data mismatch",
                        }
                    )
                    self.failed += 1

            except Exception as e:
                results.append(
                    {
                        "function": function,
                        "status": "error",
                        "input": input_data,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
                self.failed += 1

        return results

    def validate_statistical(self, test_cases):
        """Validate statistical function test cases"""
        results = []

        for case in test_cases:
            function = case["function"]
            input_data = case["input"]
            expected = case["output"]

            try:
                a = np.array(input_data["a"])

                if function == "sum":
                    result = np.sum(a)
                elif function == "mean":
                    result = np.mean(a)
                elif function == "median":
                    result = np.median(a)
                elif function == "std":
                    result = np.std(a)
                elif function == "var":
                    result = np.var(a)
                elif function == "min":
                    result = np.min(a)
                elif function == "max":
                    result = np.max(a)
                elif function == "argmin":
                    result = np.argmin(a)
                elif function == "argmax":
                    result = np.argmax(a)
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "skipped",
                            "reason": f"Unknown function: {function}",
                        }
                    )
                    continue

                if np.isclose(result, expected["value"]):
                    results.append(
                        {
                            "function": function,
                            "status": "passed",
                            "input": input_data,
                            "expected": expected["value"],
                            "actual": float(result),
                        }
                    )
                    self.passed += 1
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "failed",
                            "input": input_data,
                            "expected": expected["value"],
                            "actual": float(result),
                            "error": "Value mismatch",
                        }
                    )
                    self.failed += 1

            except Exception as e:
                results.append(
                    {
                        "function": function,
                        "status": "error",
                        "input": input_data,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
                self.failed += 1

        return results

    def validate_comparison(self, test_cases):
        """Validate comparison operation test cases"""
        results = []

        for case in test_cases:
            function = case["function"]
            input_data = case["input"]
            expected = case["output"]

            try:
                x = np.array(input_data["x"])
                y = np.array(input_data["y"])

                if function == "equal":
                    result = np.equal(x, y)
                elif function == "not_equal":
                    result = np.not_equal(x, y)
                elif function == "greater":
                    result = np.greater(x, y)
                elif function == "less":
                    result = np.less(x, y)
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "skipped",
                            "reason": f"Unknown function: {function}",
                        }
                    )
                    continue

                if np.array_equal(result, expected["data"]):
                    results.append(
                        {
                            "function": function,
                            "status": "passed",
                            "input": input_data,
                            "expected_shape": expected["shape"],
                            "actual_shape": list(result.shape),
                            "expected_dtype": expected["dtype"],
                            "actual_dtype": str(result.dtype),
                        }
                    )
                    self.passed += 1
                else:
                    results.append(
                        {
                            "function": function,
                            "status": "failed",
                            "input": input_data,
                            "expected": expected["data"],
                            "actual": result.tolist(),
                            "error": "Data mismatch",
                        }
                    )
                    self.failed += 1

            except Exception as e:
                results.append(
                    {
                        "function": function,
                        "status": "error",
                        "input": input_data,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )
                self.failed += 1

        return results

    def validate_all_categories(self, test_file):
        """Validate all test categories"""
        test_cases = self.load_test_cases(test_file)
        all_results = {}

        for category, cases in test_cases.items():
            print(f"Validating {category}...")

            if category == "array_creation":
                results = self.validate_array_creation(cases)
            elif category == "arithmetic":
                results = self.validate_arithmetic(cases)
            elif category == "mathematical":
                results = self.validate_mathematical(cases)
            elif category == "statistical":
                results = self.validate_statistical(cases)
            elif category == "comparison":
                results = self.validate_comparison(cases)
            else:
                results = [
                    {
                        "function": "unknown",
                        "status": "skipped",
                        "reason": f"Unknown category: {category}",
                    }
                ]

            all_results[category] = results

        return all_results

    def generate_report(self, results, output_file):
        """Generate validation report"""
        report = {
            "summary": {
                "total_tests": self.passed + self.failed,
                "passed": self.passed,
                "failed": self.failed,
                "success_rate": self.passed / (self.passed + self.failed) * 100
                if (self.passed + self.failed) > 0
                else 0,
            },
            "results": results,
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        return report


def main():
    """Main validation function"""
    print("Validating NumPy test examples against reference implementation...")

    validator = NumPyTestValidator()

    # Validate all test cases
    results = validator.validate_all_categories(
        "/home/grant/github/pocket-tts/test_data/exact_test_cases.json"
    )

    # Generate report
    report = validator.generate_report(
        results, "/home/grant/github/pocket-tts/test_data/validation_report.json"
    )

    # Print summary
    print("\nValidation Summary:")
    print(f"Total tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Success rate: {report['summary']['success_rate']:.2f}%")

    # Print failed tests
    for category, test_results in results.items():
        failed_tests = [r for r in test_results if r["status"] == "failed"]
        if failed_tests:
            print(f"\nFailed tests in {category}:")
            for test in failed_tests:
                print(f"  - {test['function']}: {test.get('error', 'Unknown error')}")

    return report


if __name__ == "__main__":
    main()
