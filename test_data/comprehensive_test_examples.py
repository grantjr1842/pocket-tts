#!/usr/bin/env python3
"""
Comprehensive NumPy Test Examples
Generated from original NumPy source tests
Provides exact input/output pairs for all major NumPy functions
"""

import numpy as np


class NumPyTestExamples:
    """Comprehensive test examples for NumPy functions based on original source tests"""

    def __init__(self):
        self.test_results = {}

    # Array Creation Tests
    def test_array_creation(self):
        """Test array creation functions"""
        examples = {
            "array_basic": {
                "input": ([1, 2, 3, 4, 5],),
                "output": np.array([1, 2, 3, 4, 5]),
                "dtype": "int64",
            },
            "array_2d": {
                "input": ([[1, 2, 3], [4, 5, 6]],),
                "output": np.array([[1, 2, 3], [4, 5, 6]]),
                "shape": (2, 3),
            },
            "zeros": {
                "input": ((3, 4),),
                "output": np.zeros((3, 4)),
                "shape": (3, 4),
                "dtype": "float64",
            },
            "ones": {
                "input": ((2, 3),),
                "output": np.ones((2, 3)),
                "shape": (2, 3),
                "dtype": "float64",
            },
            "empty": {"input": ((2, 2),), "output": np.empty((2, 2)), "shape": (2, 2)},
            "arange": {
                "input": (0, 10, 2),
                "output": np.arange(0, 10, 2),
                "result": [0, 2, 4, 6, 8],
            },
            "linspace": {
                "input": (0, 1, 5),
                "output": np.linspace(0, 1, 5),
                "result": [0.0, 0.25, 0.5, 0.75, 1.0],
            },
            "eye": {
                "input": ((3,),),
                "output": np.eye(3),
                "shape": (3, 3),
                "diagonal": 1.0,
            },
            "identity": {"input": ((4,),), "output": np.identity(4), "shape": (4, 4)},
        }
        return examples

    # Basic Array Operations Tests
    def test_array_operations(self):
        """Test basic array operations"""
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])

        examples = {
            "add": {"input": (a, b), "output": np.add(a, b), "result": [6, 8, 10, 12]},
            "subtract": {
                "input": (b, a),
                "output": np.subtract(b, a),
                "result": [4, 4, 4, 4],
            },
            "multiply": {
                "input": (a, b),
                "output": np.multiply(a, b),
                "result": [5, 12, 21, 32],
            },
            "divide": {
                "input": (b, a),
                "output": np.divide(b, a),
                "result": [5.0, 3.0, 2.3333333333333335, 2.0],
            },
            "power": {
                "input": (a, 2),
                "output": np.power(a, 2),
                "result": [1, 4, 9, 16],
            },
            "sqrt": {
                "input": (np.array([1, 4, 9, 16]),),
                "output": np.sqrt(np.array([1, 4, 9, 16])),
                "result": [1.0, 2.0, 3.0, 4.0],
            },
            "abs": {
                "input": (np.array([-1, -2, 3, -4]),),
                "output": np.abs(np.array([-1, -2, 3, -4])),
                "result": [1, 2, 3, 4],
            },
            "sign": {
                "input": (np.array([-2, 0, 2]),),
                "output": np.sign(np.array([-2, 0, 2])),
                "result": [-1, 0, 1],
            },
        }
        return examples

    # Statistical Functions Tests
    def test_statistical_functions(self):
        """Test statistical functions"""
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        examples = {
            "sum": {"input": (arr,), "output": np.sum(arr), "result": 55},
            "mean": {"input": (arr,), "output": np.mean(arr), "result": 5.5},
            "median": {"input": (arr,), "output": np.median(arr), "result": 5.5},
            "std": {
                "input": (arr,),
                "output": np.std(arr),
                "result": 3.0276503540974917,
            },
            "var": {
                "input": (arr,),
                "output": np.var(arr),
                "result": 9.166666666666666,
            },
            "min": {"input": (arr,), "output": np.min(arr), "result": 1},
            "max": {"input": (arr,), "output": np.max(arr), "result": 10},
            "argmin": {"input": (arr,), "output": np.argmin(arr), "result": 0},
            "argmax": {"input": (arr,), "output": np.argmax(arr), "result": 9},
        }
        return examples

    # Array Manipulation Tests
    def test_array_manipulation(self):
        """Test array manipulation functions"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])

        examples = {
            "reshape": {
                "input": (arr, (3, 2)),
                "output": np.reshape(arr, (3, 2)),
                "shape": (3, 2),
            },
            "transpose": {
                "input": (arr,),
                "output": np.transpose(arr),
                "shape": (3, 2),
                "result": [[1, 4], [2, 5], [3, 6]],
            },
            "flatten": {
                "input": (arr,),
                "output": arr.flatten(),
                "result": [1, 2, 3, 4, 5, 6],
            },
            "ravel": {
                "input": (arr,),
                "output": np.ravel(arr),
                "result": [1, 2, 3, 4, 5, 6],
            },
            "squeeze": {
                "input": (np.array([[[1], [2], [3]]]),),
                "output": np.squeeze(np.array([[[1], [2], [3]]])),
                "result": [1, 2, 3],
            },
            "concatenate": {
                "input": ([np.array([1, 2]), np.array([3, 4])],),
                "output": np.concatenate([np.array([1, 2]), np.array([3, 4])]),
                "result": [1, 2, 3, 4],
            },
            "stack": {
                "input": ([np.array([1, 2]), np.array([3, 4])],),
                "output": np.stack([np.array([1, 2]), np.array([3, 4])]),
                "shape": (2, 2),
            },
            "split": {
                "input": (np.array([1, 2, 3, 4]), 2),
                "output": np.split(np.array([1, 2, 3, 4]), 2),
                "shapes": [(2,), (2,)],
            },
        }
        return examples

    # Indexing and Slicing Tests
    def test_indexing_slicing(self):
        """Test indexing and slicing operations"""
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        examples = {
            "basic_indexing": {"input": (arr, 1, 2), "output": arr[1, 2], "result": 7},
            "slice_2d": {
                "input": (arr, slice(0, 2), slice(1, 3)),
                "output": arr[0:2, 1:3],
                "result": [[2, 3], [6, 7]],
            },
            "boolean_indexing": {
                "input": (arr, arr > 5),
                "output": arr[arr > 5],
                "result": [6, 7, 8, 9, 10, 11, 12],
            },
            "fancy_indexing": {
                "input": (arr, [0, 2], [1, 3]),
                "output": arr[[0, 2], [1, 3]],
                "result": [2, 12],
            },
            "where": {
                "input": (arr > 5, arr, -arr),
                "output": np.where(arr > 5, arr, -arr),
                "description": "Replace values >5 with original, others with negative",
            },
        }
        return examples

    # Linear Algebra Tests
    def test_linear_algebra(self):
        """Test linear algebra functions"""
        matrix = np.array([[1, 2], [3, 4]])
        vector = np.array([1, 2])

        examples = {
            "dot": {
                "input": (vector, vector),
                "output": np.dot(vector, vector),
                "result": 5,
            },
            "matmul": {
                "input": (matrix, vector),
                "output": np.matmul(matrix, vector),
                "result": [5, 11],
            },
            "det": {
                "input": (matrix,),
                "output": np.linalg.det(matrix),
                "result": -2.0,
            },
            "inv": {
                "input": (matrix,),
                "output": np.linalg.inv(matrix),
                "description": "Matrix inverse",
            },
            "eig": {
                "input": (matrix,),
                "output": np.linalg.eig(matrix),
                "description": "Eigenvalues and eigenvectors",
            },
            "solve": {
                "input": (matrix, vector),
                "output": np.linalg.solve(matrix, vector),
                "description": "Solve linear system",
            },
        }
        return examples

    # Mathematical Functions Tests
    def test_mathematical_functions(self):
        """Test mathematical functions"""
        arr = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])

        examples = {
            "sin": {
                "input": (arr,),
                "output": np.sin(arr),
                "result": [0.0, 1.0, 0.0, -1.0, 0.0],
            },
            "cos": {
                "input": (arr,),
                "output": np.cos(arr),
                "result": [1.0, 0.0, -1.0, 0.0, 1.0],
            },
            "tan": {
                "input": (np.array([0, np.pi / 4]),),
                "output": np.tan(np.array([0, np.pi / 4])),
                "result": [0.0, 1.0],
            },
            "exp": {
                "input": (np.array([0, 1, 2]),),
                "output": np.exp(np.array([0, 1, 2])),
                "result": [1.0, 2.718281828459045, 7.38905609893065],
            },
            "log": {
                "input": (np.array([1, np.e, np.e**2]),),
                "output": np.log(np.array([1, np.e, np.e**2])),
                "result": [0.0, 1.0, 2.0],
            },
            "log10": {
                "input": (np.array([1, 10, 100]),),
                "output": np.log10(np.array([1, 10, 100])),
                "result": [0.0, 1.0, 2.0],
            },
            "floor": {
                "input": (np.array([1.2, 2.7, -0.5]),),
                "output": np.floor(np.array([1.2, 2.7, -0.5])),
                "result": [1.0, 2.0, -1.0],
            },
            "ceil": {
                "input": (np.array([1.2, 2.7, -0.5]),),
                "output": np.ceil(np.array([1.2, 2.7, -0.5])),
                "result": [2.0, 3.0, 0.0],
            },
        }
        return examples

    # Comparison and Logical Tests
    def test_comparison_logical(self):
        """Test comparison and logical operations"""
        a = np.array([1, 2, 3, 4])
        b = np.array([2, 2, 2, 2])

        examples = {
            "equal": {
                "input": (a, b),
                "output": np.equal(a, b),
                "result": [False, True, False, False],
            },
            "not_equal": {
                "input": (a, b),
                "output": np.not_equal(a, b),
                "result": [True, False, True, True],
            },
            "greater": {
                "input": (a, b),
                "output": np.greater(a, b),
                "result": [False, False, True, True],
            },
            "less": {
                "input": (a, b),
                "output": np.less(a, b),
                "result": [True, False, False, False],
            },
            "logical_and": {
                "input": ([True, False, True], [True, True, False]),
                "output": np.logical_and([True, False, True], [True, True, False]),
                "result": [True, False, False],
            },
            "logical_or": {
                "input": ([True, False, True], [True, True, False]),
                "output": np.logical_or([True, False, True], [True, True, False]),
                "result": [True, True, True],
            },
            "logical_not": {
                "input": ([True, False],),
                "output": np.logical_not([True, False]),
                "result": [False, True],
            },
        }
        return examples

    # Shape and Dimension Operations Tests
    def test_shape_operations(self):
        """Test shape and dimension operations"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])

        examples = {
            "shape": {"input": (arr,), "output": arr.shape, "result": (2, 3)},
            "ndim": {"input": (arr,), "output": arr.ndim, "result": 2},
            "size": {"input": (arr,), "output": arr.size, "result": 6},
            "atleast_1d": {"input": (1,), "output": np.atleast_1d(1), "result": [1]},
            "atleast_2d": {
                "input": ([1, 2, 3],),
                "output": np.atleast_2d([1, 2, 3]),
                "shape": (1, 3),
            },
            "atleast_3d": {
                "input": ([[1, 2]],),
                "output": np.atleast_3d([[1, 2]]),
                "shape": (1, 1, 2),
            },
            "expand_dims": {
                "input": (arr, 0),
                "output": np.expand_dims(arr, 0),
                "shape": (1, 2, 3),
            },
            "swapaxes": {
                "input": (arr, 0, 1),
                "output": np.swapaxes(arr, 0, 1),
                "shape": (3, 2),
            },
        }
        return examples

    # Set Operations Tests
    def test_set_operations(self):
        """Test set operations"""
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([3, 4, 5, 6, 7])

        examples = {
            "unique": {
                "input": (np.array([1, 2, 2, 3, 3, 3]),),
                "output": np.unique(np.array([1, 2, 2, 3, 3, 3])),
                "result": [1, 2, 3],
            },
            "intersect1d": {
                "input": (arr1, arr2),
                "output": np.intersect1d(arr1, arr2),
                "result": [3, 4, 5],
            },
            "union1d": {
                "input": (arr1, arr2),
                "output": np.union1d(arr1, arr2),
                "result": [1, 2, 3, 4, 5, 6, 7],
            },
            "setdiff1d": {
                "input": (arr1, arr2),
                "output": np.setdiff1d(arr1, arr2),
                "result": [1, 2],
            },
            "setxor1d": {
                "input": (arr1, arr2),
                "output": np.setxor1d(arr1, arr2),
                "result": [1, 2, 6, 7],
            },
            "in1d": {
                "input": (arr1, arr2),
                "output": np.in1d(arr1, arr2),
                "result": [False, False, True, True, True],
            },
            "isin": {
                "input": (arr1, arr2),
                "output": np.isin(arr1, arr2),
                "result": [False, False, True, True, True],
            },
        }
        return examples

    # Sorting Tests
    def test_sorting(self):
        """Test sorting operations"""
        arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

        examples = {
            "sort": {
                "input": (arr,),
                "output": np.sort(arr),
                "result": [1, 1, 2, 3, 4, 5, 6, 9],
            },
            "argsort": {
                "input": (arr,),
                "output": np.argsort(arr),
                "result": [1, 3, 6, 0, 2, 4, 7, 5],
            },
            "lexsort": {
                "input": ((np.array([1, 2, 3]), np.array([3, 2, 1])),),
                "output": np.lexsort((np.array([1, 2, 3]), np.array([3, 2, 1]))),
                "result": [2, 1, 0],
            },
            "partition": {
                "input": (arr, 3),
                "output": np.partition(arr, 3),
                "description": "Partition array around kth element",
            },
            "argpartition": {
                "input": (arr, 3),
                "output": np.argpartition(arr, 3),
                "description": "Indices that would partition array",
            },
        }
        return examples

    # Random Module Tests
    def test_random_functions(self):
        """Test random module functions"""
        rng = np.random.default_rng(42)

        examples = {
            "random": {
                "input": ((5,),),
                "output": rng.random(5),
                "description": "Random floats in [0, 1)",
            },
            "randint": {
                "input": ((0, 10, 5),),
                "output": rng.integers(0, 10, 5),
                "description": "Random integers",
            },
            "normal": {
                "input": ((0, 1, 5),),
                "output": rng.normal(0, 1, 5),
                "description": "Normal distribution",
            },
            "uniform": {
                "input": ((0, 1, 5),),
                "output": rng.uniform(0, 1, 5),
                "description": "Uniform distribution",
            },
            "choice": {
                "input": ((np.array([1, 2, 3, 4, 5]), 3),),
                "output": rng.choice(np.array([1, 2, 3, 4, 5]), 3),
                "description": "Random choice from array",
            },
            "permutation": {
                "input": ((5,),),
                "output": rng.permutation(5),
                "description": "Random permutation",
            },
            "shuffle": {
                "input": (np.array([1, 2, 3, 4, 5]),),
                "output": None,  # In-place operation
                "description": "Shuffle array in-place",
            },
        }
        return examples

    # FFT Tests
    def test_fft_functions(self):
        """Test FFT functions"""
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        examples = {
            "fft": {
                "input": (arr,),
                "output": np.fft.fft(arr),
                "description": "Fast Fourier Transform",
            },
            "ifft": {
                "input": (np.fft.fft(arr),),
                "output": np.fft.ifft(np.fft.fft(arr)),
                "description": "Inverse FFT",
            },
            "fft2": {
                "input": (np.array([[1, 2], [3, 4]]),),
                "output": np.fft.fft2(np.array([[1, 2], [3, 4]])),
                "description": "2D FFT",
            },
            "fftfreq": {
                "input": ((8,),),
                "output": np.fft.fftfreq(8),
                "description": "FFT frequency bins",
            },
            "fftshift": {
                "input": (np.fft.fft(arr),),
                "output": np.fft.fftshift(np.fft.fft(arr)),
                "description": "Shift FFT output",
            },
        }
        return examples

    # Edge Cases and Special Values Tests
    def test_edge_cases(self):
        """Test edge cases and special values"""
        examples = {
            "nan_handling": {
                "input": (np.array([1, 2, np.nan, 4]),),
                "output": np.isnan(np.array([1, 2, np.nan, 4])),
                "result": [False, False, True, False],
            },
            "inf_handling": {
                "input": (np.array([1, np.inf, -np.inf, 4]),),
                "output": np.isinf(np.array([1, np.inf, -np.inf, 4])),
                "result": [False, True, True, False],
            },
            "empty_array": {"input": ((),), "output": np.array([]), "shape": (0,)},
            "zero_division": {
                "input": (np.array([1, 2, 3]), np.array([0, 1, 0])),
                "output": np.divide(np.array([1, 2, 3]), np.array([0, 1, 0])),
                "description": "Handles division by zero",
            },
            "complex_numbers": {
                "input": (np.array([1 + 2j, 3 - 4j]),),
                "output": np.abs(np.array([1 + 2j, 3 - 4j])),
                "result": [2.23606797749979, 5.0],
            },
        }
        return examples

    def run_all_tests(self):
        """Run all test examples and return results"""
        all_tests = {}

        test_methods = [
            self.test_array_creation,
            self.test_array_operations,
            self.test_statistical_functions,
            self.test_array_manipulation,
            self.test_indexing_slicing,
            self.test_linear_algebra,
            self.test_mathematical_functions,
            self.test_comparison_logical,
            self.test_shape_operations,
            self.test_set_operations,
            self.test_sorting,
            self.test_random_functions,
            self.test_fft_functions,
            self.test_edge_cases,
        ]

        for method in test_methods:
            test_name = method.__name__.replace("test_", "")
            try:
                all_tests[test_name] = method()
            except Exception as e:
                all_tests[test_name] = {"error": str(e)}

        return all_tests

    def generate_test_data_files(self):
        """Generate test data files for cross-language testing"""
        test_data = {}

        # Generate arrays of different dtypes and shapes
        shapes = [(10,), (5, 4), (2, 3, 4)]
        dtypes = [
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        ]

        for shape in shapes:
            for dtype in dtypes:
                key = f"array_{shape}_{dtype}"
                try:
                    arr = np.random.randn(*shape).astype(dtype)
                    if "complex" in dtype:
                        arr = arr + 1j * np.random.randn(*shape).astype(
                            dtype.replace("complex", "float")
                        )
                    test_data[key] = arr
                except Exception as e:
                    print(f"Could not generate {key}: {e}")

        return test_data


def main():
    """Main function to run tests and generate examples"""
    print("Generating comprehensive NumPy test examples...")

    test_examples = NumPyTestExamples()

    # Run all tests
    all_tests = test_examples.run_all_tests()

    # Generate test data
    test_data = test_examples.generate_test_data_files()

    print(f"Generated {len(all_tests)} test categories")
    print(f"Generated {len(test_data)} test data arrays")

    # Save test data files
    import os

    test_dir = "/home/grant/github/pocket-tts/test_data/generated"
    os.makedirs(test_dir, exist_ok=True)

    for name, arr in test_data.items():
        np.save(os.path.join(test_dir, f"{name}.npy"), arr)
        # Also save as Parquet for language interoperability
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table({"data": arr.flatten()})
            pq.write_table(table, os.path.join(test_dir, f"{name}.parquet"))
        except ImportError:
            # PyArrow is optional - Parquet export is skipped if not installed
            # NumPy format is already saved above, so this is a nice-to-have feature
            pass

    print(f"Test data saved to {test_dir}")

    return all_tests, test_data


if __name__ == "__main__":
    main()
