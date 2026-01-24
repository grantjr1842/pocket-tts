# Comprehensive Usage Guide: NumPy NumType

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Understanding NumType](#understanding-numtype)
3. [Installation and Setup](#installation-and-setup)
4. [Usage Examples](#usage-examples)
5. [API Documentation](#api-documentation)
6. [Advanced Features](#advanced-features)
7. [Integration with Type Checkers](#integration-with-type-checkers)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting](#troubleshooting)
10. [Development and Contributing](#development-and-contributing)

## Repository Overview

### Purpose and Functionality

NumType is an experimental typing stubs package for NumPy that provides enhanced type annotations and improved static type checking capabilities. It serves as a replacement for the default typing stubs bundled with NumPy, offering more accurate and comprehensive type information.

### Key Features and Capabilities

- **Enhanced Type Annotations**: More precise and comprehensive type information
- **NEP 50 Compliance**: Full support for NumPy Enhancement Proposal 50 promotion rules
- **Improved Ufunc Annotations**: Better typing for universal functions
- **Experimental Features**: Access to cutting-edge typing improvements
- **Type Checker Integration**: Works with mypy, pyright, and other type checkers

### Relationship to NumPy's Type System

NumType extends and improves upon NumPy's built-in type system by:

- Providing more accurate scalar type hierarchies
- Implementing proper dtype promotion rules
- Offering better array and scalar interaction typing
- Supporting experimental typing features not yet in main NumPy

### Current Status

**⚠️ Important**: NumType is experimental and under active development. The API and behavior may change as the project evolves.

## Understanding NumType

### What Are Typing Stubs?

Typing stubs (`.pyi` files) provide type information for Python libraries that may not have complete type annotations in their source code. They enable static type checkers to analyze code usage and catch type-related errors before runtime.

### Why NumType?

NumPy's built-in typing stubs have limitations:

1. **Incomplete Coverage**: Some functions lack proper type annotations
2. **Inaccurate Promotion**: Type promotion rules not fully implemented
3. **Legacy Constraints**: Backward compatibility limits improvements
4. **Development Lag**: Typing improvements lag behind runtime features

NumType addresses these issues by:

- **Complete Coverage**: Comprehensive type annotations for all public APIs
- **Accurate Promotion**: Full NEP 50 promotion rule implementation
- **Experimental Freedom**: Can implement cutting-edge typing features
- **Active Development**: Rapid iteration and improvement

### Key Concepts

#### 1. Scalar Type Hierarchy

NumType provides a more accurate scalar type hierarchy:

```python
# In NumType, these are proper subclasses
np.int64 <: np.integer <: np.signedinteger <: np.number
np.float64 <: np.floating <: np.number
```

#### 2. Array-Scalar Interactions

Better typing for operations between arrays and scalars:

```python
def add_arrays(a: np.ndarray[np.float64], b: float) -> np.ndarray[np.float64]:
    return a + b  # Properly typed in NumType
```

#### 3. DType Promotion

Accurate dtype promotion following NEP 50:

```python
# Result type properly determined
result = np.int32(5) + np.float64(3.14)  # -> np.float64
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- NumPy 1.20+
- Static type checker (mypy, pyright, etc.)

### Installation Methods

#### Standard Installation

```bash
# Install from PyPI (when available)
pip install numtype

# Or install directly from GitHub
pip install git+https://github.com/numpy/numtype.git
```

#### Development Installation

```bash
# Clone the repository
git clone https://github.com/numpy/numtype.git
cd numtype

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

#### Using Modern Package Managers

```bash
# With uv
uv add numtype

# With pixi
pixi add numtype
```

### Type Checker Setup

#### MyPy Configuration

```ini
# mypy.ini
[mypy]
plugins = numpy.typing.mypy_plugin

# Or with NumType
[mypy]
plugins = numtype.mypy_plugin

# Enable strict mode for best results
strict = True
```

#### Pyright Configuration

```json
// pyrightconfig.json
{
    "include": ["src"],
    "pythonVersion": "3.8",
    "typeCheckingMode": "strict",
    "reportMissingImports": "error"
}
```

### Verification

```python
# test_numtype.py
import numpy as np
from typing import Any

# Test basic typing
def test_basic_typing() -> np.ndarray[np.float64]:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return arr

# Test scalar operations
def test_scalar_operations() -> np.ndarray[np.float64]:
    arr = np.array([1, 2, 3], dtype=np.int32)
    result = arr + 1.5  # Should promote to float64
    return result

if __name__ == "__main__":
    print("NumType installation test passed!")
```

Run type checking:

```bash
# With mypy
mypy test_numtype.py

# With pyright
pyright test_numtype.py
```

## Usage Examples

### Basic Array Typing

```python
import numpy as np
from typing import TypeVar

# Generic array typing
DType = TypeVar('DType', bound=np.generic)

def process_array(arr: np.ndarray[DType]) -> np.ndarray[DType]:
    """Process array while preserving dtype"""
    return arr * 2

# Specific dtype typing
def process_floats(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """Process float64 array"""
    return np.sqrt(arr)

# Usage
float_arr = np.array([1.0, 4.0, 9.0], dtype=np.float64)
result = process_floats(float_arr)  # Properly typed
```

### Scalar Operations

```python
import numpy as np

def scalar_operations() -> None:
    """Demonstrate scalar typing"""

    # Integer operations
    i: np.int64 = np.int64(42)
    j: np.int32 = np.int32(10)

    # Type promotion - NumType handles this correctly
    result1: np.float64 = i + j  # Promotes to float64

    # Float operations
    f: np.float64 = np.float64(3.14)
    d: np.float32 = np.float32(2.71)

    result2: np.float64 = f + d  # Promotes to float64

    # Array-scalar operations
    arr = np.array([1, 2, 3], dtype=np.int32)
    scalar_result: np.ndarray[np.float64] = arr + 1.5  # Proper promotion
```

### Universal Functions

```python
import numpy as np

def ufunc_examples() -> None:
    """Universal function typing examples"""

    # Basic ufuncs
    arr1 = np.array([1, 2, 3], dtype=np.float64)
    arr2 = np.array([4, 5, 6], dtype=np.float32)

    # NumType provides accurate return typing
    result1: np.ndarray[np.float64] = np.add(arr1, arr2)
    result2: np.ndarray[np.float64] = np.multiply(arr1, 1.5)

    # Reduction operations
    sum_result: np.float64 = np.sum(arr1)  # Correct scalar type
    mean_result: np.float64 = np.mean(arr1)

    # Comparison operations
    comparison: np.ndarray[np.bool_] = arr1 > 2.0
```

### Custom DTypes

```python
import numpy as np
from typing import NewType

# Define custom dtype types
CustomFloat = NewType('CustomFloat', np.float64)
CustomInt = NewType('CustomInt', np.int64)

def custom_dtype_operations() -> None:
    """Operations with custom dtype annotations"""

    # Create arrays with custom dtype annotations
    custom_float_arr: np.ndarray[CustomFloat] = np.array(
        [1.0, 2.0, 3.0], dtype=np.float64
    )

    custom_int_arr: np.ndarray[CustomInt] = np.array(
        [1, 2, 3], dtype=np.int64
    )

    # Operations preserve type information
    result: np.ndarray[CustomFloat] = custom_float_arr + 1.0
```

### Function Signatures

```python
import numpy as np
from typing import Union, Literal

def complex_function(
    data: Union[np.ndarray[np.float64], np.ndarray[np.int32]],
    axis: Union[int, None] = None,
    keepdims: bool = False,
    dtype: Union[np.dtype[np.float64], np.dtype[np.int32], None] = None
) -> np.ndarray[np.float64]:
    """
    Complex function with comprehensive typing.

    Args:
        data: Input array of float64 or int32
        axis: Axis along which to operate
        keepdims: Whether to keep dimensions
        dtype: Output dtype specification

    Returns:
        Array with float64 dtype
    """
    return np.mean(data, axis=axis, keepdims=keepdims, dtype=dtype)

# Usage with proper typing
arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
result = complex_function(arr, axis=0, keepdims=True)
```

### Generic Functions

```python
import numpy as np
from typing import TypeVar, Generic

# Define generic type variables
T = TypeVar('T', bound=np.generic)
U = TypeVar('U', bound=np.generic)

class ArrayProcessor(Generic[T, U]):
    """Generic array processor class"""

    def __init__(self, input_dtype: type[np.dtype[T]], output_dtype: type[np.dtype[U]]):
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def process(self, arr: np.ndarray[T]) -> np.ndarray[U]:
        """Process array with type conversion"""
        return arr.astype(self.output_dtype)

# Usage
processor = ArrayProcessor[np.int32, np.float64](np.int32, np.float64)
input_arr = np.array([1, 2, 3], dtype=np.int32)
output_arr = processor.process(input_arr)  # Properly typed as float64
```

## API Documentation

### Core Type Classes

#### Scalar Types

NumType provides proper scalar type hierarchies:

```python
# Number hierarchy
np.number
├── np.integer
│   ├── np.signedinteger
│   │   ├── np.int8, np.int16, np.int32, np.int64
│   │   └── np.longlong
│   └── np.unsignedinteger
│       ├── np.uint8, np.uint16, np.uint32, np.uint64
│       └── np.ulonglong
├── np.floating
│   ├── np.float16, np.float32, np.float64
│   └── np.longdouble
├── np.complexfloating
│   ├── np.complex64, np.complex128
│   └── np.clongdouble
├── np.bool_
├── np.object_
└── np.str_
```

#### Array Types

```python
# Generic array type
np.ndarray[ScalarType]

# Specific array types
np.ndarray[np.float64]
np.ndarray[np.int32]
np.ndarray[np.bool_]

# Multi-dimensional arrays
np.ndarray[np.float64, Any]  # Using Any for shape
```

### Function Types

#### Universal Functions

```python
# Ufunc signatures
def ufunc_func(
    *inputs: Union[np.ndarray[T], T],
    **kwargs: Any
) -> Union[np.ndarray[U], U]:
    """Generic ufunc signature"""
    pass

# Specific ufuncs
np.add: Callable[[np.ndarray[T], np.ndarray[T]], np.ndarray[T]]
np.multiply: Callable[[np.ndarray[T], T], np.ndarray[T]]
np.sum: Callable[[np.ndarray[T]], T]
```

#### Creation Functions

```python
# Array creation functions
np.array: Callable[[Any], np.ndarray[Any]]
np.zeros: Callable[[ShapeLike, DTypeLike], np.ndarray[DType]]
np.ones: Callable[[ShapeLike, DTypeLike], np.ndarray[DType]]
np.empty: Callable[[ShapeLike, DTypeLike], np.ndarray[DType]]
```

### Type Aliases

#### Common Aliases

```python
# Shape types
Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]

# Dtype types
DType = Type[np.dtype[Any]]
DTypeLike = Union[type, str, np.dtype[Any]]

# Array types
Array1D = np.ndarray[T, Tuple[int]]
Array2D = np.ndarray[T, Tuple[int, int]]
ArrayND = np.ndarray[T, Shape]
```

## Advanced Features

### NEP 50 Promotion Rules

NumType fully implements NEP 50 promotion rules:

```python
def nep50_examples() -> None:
    """Examples of NEP 50 promotion behavior"""

    # Weak scalar promotion
    weak_int: np.int64 = np.int64(42)
    weak_float: np.float64 = np.float64(3.14)

    # Array operations
    arr_int = np.array([1, 2, 3], dtype=np.int32)
    arr_float = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    # Promotion follows NEP 50 rules
    result1: np.ndarray[np.float64] = arr_int + weak_float
    result2: np.ndarray[np.float64] = weak_int + arr_float
    result3: np.ndarray[np.float64] = arr_int + arr_float
```

### Shape Typing

```python
from typing import Tuple, Literal

def shape_typing_examples() -> None:
    """Examples of shape typing"""

    # Fixed shape arrays
    vector: np.ndarray[np.float64, Tuple[int]] = np.array([1.0, 2.0, 3.0])
    matrix: np.ndarray[np.float64, Tuple[int, int]] = np.array([[1.0, 2.0], [3.0, 4.0]])

    # Literal shapes
    vec3: np.ndarray[np.float64, Tuple[Literal[3]]] = np.array([1.0, 2.0, 3.0])
    mat2x2: np.ndarray[np.float64, Tuple[Literal[2], Literal[2]]] = np.eye(2)
```

### Protocol Support

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class ArrayLike(Protocol[np.generic]):
    """Protocol for array-like objects"""

    def __array__(self, dtype: Any = None) -> np.ndarray[Any]: ...

def process_array_like(obj: ArrayLike[np.float64]) -> np.ndarray[np.float64]:
    """Process any array-like object"""
    return np.asarray(obj, dtype=np.float64)
```

### Generic Constraints

```python
from typing import TypeVar, Union

# Constrained type variables
IntType = TypeVar('IntType', np.int8, np.int16, np.int32, np.int64)
FloatType = TypeVar('FloatType', np.float16, np.float32, np.float64)

def process_integers(arr: np.ndarray[IntType]) -> np.ndarray[IntType]:
    """Process any integer array"""
    return arr * 2

def process_floats(arr: np.ndarray[FloatType]) -> np.ndarray[FloatType]:
    """Process any float array"""
    return arr / 2.0
```

## Integration with Type Checkers

### MyPy Integration

#### Configuration

```ini
# mypy.ini
[mypy]
plugins = numtype.mypy_plugin
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

[mypy-numpy.*]
ignore_errors = False

[mypy-numtype.*]
ignore_errors = False
```

#### Plugin Features

NumType's MyPy plugin provides:

- **Enhanced Type Inference**: Better type inference for NumPy operations
- **DType Promotion**: Accurate promotion rule checking
- **Shape Validation**: Optional shape checking
- **Error Messages**: Improved error messages for NumPy-specific issues

### Pyright Integration

#### Configuration

```json
// pyrightconfig.json
{
    "include": ["src"],
    "exclude": ["**/node_modules", "**/__pycache__"],
    "pythonVersion": "3.8",
    "typeCheckingMode": "strict",
    "reportMissingImports": "error",
    "reportMissingTypeStubs": "error",
    "reportUnknownParameterType": "error",
    "reportUnknownVariableType": "error",
    "reportUnknownLambdaType": "error",
    "reportUnknownMemberType": "error"
}
```

#### VS Code Integration

```json
// .vscode/settings.json
{
    "python.analysis.typeCheckingMode": "strict",
    "python.analysis.autoImportCompletions": true,
    "python.analysis.diagnosticSeverityOverrides": {
        "reportUnknownMemberType": "information"
    }
}
```

### Other Type Checkers

#### PyCharm

```python
# PyCharm typically works out of the box with NumType
# Ensure "Python type checking" is enabled in settings
```

#### Pyre

```bash
# Pyre configuration
# .pyre_configuration
{
    "source_directories": ["src"],
    "strict": true
}
```

## Performance Considerations

### Type Checking Performance

#### Optimization Strategies

```python
# Use specific types when possible
def good_typing(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    return arr * 2

# Avoid overly generic types
def avoid_generic(arr: np.ndarray[Any]) -> np.ndarray[Any]:
    return arr * 2
```

#### Caching Type Information

```python
from functools import lru_cache
from typing import Type

@lru_cache(maxsize=128)
def get_dtype_info(dtype: Type[np.dtype[Any]]) -> dict:
    """Cache dtype information for better performance"""
    return {
        'itemsize': dtype().itemsize,
        'kind': dtype().kind,
        'name': dtype().name
    }
```

### Runtime Performance

#### Type Checking Overhead

NumType is a typing-only package and has **zero runtime overhead**:

- **No Runtime Impact**: Type information is only used during static analysis
- **No Import Cost**: Stub files don't affect runtime performance
- **No Memory Overhead**: No additional memory usage at runtime

#### Compilation Considerations

```python
# Type annotations don't affect runtime performance
def optimized_function(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    # This runs just as fast as untyped version
    return np.sum(arr, axis=0)  # Optimized NumPy operation
```

### Best Practices

#### Type Annotation Guidelines

1. **Be Specific**: Use precise types when possible
2. **Avoid Over-Generics**: Don't use `Any` unless necessary
3. **Document Complex Types**: Use type aliases for complex signatures
4. **Consistent Naming**: Use consistent naming for type variables

#### Performance Tips

```python
# Good: Specific types
def process_specific(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    return arr * 2

# Avoid: Overly generic
def process_generic(arr: np.ndarray[Any]) -> np.ndarray[Any]:
    return arr * 2

# Good: Type aliases for complex types
FloatArray = np.ndarray[np.float64]
IntArray = np.ndarray[np.int32]

def process_with_aliases(arr: FloatArray) -> FloatArray:
    return arr * 2
```

## Troubleshooting

### Common Issues

#### 1. Installation Problems

**Problem**: `ImportError: cannot import name 'numtype'`

**Solution**: Verify installation and Python path:

```bash
# Check installation
pip list | grep numtype

# Reinstall if necessary
pip uninstall numtype
pip install git+https://github.com/numpy/numtype.git

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 2. Type Checker Conflicts

**Problem**: Type checker doesn't use NumType stubs

**Solution**: Configure type checker properly:

```bash
# For mypy, check plugin loading
mypy --show-traceback your_file.py

# Verify stub files are found
mypy --no-site-packages your_file.py

# Check stub file locations
python -c "import numtype; print(numtype.__file__)"
```

#### 3. Type Promotion Issues

**Problem**: Unexpected type promotion behavior

**Solution**: Understand NEP 50 rules:

```python
# Check promotion behavior
def check_promotion():
    i: np.int32 = np.int32(5)
    f: np.float64 = np.float64(3.14)

    # This should promote to float64
    result = i + f
    print(f"Result type: {type(result)}")
    print(f"Result dtype: {result.dtype if hasattr(result, 'dtype') else 'scalar'}")
```

#### 4. Array Shape Issues

**Problem**: Shape typing not working as expected

**Solution**: NumType focuses on dtype, not shape typing:

```python
# Shape typing is limited in NumType
# Use runtime checks for shape validation

def validate_shape(arr: np.ndarray[np.float64], expected_shape: tuple) -> np.ndarray[np.float64]:
    if arr.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {arr.shape}")
    return arr
```

#### 5. Compatibility Issues

**Problem**: NumPy version incompatibility

**Solution**: Check NumPy version compatibility:

```python
import numpy as np
import sys

def check_compatibility():
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")

    # Check for minimum required version
    if tuple(map(int, np.__version__.split('.')[:2])) < (1, 20):
        raise ValueError("NumPy 1.20+ required for NumType")
```

### Debugging Tips

#### Enable Verbose Type Checking

```bash
# MyPy verbose mode
mypy --verbose your_file.py

# Show type inference
mypy --show-traceback --show-error-codes your_file.py
```

#### Type Checking Debug

```python
# Add explicit type annotations for debugging
def debug_function(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    # Add type comments for clarity
    result: np.ndarray[np.float64] = arr * 2  # type: ignore[assignment]
    return result
```

#### Stub File Inspection

```python
# Check which stub files are being used
import mypy.api
result = mypy.api.run(["--show-traceback", "your_file.py"])
print(result[0])  # stdout
print(result[1])  # stderr
```

### Getting Help

#### Community Resources

1. **GitHub Issues**: https://github.com/numpy/numtype/issues
2. **NumPy Discourse**: https://numpy.discourse.group/
3. **Stack Overflow**: Tag with `numpy` and `numtype`

#### Reporting Issues

When reporting issues, include:

1. **Environment Information**:
   ```python
   import sys
   import numpy as np
   import mypy

   print(f"Python: {sys.version}")
   print(f"NumPy: {np.__version__}")
   print(f"MyPy: {mypy.__version__}")
   ```

2. **Minimal Reproducible Example**
3. **Expected vs Actual Behavior**
4. **Error Messages and Tracebacks**

#### Workarounds

```python
# If NumType causes issues, temporarily disable
# type: ignore[import]
import numpy as np

# Or use specific type ignores
result = some_operation()  # type: ignore[assignment]
```

## Development and Contributing

### Development Setup

#### Clone and Install

```bash
# Clone repository
git clone https://github.com/numpy/numtype.git
cd numtype

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Development Dependencies

```bash
# Install additional tools
pip install mypy pyright pytest
pip install numpy scipy  # For testing
```

### Testing

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_basic.py

# Run with coverage
pytest --cov=numtype

# Run type checking tests
pytest tests/test_typing.py
```

#### Type Checking Tests

```bash
# Test with mypy
mypy tests/

# Test with pyright
pyright tests/

# Test multiple configurations
mypy --strict tests/
mypy --no-strict tests/
```

### Contributing Guidelines

#### Code Style

```python
# Follow PEP 8 and NumPy style guidelines
# Use type annotations consistently
def example_function(arr: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """Example function with proper typing."""
    return arr * 2
```

#### Documentation

```python
# Document complex types with examples
def complex_function(
    data: np.ndarray[np.float64],
    weights: Optional[np.ndarray[np.float64]] = None
) -> np.ndarray[np.float64]:
    """
    Complex function with comprehensive documentation.

    Args:
        data: Input array of float64 values
        weights: Optional weights array, same shape as data

    Returns:
        Weighted sum of input data

    Example:
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> result = complex_function(arr)
        >>> print(result)
        [1.0 2.0 3.0]
    """
    if weights is not None:
        return data * weights
    return data
```

#### Pull Request Process

1. **Fork Repository**: Create fork on GitHub
2. **Create Branch**: `git checkout -b feature/your-feature`
3. **Make Changes**: Implement your changes with tests
4. **Run Tests**: Ensure all tests pass
5. **Type Check**: Verify type checking works
6. **Submit PR**: Create pull request with description

### Release Process

#### Version Management

```bash
# Update version in setup.py or pyproject.toml
# Update changelog
# Tag release
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# Build and upload
python -m build
twine upload dist/*
```

#### Documentation Updates

```bash
# Build documentation
cd docs/
make html

# Check documentation
open _build/html/index.html
```

## Conclusion

NumType represents a significant advancement in NumPy typing capabilities, providing:

- **Enhanced Type Safety**: Better static type checking for NumPy code
- **Accurate Promotions**: Full NEP 50 compliance
- **Experimental Features**: Access to cutting-edge typing improvements
- **Zero Runtime Overhead**: Pure typing improvements without performance cost

### Key Benefits

1. **Development Experience**: Better IDE support and error detection
2. **Code Quality**: Catch type errors before runtime
3. **Documentation**: Types serve as inline documentation
4. **Refactoring**: Safer code refactoring with type guarantees

### Future Directions

- **Shape Typing**: Enhanced shape and dimension typing
- **Runtime Integration**: Potential runtime type checking features
- **Ecosystem Integration**: Better integration with scientific Python stack
- **Performance**: Optimized type checking for large codebases

### Getting Started

1. **Install NumType**: `pip install numtype`
2. **Configure Type Checker**: Set up mypy or pyright
3. **Update Code**: Add type annotations to NumPy code
4. **Run Checks**: Verify type checking works
5. **Contribute**: Help improve NumType for the community

NumType is paving the way for better typed scientific computing in Python, making NumPy code more robust, maintainable, and developer-friendly.
