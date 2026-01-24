# Comprehensive Usage Guide: NumPy User-Defined DTypes

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Understanding NEP 42](#understanding-nep-42)
3. [Installation and Setup](#installation-and-setup)
4. [Creating Custom DTypes](#creating-custom-dtypes)
5. [Usage Examples](#usage-examples)
6. [API Documentation](#api-documentation)
7. [Advanced Features](#advanced-features)
8. [Integration with NumPy](#integration-with-numpy)
9. [Performance Considerations](#performance-considerations)
10. [Troubleshooting](#troubleshooting)

## Repository Overview

### Purpose and Functionality

The `numpy-user-dtypes` repository serves as a collection of example implementations demonstrating the NEP 42 extensible dtype API. This repository provides practical examples of how to create custom data types that integrate seamlessly with NumPy's array ecosystem.

### Key Features and Capabilities

- **NEP 42 Compliance**: All examples use the new extensible dtype API
- **Real-world Patterns**: Demonstrates common use cases for custom dtypes
- **Educational Resource**: Learning materials for dtype developers
- **Reference Implementations**: Working examples to build upon
- **Integration Examples**: Shows how custom dtypes work with NumPy functions

### Relationship to NumPy's Dtype System

This repository is part of NumPy's modernization effort to make the dtype system:

- **Extensible**: Allow users to create custom data types
- **Consistent**: Uniform behavior across all dtypes
- **Performant**: Efficient operations with custom types
- **Maintainable**: Clean separation of concerns

### Current Status

**⚠️ Important**: The dtype API is not finalized and the examples in this repository are still active works in progress. These dtypes are not meant for real-world use yet.

## Understanding NEP 42

### What is NEP 42?

NEP 42 (NumPy Enhancement Proposal 42) proposes a new and extensible dtype system for NumPy. It addresses limitations in the current monolithic dtype architecture.

### Key Concepts

#### 1. DType Classes vs Instances

```python
# DType class (like a type)
int64_dtype_class = np.dtype[np.int64]

# DType instance (like a specific configuration)
int64_dtype_instance = np.dtype(np.int64)
```

#### 2. Common DType Resolution

When operations involve different dtypes, NumPy finds a common dtype:

```python
# Common dtype resolution
result_dtype = np.find_common_type([np.int16, np.uint16])
# Returns np.int32 (minimal type that can hold both)
```

#### 3. Extensible Architecture

The new system allows:
- Custom dtype classes
- User-defined operations
- Seamless integration with NumPy functions
- Proper type hinting support

### Benefits of NEP 42

- **Modularity**: Clean separation between dtype logic and operations
- **Extensibility**: Easy to add new data types
- **Performance**: Optimized dispatch for custom types
- **Consistency**: Uniform behavior across all dtypes

## Installation and Setup

### Prerequisites

- NumPy 1.24+ (for experimental dtype support)
- Python 3.8+
- C compiler (for building extensions)
- Development tools: `meson`, `ninja`

### Getting the Repository

```bash
# Clone the repository
git clone https://github.com/numpy/numpy-user-dtypes.git
cd numpy-user-dtypes

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy cython
```

### Building Examples

```bash
# Build all examples
python setup.py build_ext --inplace

# Or build specific examples
cd examples/unit_dtype
python setup.py build_ext --inplace
```

### Development Setup

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/

# Run examples
python examples/basic_usage.py
```

## Creating Custom DTypes

### Basic DType Structure

```python
import numpy as np
from numpy.core.dtype import dtype

class CustomDtype(dtype):
    """Base class for custom dtypes"""

    def __new__(cls, *args, **kwargs):
        # Create dtype instance
        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def from_scalar(cls, scalar):
        """Create dtype from scalar value"""
        # Implementation here
        pass

    def __common_dtype__(cls, other):
        """Find common dtype with another dtype"""
        # Implementation here
        pass
```

### Essential Methods

#### 1. Type Discovery

```python
@classmethod
def discover_type_from_scalar(cls, scalar):
    """Discover if scalar can be represented by this dtype"""
    # Return dtype instance if compatible, None otherwise
    pass
```

#### 2. Casting Support

```python
def can_cast_from(self, other):
    """Check if casting from other dtype is possible"""
    # Return True/False
    pass

def cast_from(self, other, array):
    """Cast array from other dtype to this dtype"""
    # Implementation
    pass
```

#### 3. Common DType Resolution

```python
@classmethod
def __common_dtype__(cls, other):
    """Find common dtype between this and another dtype"""
    # Return common dtype class
    pass
```

### Example: Unit DType

```python
class UnitDtype(dtype):
    """Dtype for physical units"""

    def __init__(self, unit):
        self.unit = unit
        super().__init__()

    @classmethod
    def from_string(cls, unit_str):
        """Create from unit string like 'm/s'"""
        # Parse unit string
        return cls(unit_str)

    def __common_dtype__(cls, other):
        """Find common unit dtype"""
        if isinstance(other, UnitDtype):
            # Handle unit conversion logic
            return cls.find_common_unit(cls.unit, other.unit)
        return NotImplemented
```

## Usage Examples

### Basic Custom DType

```python
import numpy as np
from numpy_user_dtypes.unit_dtype import UnitDtype

# Create a unit dtype
length_dtype = UnitDtype('m')
time_dtype = UnitDtype('s')

# Create arrays with custom dtype
distances = np.array([1.0, 2.0, 3.0], dtype=length_dtype)
times = np.array([0.5, 1.0, 1.5], dtype=time_dtype)

# Operations work with unit awareness
velocities = distances / times  # Results in m/s unit
```

### Custom Enum DType

```python
from enum import Enum
import numpy as np

class Status(Enum):
    ACTIVE = 1
    INACTIVE = 2
    PENDING = 3

class EnumDtype(dtype):
    """Dtype for enumerated values"""

    def __init__(self, enum_class):
        self.enum_class = enum_class
        super().__init__()

    def item(self, value):
        """Convert integer to enum"""
        return self.enum_class(value)

# Usage
status_dtype = EnumDtype(Status)
statuses = np.array([1, 2, 3, 1], dtype=status_dtype)

# Access enum values
for status in statuses:
    print(status.item())  # Status.ACTIVE, Status.INACTIVE, etc.
```

### Complex Number with Units

```python
class ComplexUnitDtype(dtype):
    """Complex numbers with physical units"""

    def __init__(self, unit):
        self.unit = unit
        super().__init__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle universal functions"""
        if ufunc in (np.add, np.subtract):
            # Unit consistency check
            return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        elif ufunc == np.multiply:
            # Unit multiplication logic
            return self.handle_multiplication(*inputs, **kwargs)
        # ... other operations

# Usage
impedance_dtype = ComplexUnitDtype('Ω')
impedances = np.array([1+2j, 3+4j], dtype=impedance_dtype)
```

### Custom String DType

```python
class FixedStringDtype(dtype):
    """Fixed-length string dtype with encoding"""

    def __init__(self, length, encoding='utf-8'):
        self.length = length
        self.encoding = encoding
        super().__init__()

    def __str__(self):
        return f"FixedString(length={self.length}, encoding={self.encoding})"

# Usage
name_dtype = FixedStringDtype(20, 'utf-8')
names = np.array(['Alice', 'Bob', 'Charlie'], dtype=name_dtype)
```

## API Documentation

### Core Classes

#### `dtype` (Base Class)

The base class for all custom dtypes.

```python
class dtype:
    """Base class for NumPy dtypes"""

    def __new__(cls, *args, **kwargs):
        """Create new dtype instance"""
        pass

    @classmethod
    def __common_dtype__(cls, other):
        """Find common dtype with another dtype class"""
        pass

    def __common_instance__(self, other):
        """Find common instance with another dtype instance"""
        pass
```

#### Essential Methods

##### Type Discovery

```python
@classmethod
def discover_type_from_scalar(cls, scalar):
    """
    Discover if scalar can be represented by this dtype.

    Parameters:
    -----------
    scalar : any
        Scalar value to check

    Returns:
    --------
    dtype or None
        Dtype instance if compatible, None otherwise
    """
    pass
```

##### Casting Operations

```python
def can_cast_from(self, other):
    """
    Check if casting from another dtype is possible.

    Parameters:
    -----------
    other : dtype
        Source dtype

    Returns:
    --------
    bool
        True if casting is possible
    """
    pass

def cast_from(self, other, array):
    """
    Cast array from another dtype to this dtype.

    Parameters:
    -----------
    other : dtype
        Source dtype
    array : ndarray
        Array to cast

    Returns:
    --------
    ndarray
        Casted array
    """
    pass
```

##### Common DType Resolution

```python
@classmethod
def __common_dtype__(cls, other):
    """
    Find common dtype between this and another dtype class.

    Parameters:
    -----------
    other : dtype class
        Other dtype class

    Returns:
    --------
    dtype class or NotImplemented
        Common dtype class, or NotImplemented if no common dtype
    """
    pass

def __common_instance__(self, other):
    """
    Find common instance with another dtype instance.

    Parameters:
    -----------
    other : dtype instance
        Other dtype instance

    Returns:
    --------
    dtype instance
        Common dtype instance
    """
    pass
```

### Universal Function Integration

```python
def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """
    Handle universal function calls.

    Parameters:
    -----------
    ufunc : ufunc
        Universal function being called
    method : str
        Method name ('__call__', 'reduce', etc.)
    inputs : tuple
        Input arrays
    kwargs : dict
        Additional keyword arguments

    Returns:
    --------
    ndarray or NotImplemented
        Result of operation, or NotImplemented if not handled
    """
    pass
```

### Array Method Integration

```python
def __array_function__(self, func, types, args, kwargs):
    """
    Handle NumPy array function calls.

    Parameters:
    -----------
    func : callable
        NumPy function being called
    types : tuple
        Types of arguments
    args : tuple
        Positional arguments
    kwargs : dict
        Keyword arguments

    Returns:
    --------
    any or NotImplemented
        Result of function, or NotImplemented if not handled
    """
    pass
```

## Advanced Features

### Custom Universal Functions

```python
from numpy.core.umath import create_ufunc

# Create custom ufunc for unit operations
unit_multiply = create_ufunc(
    'unit_multiply',
    nin=2, nout=1,
    types=[
        (UnitDtype, UnitDtype, UnitDtype),  # Unit * Unit -> Unit
        (UnitDtype, float, UnitDtype),      # Unit * scalar -> Unit
        (float, UnitDtype, UnitDtype),      # scalar * Unit -> Unit
    ],
    doc="Multiply with unit handling"
)

# Register with dtype
UnitDtype.__array_ufunc__ = unit_multiply
```

### Custom Reductions

```python
class CustomReductionDtype(dtype):
    """Dtype with custom reduction operations"""

    def reduce_sum(self, array, axis=None, dtype=None, out=None):
        """Custom sum reduction"""
        # Implementation
        pass

    def reduce_mean(self, array, axis=None, dtype=None, out=None):
        """Custom mean reduction"""
        # Implementation
        pass

    def reduce_max(self, array, axis=None, dtype=None, out=None):
        """Custom max reduction"""
        # Implementation
        pass
```

### Memory Management

```python
class MemoryEfficientDtype(dtype):
    """Dtype with custom memory management"""

    def itemsize(self):
        """Return size of individual items in bytes"""
        return self._itemsize

    def alignment(self):
        """Return required alignment for this dtype"""
        return self._alignment

    def newbyteorder(self, byteorder):
        """Return dtype with different byte order"""
        # Implementation
        pass
```

### Serialization Support

```python
class SerializableDtype(dtype):
    """Dtype with serialization support"""

    def __reduce__(self):
        """Support for pickle serialization"""
        return (self.__class__, (self._param1, self._param2))

    def __getstate__(self):
        """Get state for serialization"""
        return {'param1': self._param1, 'param2': self._param2}

    def __setstate__(self, state):
        """Set state during deserialization"""
        self._param1 = state['param1']
        self._param2 = state['param2']
```

## Integration with NumPy

### Type Hinting Support

```python
from typing import TypeVar, Generic
import numpy.typing as npt

# Generic array type
T = TypeVar('T', bound=dtype)

class TypedArray(Generic[T]):
    """Array with specific dtype type hinting"""

    def __init__(self, data: npt.NDArray[T]):
        self.data = data

# Usage
IntArray = TypedArray[np.dtype[np.int64]]
UnitArray = TypedArray[UnitDtype]
```

### NumPy Function Integration

```python
# Register custom dtype with NumPy functions
np.register_dtype(UnitDtype)

# Now NumPy functions work with custom dtype
arr = np.array([1, 2, 3], dtype=UnitDtype('m'))
result = np.sum(arr)  # Uses custom sum implementation
```

### Pandas Integration

```python
import pandas as pd

# Custom dtype works with pandas
df = pd.DataFrame({
    'distance': pd.Series([1.0, 2.0, 3.0], dtype=UnitDtype('m')),
    'time': pd.Series([0.5, 1.0, 1.5], dtype=UnitDtype('s'))
})

# Operations preserve dtype information
df['velocity'] = df['distance'] / df['time']
```

### Dask Integration

```python
import dask.array as da

# Custom dtype works with Dask arrays
dask_arr = da.from_array(np.array([1, 2, 3], dtype=UnitDtype('m')))
result = dask_arr.sum().compute()  # Uses custom implementation
```

## Performance Considerations

### Optimization Strategies

#### 1. Efficient Memory Layout

```python
class OptimizedDtype(dtype):
    """Dtype with optimized memory layout"""

    def __init__(self):
        # Use contiguous memory
        self._itemsize = 8  # 64-bit
        self._alignment = 8  # 64-bit aligned
```

#### 2. Vectorized Operations

```python
def vectorized_operation(self, array):
    """Use vectorized operations when possible"""
    # Leverage NumPy's vectorization
    return np.vectorize(self.element_operation)(array)
```

#### 3. Caching

```python
class CachedDtype(dtype):
    """Dtype with caching for expensive operations"""

    def __init__(self):
        self._cache = {}

    def cached_operation(self, key, operation):
        """Cache expensive operations"""
        if key not in self._cache:
            self._cache[key] = operation()
        return self._cache[key]
```

### Performance Benchmarks

```python
import timeit

def benchmark_dtype_operations():
    """Benchmark custom dtype operations"""

    # Setup
    custom_arr = np.array([1, 2, 3, 4, 5], dtype=UnitDtype('m'))
    regular_arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    # Benchmark operations
    custom_time = timeit.timeit(
        lambda: np.sum(custom_arr),
        number=10000
    )

    regular_time = timeit.timeit(
        lambda: np.sum(regular_arr),
        number=10000
    )

    print(f"Custom dtype: {custom_time:.6f}s")
    print(f"Regular dtype: {regular_time:.6f}s")
    print(f"Ratio: {custom_time / regular_time:.2f}x")
```

### Memory Usage

```python
def analyze_memory_usage():
    """Analyze memory usage of custom dtypes"""

    # Compare memory usage
    custom_arr = np.array([1, 2, 3, 4, 5], dtype=UnitDtype('m'))
    regular_arr = np.array([1, 2, 3, 4, 5], dtype=np.float64)

    print(f"Custom dtype itemsize: {custom_arr.dtype.itemsize} bytes")
    print(f"Regular dtype itemsize: {regular_arr.dtype.itemsize} bytes")
    print(f"Memory ratio: {custom_arr.dtype.itemsize / regular_arr.dtype.itemsize:.2f}x")
```

### Best Practices

1. **Minimize Overhead**: Keep dtype metadata small
2. **Leverage NumPy**: Use NumPy's optimized operations when possible
3. **Cache Results**: Cache expensive computations
4. **Profile Regularly**: Monitor performance with real data
5. **Consider Trade-offs**: Balance features vs performance

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ImportError: cannot import name CustomDtype`

**Solution**: Ensure proper build and installation:
```bash
# Rebuild extensions
python setup.py build_ext --inplace

# Check installation
python -c "from your_module import CustomDtype; print('OK')"
```

#### 2. Dtype Registration

**Problem**: Custom dtype not recognized by NumPy

**Solution**: Register dtype with NumPy:
```python
import numpy as np

# Register dtype class
np.core._multiarray_umath.register_dtype(CustomDtype)

# Verify registration
print(CustomDtype in np.core._multiarray_umath._registered_dtypes)
```

#### 3. Casting Issues

**Problem**: `TypeError: Cannot cast from dtype to dtype`

**Solution**: Implement proper casting methods:
```python
def can_cast_from(self, other):
    """Implement casting compatibility check"""
    if isinstance(other, (int, float)):
        return True
    return super().can_cast_from(other)

def cast_from(self, other, array):
    """Implement casting logic"""
    if isinstance(other, (int, float)):
        return array.astype(self)
    raise TypeError(f"Cannot cast from {other} to {self}")
```

#### 4. Common DType Resolution

**Problem**: Operations between different dtypes fail

**Solution**: Implement `__common_dtype__`:
```python
@classmethod
def __common_dtype__(cls, other):
    """Find common dtype with another dtype"""
    if isinstance(other, cls):
        return cls
    elif other == np.dtype('float64'):
        return cls  # Can handle float64
    return NotImplemented
```

#### 5. Universal Function Issues

**Problem**: Custom ufuncs not working

**Solution**: Implement `__array_ufunc__`:
```python
def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """Handle universal function calls"""
    if ufunc in (np.add, np.subtract):
        # Handle addition/subtraction
        return self.handle_arithmetic(ufunc, method, *inputs, **kwargs)
    return NotImplemented
```

### Debugging Tips

#### 1. Enable Debug Mode

```python
import numpy as np

# Enable NumPy debug mode
np.seterr(all='raise')

# Enable custom dtype debugging
CustomDtype.debug = True
```

#### 2. Use Verbose Operations

```python
# Add debug prints to custom methods
def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    print(f"Calling {ufunc} with method {method}")
    print(f"Input types: {[type(x) for x in inputs]}")
    # ... implementation
```

#### 3. Profile Memory Usage

```python
import tracemalloc

def profile_memory():
    """Profile memory usage of custom dtype operations"""
    tracemalloc.start()

    # Perform operations
    arr = np.array([1, 2, 3, 4, 5], dtype=CustomDtype())
    result = np.sum(arr)

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024:.2f} KB")
    print(f"Peak memory usage: {peak / 1024:.2f} KB")

    tracemalloc.stop()
```

### Compatibility Issues

#### NumPy Version Compatibility

```python
import numpy as np

def check_numpy_version():
    """Check NumPy version compatibility"""
    version = np.__version__
    major, minor = map(int, version.split('.')[:2])

    if major < 1 or (major == 1 and minor < 24):
        raise ValueError("NumPy 1.24+ required for custom dtypes")

    print(f"NumPy {version} - Compatible")
```

#### Platform-Specific Issues

```python
import platform

def check_platform():
    """Check platform compatibility"""
    system = platform.system()

    if system == "Windows":
        print("Windows detected - check MSVC compatibility")
    elif system == "Linux":
        print("Linux detected - check GCC version")
    elif system == "Darwin":
        print("macOS detected - check Clang version")
```

### Getting Help

#### Community Resources

1. **NumPy Discourse**: https://numpy.discourse.group/
2. **GitHub Issues**: https://github.com/numpy/numpy/issues
3. **Stack Overflow**: Tag with `numpy` and `dtype`

#### Reporting Issues

When reporting issues, include:

1. **NumPy version**: `print(numpy.__version__)`
2. **Python version**: `print(sys.version)`
3. **Minimal reproducible example**
4. **Error traceback**
5. **Expected vs actual behavior**

#### Contributing

1. **Fork the repository**
2. **Create feature branch**
3. **Add tests**
4. **Update documentation**
5. **Submit pull request**

## Conclusion

The NumPy user-defined dtype system, as demonstrated in the `numpy-user-dtypes` repository, represents a significant advancement in making NumPy more extensible and user-friendly. While still experimental, these examples provide a foundation for:

- **Custom Data Types**: Create domain-specific data representations
- **Better Integration**: Seamless operation with NumPy ecosystem
- **Type Safety**: Proper type hinting and validation
- **Performance**: Optimized operations for custom types

### Key Takeaways

1. **Experimental Status**: API is not finalized and subject to change
2. **Educational Value**: Excellent learning resource for dtype developers
3. **Foundation**: Building blocks for future dtype implementations
4. **Community**: Active development and community support

### Next Steps

1. **Experiment**: Try the examples in the repository
2. **Contribute**: Help improve the implementations
3. **Follow Development**: Track NEP 42 progress
4. **Provide Feedback**: Share experiences and suggestions

The future of NumPy's dtype system is bright, and these examples pave the way for a more extensible and powerful scientific computing ecosystem.
