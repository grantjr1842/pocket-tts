# Comprehensive Usage Guide: NumPy x86-simd-sort

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Core Functionality](#core-functionality)
4. [Usage Examples](#usage-examples)
5. [API Documentation](#api-documentation)
6. [Performance Considerations](#performance-considerations)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)
9. [Integration with NumPy](#integration-with-numpy)

## Repository Overview

### Purpose and Functionality

`x86-simd-sort` is a high-performance C++ template library that provides SIMD-accelerated sorting routines for built-in data types and custom objects. The library leverages AVX-512 and AVX2 instruction sets to deliver significant performance improvements over standard sorting algorithms.

### Key Features

- **SIMD Acceleration**: Automatic selection of optimal SIMD implementation based on processor capabilities
- **Multiple Data Types**: Support for 16-bit, 32-bit, and 64-bit integers and floats
- **Custom Object Sorting**: Sort user-defined structs/classes using key functions
- **Key-Value Sorting**: Parallel sorting of paired arrays
- **Argument Sorting**: Get sorted indices without modifying original data
- **Partial Sorting**: Select top-k elements or partial sorts
- **OpenMP Support**: Optional parallelization for large arrays

### Supported Architectures

- **AVX-512**: Latest Intel Xeon and Core processors
- **AVX2**: Intel Sandy Bridge and later, AMD Bulldozer and later
- **Fallback**: Works on any x86 processor with appropriate fallback

### Supported Data Types

| Category   | Types                             |
|------------|-----------------------------------|
| **16-bit** | `uint16_t`, `int16_t`, `_Float16` |
| **32-bit** | `uint32_t`, `int32_t`, `float`    |
| **64-bit** | `uint64_t`, `int64_t`, `double`   |

## Installation and Setup

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Meson build system (version 0.55+)
- For `_Float16` support: GCC 12.x or later

### Building from Source

```bash
# Clone the repository
git clone https://github.com/numpy/x86-simd-sort.git
cd x86-simd-sort

# Setup build directory
meson setup --buildtype release builddir
cd builddir

# Compile
meson compile

# Install (optional)
sudo meson install
```

### Build Options

```bash
# Enable tests and benchmarks
meson setup --buildtype release -Dbuild_tests=true -Dbuild_benchmarks=true builddir

# Enable OpenMP support
meson setup --buildtype release -Duse_openmp=true builddir
```

### Using as Meson Subproject

```meson
# In your meson.build file
xss = subproject('x86-simd-sort')
xss_dep = xss.get_variable('x86simdsortcpp_dep')

# In your executable
executable('my_program', 'main.cpp',
           dependencies: [xss_dep])
```

### Using with pkg-config

```bash
# After installation
pkg-config --cflags --libs x86simdsortcpp

# Compile your program
g++ -O3 main.cpp $(pkg-config --cflags --libs x86simdsortcpp) -o my_program
```

## Core Functionality

### 1. Basic Sorting

```cpp
#include "x86simdsort.h"

// Sort array in-place
x86simdsort::qsort(arr.data(), size, hasnan, descending);
```

### 2. Object Sorting

```cpp
// Sort custom objects using key function
x86simdsort::object_qsort(objects.data(), count, key_function);
```

### 3. Key-Value Sorting

```cpp
// Sort parallel arrays maintaining correspondence
x86simdsort::keyvalue_qsort(keys.data(), values.data(), size, hasnan, descending);
```

### 4. Argument Sorting

```cpp
// Get sorted indices
auto indices = x86simdsort::argsort(arr.data(), size, hasnan, descending);
```

### 5. Selection Operations

```cpp
// Find k-th smallest element
x86simdsort::qselect(arr.data(), k, size, hasnan, descending);

// Partial sort (top-k elements)
x86simdsort::partial_qsort(arr.data(), k, size, hasnan, descending);
```

## Usage Examples

### Basic Array Sorting

```cpp
#include "x86simdsort.h"
#include <vector>
#include <iostream>

int main() {
    // Create sample data
    std::vector<float> data = {3.14f, 1.59f, 2.65f, 0.5f, 4.2f};

    // Sort ascending (default)
    x86simdsort::qsort(data.data(), data.size(), false, false);

    // Sort descending
    x86simdsort::qsort(data.data(), data.size(), false, true);

    // Handle NaN values
    std::vector<float> data_with_nan = {3.14f, NAN, 1.59f, 2.65f};
    x86simdsort::qsort(data_with_nan.data(), data_with_nan.size(), true, false);

    return 0;
}
```

### Custom Object Sorting

```cpp
#include "x86simdsort.h"
#include <vector>
#include <cmath>

struct Point3D {
    double x, y, z;

    // Constructor
    Point3D(double x, double y, double z) : x(x), y(y), z(z) {}
};

int main() {
    std::vector<Point3D> points = {
        {1.0, 2.0, 3.0},
        {4.0, 1.0, 2.0},
        {2.0, 3.0, 1.0},
        {0.5, 1.5, 2.5}
    };

    // Sort by x-coordinate
    x86simdsort::object_qsort(points.data(), points.size(),
        [](const Point3D& p) { return p.x; });

    // Sort by distance from origin
    x86simdsort::object_qsort(points.data(), points.size(),
        [](const Point3D& p) {
            return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        });

    // Sort by Manhattan distance
    x86simdsort::object_qsort(points.data(), points.size(),
        [](const Point3D& p) {
            return std::abs(p.x) + std::abs(p.y) + std::abs(p.z);
        });

    return 0;
}
```

### Key-Value Sorting

```cpp
#include "x86simdsort.h"
#include <vector>
#include <string>

int main() {
    // Parallel arrays
    std::vector<int> scores = {85, 92, 78, 95, 88};
    std::vector<std::string> names = {"Alice", "Bob", "Charlie", "Diana", "Eve"};

    // Sort scores and names together
    x86simdsort::keyvalue_qsort(scores.data(), names.data(), scores.size(), false, false);

    // Now scores and names are sorted correspondingly
    // scores: 78, 85, 88, 92, 95
    // names:  "Charlie", "Alice", "Eve", "Bob", "Diana"

    return 0;
}
```

### Argument Sorting

```cpp
#include "x86simdsort.h"
#include <vector>
#include <iostream>

int main() {
    std::vector<double> data = {3.14, 1.59, 2.65, 0.5, 4.2};

    // Get sorted indices
    auto indices = x86simdsort::argsort(data.data(), data.size(), false, false);

    // Print sorted order
    std::cout << "Sorted indices: ";
    for (size_t idx : indices) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;

    // Access data in sorted order without modifying original
    std::cout << "Data in sorted order: ";
    for (size_t idx : indices) {
        std::cout << data[idx] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### Selection and Partial Sorting

```cpp
#include "x86simdsort.h"
#include <vector>
#include <iostream>

int main() {
    std::vector<int> data = {5, 2, 8, 1, 9, 3, 7, 4, 6};

    // Find median (k-th smallest element)
    size_t k = data.size() / 2;
    x86simdsort::qselect(data.data(), k, data.size(), false, false);
    // First k elements are the k smallest, unsorted

    // Partial sort - get top 3 elements
    x86simdsort::partial_qsort(data.data(), 3, data.size(), false, false);
    // First 3 elements are the 3 smallest, sorted

    return 0;
}
```

## API Documentation

### Function Signatures

#### Basic Sorting

```cpp
template<typename T>
void x86simdsort::qsort(T* arr, size_t size, bool hasnan = false, bool descending = false);
```

**Parameters:**
- `arr`: Pointer to array data
- `size`: Number of elements in array
- `hasnan`: Whether array contains NaN values (floating-point only)
- `descending`: Sort direction (false for ascending, true for descending)

#### Object Sorting

```cpp
template<typename T, typename U, typename Func>
void x86simdsort::object_qsort(T* arr, U arrsize, Func key_func);
```

**Parameters:**
- `arr`: Pointer to array of custom objects
- `arrsize`: Number of objects (any integer type)
- `key_func`: Lambda function to extract sorting key

**Key Function Signature:**
```cpp
[](T obj) -> key_t { return key_value; }
```

Where `key_t` must be one of: `float, uint32_t, int32_t, double, uint64_t, int64_t`

#### Key-Value Sorting

```cpp
template<typename T1, typename T2>
void x86simdsort::keyvalue_qsort(T1* key, T2* val, size_t size, bool hasnan = false, bool descending = false);
```

**Parameters:**
- `key`: Pointer to key array
- `val`: Pointer to value array
- `size`: Number of elements
- `hasnan`: Whether key array contains NaN values
- `descending`: Sort direction

#### Argument Sorting

```cpp
template<typename T>
std::vector<size_t> x86simdsort::argsort(T* arr, size_t size, bool hasnan = false, bool descending = false);

template<typename T>
std::vector<size_t> x86simdsort::argselect(T* arr, size_t k, size_t size, bool hasnan = false);
```

#### Selection Operations

```cpp
template<typename T>
void x86simdsort::qselect(T* arr, size_t k, size_t size, bool hasnan = false, bool descending = false);

template<typename T>
void x86simdsort::partial_qsort(T* arr, size_t k, size_t size, bool hasnan = false, bool descending = false);
```

### Return Types

- **Sorting functions**: `void` (in-place modification)
- **Argument functions**: `std::vector<size_t>` (sorted indices)

### Space Complexity

- **Basic sorting**: `O(1)` additional space
- **Object sorting**: `O(N)` additional space for keys and indices
- **Key-value sorting**: `O(1)` additional space

## Performance Considerations

### SIMD Acceleration

The library automatically selects the optimal implementation based on CPU capabilities:

1. **AVX-512**: Best performance on supported processors
2. **AVX2**: Good performance on older processors
3. **Scalar fallback**: Works on any x86 processor

### Performance Benchmarks

Typical speedup factors compared to `std::sort`:

| Data Type       | AVX-512 Speedup | AVX2 Speedup |
|-----------------|-----------------|--------------|
| 32-bit integers | 3-5x            | 2-3x         |
| 64-bit integers | 2-4x            | 1.5-2.5x     |
| 32-bit floats   | 3-5x            | 2-3x         |
| 64-bit doubles  | 2-4x            | 1.5-2.5x     |

### Object Sorting Performance

Performance varies based on key function complexity:

- **Simple key** (direct member access): 4-5x speedup
- **Moderate key** (basic arithmetic): 6-8x speedup
- **Complex key** (sqrt, trigonometric): 8-10x speedup

### OpenMP Parallelization

When enabled with `-Duse_openmp=true`:

- **Large arrays** (>10,000 elements): 2-3x additional speedup
- **Small arrays**: No significant benefit
- **Thread limit**: Maximum 16 threads
- **Control**: Set `OMP_NUM_THREADS` environment variable

### Best Practices

1. **Use appropriate data types**: Choose the smallest type that fits your needs
2. **Batch operations**: Process multiple arrays together when possible
3. **Memory alignment**: Ensure arrays are properly aligned for SIMD operations
4. **Avoid NaN handling overhead**: Set `hasnan=false` when no NaNs are present
5. **Consider object sorting cost**: For simple objects, standard sort might be faster

### Memory Usage

- **Basic sorting**: No additional memory beyond input
- **Object sorting**: Requires `N * sizeof(key_t) + N * sizeof(uint32_t)` bytes
- **Recommendation**: Use object sorting for arrays ≤ `UINT32_MAX` elements

## Advanced Features

### Custom Comparators

```cpp
#include "x86simdsort.h"
#include <vector>

// Custom comparison logic
struct CustomCompare {
    bool operator()(const MyType& a, const MyType& b) const {
        // Your comparison logic
        return a.value < b.value;
    }
};

int main() {
    std::vector<MyType> data;
    // Use with custom key function
    x86simdsort::object_qsort(data.data(), data.size(),
        [](const MyType& obj) { return obj.get_sort_key(); });

    return 0;
}
```

### Multi-Key Sorting

```cpp
// Sort by multiple criteria
x86simdsort::object_qsort(people.data(), people.size(),
    [](const Person& p) {
        // Create composite key
        return std::make_tuple(p.age, p.name, p.score);
    });
```

### Integration with Existing Code

```cpp
// Wrapper function for legacy code
template<typename T>
void fast_sort(std::vector<T>& data, bool descending = false) {
    if constexpr (std::is_floating_point_v<T>) {
        // Check for NaNs in floating point data
        bool hasnan = std::any_of(data.begin(), data.end(),
                                 [](T x) { return std::isnan(x); });
        x86simdsort::qsort(data.data(), data.size(), hasnan, descending);
    } else {
        x86simdsort::qsort(data.data(), data.size(), false, descending);
    }
}
```

## Troubleshooting

### Common Issues

#### 1. Compilation Errors

**Problem**: `error: 'x86simdsort' has not been declared`

**Solution**: Ensure you're including the correct header:
```cpp
#include "x86simdsort.h"  // Not <x86simdsort.h>
```

#### 2. Linking Errors

**Problem**: Undefined reference to x86simdsort functions

**Solution**:
- Install the library: `sudo meson install`
- Use pkg-config: `$(pkg-config --cflags --libs x86simdsortcpp)`
- Or use as Meson subproject

#### 3. Runtime Performance Issues

**Problem**: No performance improvement over std::sort

**Solutions**:
- Check CPU supports AVX2/AVX-512: `grep avx2 /proc/cpuinfo`
- Ensure release build with optimizations: `-O3 -DNDEBUG`
- Verify data types are supported
- Check array size (small arrays may not benefit from SIMD)

#### 4. NaN Handling Issues

**Problem**: Incorrect sorting with NaN values

**Solution**: Always set `hasnan=true` when NaNs might be present:
```cpp
bool hasnan = std::any_of(arr.begin(), arr.end(),
                         [](T x) { return std::isnan(x); });
x86simdsort::qsort(arr.data(), arr.size(), hasnan, false);
```

#### 5. Object Sorting Performance

**Problem**: Object sorting slower than expected

**Solutions**:
- Keep key function simple and fast
- Avoid expensive operations in key function
- Consider caching keys if they're expensive to compute
- Benchmark with your specific use case

### Platform-Specific Considerations

#### Linux
- Ensure proper CPU features are available
- Use `-march=native` for best performance
- Consider NUMA effects for large arrays

#### Windows
- Use appropriate Visual Studio version (2017+)
- May need `/arch:AVX2` or `/arch:AVX512` flags
- Check Windows SDK compatibility

#### macOS
- Xcode 10+ recommended
- Apple Silicon uses different SIMD instructions (ARM NEON)
- May need Rosetta for Intel SIMD instructions

### Debugging Tips

1. **Enable assertions**: Build with debug flags for development
2. **Validate input**: Check array pointers and sizes
3. **Benchmark**: Compare with std::sort to verify improvements
4. **Profile**: Use perf/VTune to identify bottlenecks
5. **Test edge cases**: Empty arrays, single elements, duplicate values

## Integration with NumPy

### NumPy Integration

NumPy uses x86-simd-sort as a submodule to accelerate sorting operations:

```python
import numpy as np

# These operations use x86-simd-sort internally
arr = np.random.random(1000000)
sorted_arr = np.sort(arr)           # Uses qsort
indices = np.argsort(arr)           # Uses argsort
partitioned = np.partition(arr, k)  # Uses qselect
```

### Building NumPy with x86-simd-sort

NumPy automatically detects and uses x86-simd-sort when available:

```bash
# NumPy build will automatically use x86-simd-sort if present
python -m pip install numpy
```

### Performance Benefits in NumPy

Typical NumPy sorting speedups:

| Operation        | Speedup (AVX-512) | Speedup (AVX2) |
|------------------|-------------------|----------------|
| `np.sort()`      | 3-5x              | 2-3x           |
| `np.argsort()`   | 2-4x              | 1.5-2.5x       |
| `np.partition()` | 3-5x              | 2-3x           |

### Verifying NumPy Integration

```python
import numpy as np
print(np.show_config())  # Should show x86-simd-sort in build info
```

## Conclusion

x86-simd-sort provides significant performance improvements for sorting operations on x86 processors with SIMD capabilities. The library is particularly effective for:

- Large arrays of built-in data types
- Custom object sorting with complex key functions
- NumPy operations requiring high-performance sorting

Key takeaways:

1. **Automatic optimization**: Library selects best implementation based on hardware
2. **Easy integration**: Simple API compatible with existing code
3. **Significant speedups**: 2-10x performance improvements typical
4. **Broad compatibility**: Works with various data types and use cases
5. **NumPy integration**: Directly benefits NumPy sorting operations

For best results, benchmark with your specific data and use case, and consider the trade-offs between performance and code complexity.
