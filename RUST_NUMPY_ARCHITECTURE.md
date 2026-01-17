# Rust NumPy Library Architecture Plan

## Overview

This document outlines the architecture for a 100% pure-Rust NumPy library that achieves full API parity with Python's NumPy 2.1+. The library will be developed within the pocket-tts repository as `rust-numpy/`.

## Design Philosophy

1. **Complete API Parity**: Every NumPy function, class, and module will be available
2. **Rust-Native Design**: Leverages Rust's type system, memory safety, and performance
3. **Multi-Backend Support**: Pluggable backends (CPU, CUDA, Metal, WASM)
4. **C-ABI Compatibility**: Drop-in replacement for existing NumPy C-extensions
5. **Semantic Equivalence**: Same behavior, edge cases, and error handling as NumPy

## Project Structure

```
rust-numpy/
├── Cargo.toml
├── README.md
├── LICENSE
├── src/
│   ├── lib.rs                 # Main library entry point and public API
│   ├── array.rs               # Core ndarray implementation
│   ├── dtype.rs               # Complete dtype system
│   ├── memory.rs              # Memory management and views
│   ├── ufunc.rs               # Universal functions system
│   ├── broadcasting.rs        # Broadcasting logic
│   ├── strides.rs             # Stride calculations
│   ├── constants.rs           # NumPy constants (inf, nan, etc.)
│   └── modules/
│       ├── mod.rs             # Module exports
│       ├── core.rs            # Core array operations
│       ├── linalg.rs          # Linear algebra (LAPACK integration)
│       ├── fft.rs             # Fast Fourier Transform
│       ├── random.rs          # Random number generation
│       ├── lib.rs             # Additional library functions
│       ├── ma.rs              # Masked arrays
│       ├── strings.rs         # String operations
│       ├── datetime.rs        # Datetime support
│       └── polynomial.rs      # Polynomial operations
├── backends/
│   ├── mod.rs                 # Backend trait definition
│   ├── cpu/                   # CPU backend (Rayon + BLAS)
│   │   ├── mod.rs
│   │   ├── blas.rs            # OpenBLAS/Intel MKL integration
│   │   └── parallel.rs        # Rayon parallelization
│   ├── cuda/                  # CUDA backend
│   │   ├── mod.rs
│   │   ├── kernels/           # Custom CUDA kernels
│   │   └── cublas.rs          # cuBLAS integration
│   ├── metal/                 # Apple Metal backend
│   │   ├── mod.rs
│   │   └── kernels/           # Metal compute shaders
│   └── wasm/                  # WebAssembly backend
│       ├── mod.rs
│       └── js/                # WebGL/WASM implementations
├── c-api/                     # C-ABI compatibility layer
│   ├── mod.rs
│   ├── headers/               # C header files
│   └── python/                # Python C-API compatibility
├── tests/
│   ├── conformance/           # NumPy conformance tests
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── benchmarks/            # Performance benchmarks
├── examples/
│   ├── basic_usage.rs
│   ├── linear_algebra.rs
│   ├── signal_processing.rs
│   └── machine_learning.rs
├── docs/
│   ├── api/                   # API documentation
│   ├── tutorials/             # Tutorial notebooks
│   └── migration/             # Migration from NumPy guides
└── tools/
    ├── cli.rs                 # Command-line tool
    └── benchmark.rs           # Benchmarking utility
```

## Core Architecture

### 1. Array Data Structure

```rust
pub struct Array<T> {
    data: Box<[T]>,                    // Contiguous data storage
    shape: Vec<usize>,                 // Dimensions
    strides: Vec<isize>,               // Strides for each dimension
    dtype: Dtype,                      // Type information
    view_info: ViewInfo,               // View/slice metadata
    backend: Arc<dyn Backend>,         // Execution backend
}
```

### 2. Dtype System

Complete dtype parity including:
- All numeric types (int8-64, uint8-64, float16-128, complex64-256)
- String types (fixed-length, variable-length, Unicode)
- Datetime types (datetime64, timedelta64)
- Structured dtypes and record arrays
- Object dtypes for custom types

### 3. Backend System

```rust
pub trait Backend: Send + Sync {
    fn array_from_data<T>(&self, data: Vec<T>, shape: Vec<usize>) -> Result<Array<T>>;
    fn compute_ufunc(&self, ufunc: &Ufunc, inputs: &[&Array], output: &mut Array) -> Result<()>;
    fn linear_algebra(&self, op: LinAlgOp, inputs: &[&Array], output: &mut Array) -> Result<()>;
    fn fft(&self, op: FFTOp, input: &Array, output: &mut Array) -> Result<()>;
}
```

## Implementation Phases

### Phase 1: Core Foundation (High Priority)
1. **Array Structure**: Basic N-dimensional array with memory management
2. **Dtype System**: Complete dtype support with casting
3. **Core Operations**: Creation, indexing, slicing, reshaping
4. **Memory Model**: Views, broadcasting, stride calculations
5. **Basic Ufuncs**: Arithmetic operations (+, -, *, /, **)

### Phase 2: Universal Functions (Medium Priority)
1. **Ufunc System**: Complete ufunc framework with broadcasting
2. **Mathematical Functions**: sin, cos, exp, log, sqrt, etc.
3. **Comparison Functions**: >, <, ==, !=, logical operations
4. **Reduction Operations**: sum, mean, std, min, max, argmin/argmax

### Phase 3: Advanced Modules (Medium Priority)
1. **Linear Algebra**: Matrix operations, decompositions, eigensolvers
2. **FFT**: Discrete Fourier transforms and related operations
3. **Random**: Complete random number generation with all distributions

### Phase 4: Specialized Features (Low Priority)
1. **Masked Arrays**: ma module implementation
2. **String Operations**: Vectorized string operations
3. **Datetime**: Complete datetime64 support
4. **Polynomials**: Polynomial operations and fitting

### Phase 5: Performance & Integration (Low Priority)
1. **Multi-Backend**: CUDA, Metal, WASM backends
2. **C-ABI Layer**: Drop-in compatibility for Python extensions
3. **Benchmarks**: Comprehensive performance testing
4. **Documentation**: API docs and tutorials

## Technical Decisions

### Memory Management
- **Ownership Model**: Clear ownership with copy-on-write for views
- **Memory Layout**: Row-major and column-major support
- **Zero-Copy**: Views and slicing without data copying
- **Alignment**: Proper memory alignment for SIMD operations

### Performance Strategy
- **SIMD**: Use std::simd for vectorized operations
- **Parallelization**: Rayon for parallel execution
- **BLAS Integration**: OpenBLAS/MKL for linear algebra
- **Cache Optimization**: Memory access patterns optimized for cache locality

### Error Handling
- **Result Types**: Use Result<T, NumPyError> throughout
- **Error Types**: Comprehensive error types matching NumPy exceptions
- **Panic Safety**: No panics in library code, graceful degradation

### Compatibility Layer
- **C-API**: Full NumPy C-API compatibility
- **Python Integration**: Seamless PyO3 integration
- **Drop-in Replacement**: Can be imported as `numpy` replacement

## Integration with Pocket-TTS

Since pocket-tts currently uses NumPy for:
1. Audio processing (seanet.py, audio.py, audio_output.py)
2. Audio analysis (analyze_audio.py)
3. Potential Rust audio processing (rust_audio.py)

The Rust NumPy library will:
1. **Replace NumPy dependency** with Rust implementation
2. **Maintain Python compatibility** through PyO3 bindings
3. **Improve performance** for audio processing operations
4. **Enable Rust-only workflows** for better performance

## Success Criteria

1. **100% API Parity**: Every NumPy function available with identical behavior
2. **Performance Parity**: Equal or better performance than NumPy
3. **Conformance**: Pass all NumPy conformance tests
4. **Documentation**: Complete API documentation with examples
5. **Integration**: Seamless integration with existing Python code

## Risks and Mitigations

1. **Scope Creep**: Massive API surface - mitigate with phased approach
2. **Performance**: Matching NumPy's highly optimized code - mitigate with SIMD/BLAS
3. **Complexity**: NumPy has many edge cases - mitigate with comprehensive testing
4. **Maintenance**: Large codebase - mitigate with automated testing and CI

## Next Steps

1. Complete research on existing Rust array libraries
2. Set up basic project structure and build system
3. Implement core array structure and basic operations
4. Create comprehensive test suite
5. Gradually implement NumPy modules following the phases above