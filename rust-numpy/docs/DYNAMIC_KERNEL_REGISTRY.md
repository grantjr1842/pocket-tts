# Dynamic Kernel Registry System

This document describes the implementation of a dynamic kernel registry system for the Rust NumPy library, providing runtime registration and dtype-specific optimization for ufunc operations.

## Overview

The dynamic kernel registry system allows:

1. **Runtime Registration**: Register new kernel implementations at runtime without recompilation
2. **Dtype-Specific Optimization**: Automatically select the best kernel implementation based on input dtypes and performance characteristics
3. **Performance Hints**: Guide kernel selection with performance characteristics (vectorized, memory-bound, compute-bound, etc.)
4. **Caching**: Cache kernel lookup results for improved performance

## Architecture

### Core Components

1. **Kernel Trait**: Base trait for all kernel implementations
2. **DynamicKernelRegistry**: Thread-safe registry with optimization cache
3. **Kernel Implementations**: Pre-built kernel types (Binary, Unary, SIMD, Memory-optimized)
4. **Runtime API**: High-level API for registration and execution

### Files

- `src/kernel_registry.rs` - Core registry and kernel trait
- `src/kernel_impls.rs` - Kernel implementation types
- `src/kernel_api.rs` - Runtime registration and execution API
- `src/kernel_tests.rs` - Comprehensive tests

## Usage Examples

### Basic Registration

```rust
use numpy::kernel_api::*;

// Register a simple binary kernel
register_binary_kernel("custom_add", |a: i32, b: i32| a + b)?;

// Register a unary kernel
register_unary_kernel("custom_neg", |a: f64| -a)?;
```

### Performance Hints

```rust
use numpy::kernel_registry::PerformanceHint;

// Register with performance hints for optimization
register_binary_kernel_with_hint(
    "vectorized_add",
    |a: f64, b: f64| a + b,
    PerformanceHint::Vectorized
)?;

register_binary_kernel_with_hint(
    "memory_opt_add",
    |a: f64, b: f64| a + b,
    PerformanceHint::MemoryBound
)?;
```

### Kernel Execution

```rust
use numpy::*;

// Create arrays
let a = array![1, 2, 3];
let b = array![4, 5, 6];

// Execute using dynamic kernel registry
let result = execute_binary("custom_add", &a, &b)?;

// Unary operations
let c = array![1.0, -2.0, 3.0];
let negated = execute_unary("custom_neg", &c)?;
```

### Registry Builder Pattern

```rust
use numpy::kernel_api::*;

// Initialize registry with common kernels
init_kernel_registry()?;

// Or use builder for custom registration
KernelRegistryBuilder::new()
    .with_binary_kernel("mul", |a: i32, b: i32| a * b)
    .with_unary_kernel("abs", |a: f64| a.abs())
    .with_common_kernels()  // Add standard kernels
    .build()?;
```

## Performance Optimization

### Kernel Selection Algorithm

The registry uses a scoring system to select the best kernel:

1. **Exact Type Matches**: +10 points per exact dtype match
2. **Castable Matches**: +5 points per castable dtype match
3. **Performance Hints**: Bonus points based on input characteristics
   - Vectorized: +20 points
   - Small input optimization: +15 points
   - Large input optimization: +15 points
   - Memory/Compute bound: +10 points

### Caching

Kernel lookup results are cached based on:
- Kernel name
- Input dtype signature

Cache statistics can be monitored:

```rust
let stats = get_registry_stats()?;
println!("Kernel names: {}", stats.kernel_names);
println!("Total implementations: {}", stats.total_implementations);
println!("Cache entries: {}", stats.cache_entries);
```

## Advanced Features

### Custom Kernel Implementations

```rust
use numpy::kernel_registry::*;
use std::sync::Arc;

struct CustomKernel {
    name: String,
}

impl Kernel for CustomKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> KernelSignature {
        KernelSignature::new(
            vec![Dtype::from_type::<f64>(), Dtype::from_type::<f64>()],
            vec![Dtype::from_type::<f64>()],
        )
    }

    fn execute(&self, inputs: &[&dyn ArrayView], outputs: &mut [&mut dyn ArrayViewMut]) -> Result<()> {
        // Custom implementation
        Ok(())
    }

    fn performance_hint(&self) -> PerformanceHint {
        PerformanceHint::Vectorized
    }
}

// Register custom kernel
let kernel = Arc::new(CustomKernel { name: "custom".to_string() });
register_kernel(kernel)?;
```

### SIMD and Memory Optimization

```rust
// SIMD-optimized kernel (placeholder for real SIMD implementation)
register_simd_binary_kernel("simd_add", |a: f64, b: f64| a + b)?;

// Memory-optimized kernel for large arrays
register_memory_optimized_kernel("mem_add", |a: f64, b: f64| a + b)?;
```

## Integration with Existing Ufunc System

The dynamic kernel registry is designed to complement the existing ufunc system:

```rust
// Both systems can coexist
let a = array![1.0, 2.0, 3.0];
let b = array![4.0, 5.0, 6.0];

// Dynamic kernel system
let dynamic_result = execute_binary("add", &a, &b)?;

// Existing ufunc system
let ufunc_result = a.add(&b, None, Casting::Safe)?;
```

## Testing

Comprehensive tests are provided in `src/kernel_tests.rs`:

- Basic registration and execution
- Performance hint functionality
- Dtype-specific optimization
- Registry builder pattern
- Cache behavior
- Integration with existing ufunc system
- Performance comparison

Run tests with:

```bash
cargo test kernel_tests
```

## Future Enhancements

1. **Real SIMD Implementation**: Replace placeholder SIMD kernels with actual intrinsics
2. **Automatic Kernel Generation**: Generate kernels from function signatures
3. **Plugin System**: Load kernels from external libraries
4. **Performance Profiling**: Automatic performance tuning based on runtime metrics
5. **GPU Support**: Extend registry to support GPU kernels

## Performance Considerations

- Registry lookup is O(1) with caching
- Kernel selection uses scoring algorithm for optimal choice
- Thread-safe design allows concurrent access
- Memory overhead is minimal due to Arc sharing
- Cache can be cleared if memory becomes constrained

## Error Handling

The system uses comprehensive error handling:

```rust
match execute_binary("nonexistent", &a, &b) {
    Ok(result) => println!("Success: {:?}", result),
    Err(NumPyError::UfuncError { ufunc, message }) => {
        eprintln!("Ufunc error in {}: {}", ufunc, message);
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

This dynamic kernel registry system provides a flexible, performant foundation for runtime-extensible numerical operations in Rust NumPy.
