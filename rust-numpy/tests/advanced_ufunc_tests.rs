//! Tests for advanced ufunc features
//!
//! This module tests:
//! - Custom ufunc registration and execution
//! - Generalized ufunc (gufunc) framework
//! - Inner loop support
//! - Performance profiling
//! - Ufunc metadata

use rust_numpy::*;
use rust_numpy::ufunc::{ArrayView, ArrayViewMut};
use std::sync::Arc;

/// Simple custom ufunc implementation for testing
struct TestCustomUfunc {
    name: String,
    metadata: UfuncMetadata,
}

impl TestCustomUfunc {
    fn new(name: &str) -> Self {
        let metadata = UfuncMetadata::new(name.to_string(), 2, 1)
            .with_doc("Test custom ufunc for squared difference".to_string())
            .with_dtype(DtypeKind::Float)
            .with_dtype(DtypeKind::Integer);

        Self {
            name: name.to_string(),
            metadata,
        }
    }
}

impl CustomUfunc for TestCustomUfunc {
    fn name(&self) -> &str {
        &self.name
    }

    fn nin(&self) -> usize {
        self.metadata.nin
    }

    fn nout(&self) -> usize {
        self.metadata.nout
    }

    fn call(
        &self,
        inputs: &[&dyn ArrayView],
        outputs: &mut [&mut dyn ArrayViewMut],
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NumPyError::invalid_value("Expected 2 inputs and 1 output"));
        }

        // Get input arrays as f64
        let input1 = unsafe { &*(inputs[0] as *const _ as *const Array<f64>) };
        let input2 = unsafe { &*(inputs[1] as *const _ as *const Array<f64>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<f64>) };

        // Compute squared difference: (a - b)^2
        for i in 0..output.size() {
            if let (Some(a), Some(b)) = (input1.get(i), input2.get(i)) {
                let diff = a - b;
                output.set(i, diff * diff)?;
            }
        }

        Ok(())
    }

    fn inner_loop(
        &self,
        _args: &[&[u8]],
        _dimensions: &[isize],
        _steps: &[isize],
        _data: &mut [*mut u8],
    ) -> Result<()> {
        // For this test, inner loop is a no-op
        // Real implementations would process raw data here
        Ok(())
    }

    fn metadata(&self) -> UfuncMetadata {
        self.metadata.clone()
    }
}

/// Test gufunc implementation for matrix-vector multiplication
struct TestMatrixVectorGufunc {
    signature: GufuncSignature,
    metadata: UfuncMetadata,
}

impl TestMatrixVectorGufunc {
    fn new() -> Result<Self> {
        let signature = GufuncSignature::parse("(m,n),(n)->(m)")?;
        let metadata = UfuncMetadata::new("matvec".to_string(), 2, 1)
            .with_doc("Matrix-vector multiplication: (m,n) @ (n,) -> (m,)".to_string())
            .with_dtype(DtypeKind::Float);

        Ok(Self { signature, metadata })
    }
}

impl GeneralizedUfunc for TestMatrixVectorGufunc {
    fn name(&self) -> &str {
        "matvec"
    }

    fn signature(&self) -> &GufuncSignature {
        &self.signature
    }

    fn call_gufunc(
        &self,
        _inputs: &[&dyn ArrayView],
        _outputs: &mut [&mut dyn ArrayViewMut],
    ) -> Result<()> {
        // Simplified implementation - would do full matrix-vector multiply
        // in a real implementation
        Ok(())
    }

    fn inner_loop_gufunc(
        &self,
        _args: &[&[u8]],
        _core_dims: &[Vec<usize>],
        _steps: &[isize],
        _data: &mut [*mut u8],
    ) -> Result<()> {
        Ok(())
    }

    fn metadata(&self) -> UfuncMetadata {
        self.metadata.clone()
    }
}

#[test]
fn test_ufunc_metadata_creation() {
    let metadata = UfuncMetadata::new("test_ufunc".to_string(), 2, 1)
        .with_doc("Test ufunc".to_string())
        .with_dtype(DtypeKind::Float)
        .with_dtype(DtypeKind::Integer)
        .with_attribute("category".to_string(), "math".to_string());

    assert_eq!(metadata.name, "test_ufunc");
    assert_eq!(metadata.nin, 2);
    assert_eq!(metadata.nout, 1);
    assert!(metadata.doc.is_some());
    assert_eq!(metadata.supported_dtypes.len(), 2);
    assert_eq!(metadata.attributes.get("category"), Some(&"math".to_string()));
}

#[test]
fn test_performance_metrics() {
    let metrics = UfuncPerformanceMetrics::new();

    assert_eq!(metrics.call_count, 0);
    assert_eq!(metrics.total_time_us, 0);
    assert_eq!(metrics.total_elements, 0);
    assert_eq!(metrics.last_exec_time_us, 0);
    assert_eq!(metrics.avg_time_us(), 0.0);
    assert_eq!(metrics.throughput(), 0.0);

    // Simulate some activity
    let mut metrics = UfuncPerformanceMetrics::new();
    metrics.call_count = 100;
    metrics.total_time_us = 1000; // 1ms total
    metrics.total_elements = 10000;

    assert!((metrics.avg_time_us() - 10.0).abs() < 0.01);
    assert!(metrics.throughput() > 0.0);
}

#[test]
fn test_gufunc_signature_parse() {
    // Test matrix multiplication signature
    let sig = GufuncSignature::parse("(i,j),(j,k)->(i,k)").unwrap();
    assert_eq!(sig.signature, "(i,j),(j,k)->(i,k)");
    assert_eq!(sig.input_core_dims.len(), 2);
    assert_eq!(sig.output_core_dims.len(), 1);

    // Test invalid signature
    let result = GufuncSignature::parse("invalid");
    assert!(result.is_err());
}

#[test]
fn test_custom_ufunc_registry() {
    let registry = CustomUfuncRegistry::new();

    // Test initial state
    assert_eq!(registry.list_custom_ufuncs().len(), 0);
    assert_eq!(registry.list_gufuncs().len(), 0);
    assert!(!registry.is_profiling_enabled());

    // Test enabling profiling
    registry.set_profiling_enabled(true);
    assert!(registry.is_profiling_enabled());

    // Test getting metrics for non-existent ufunc
    assert!(registry.get_metrics("nonexistent").is_none());
}

#[test]
fn test_custom_ufunc_trait_impl() {
    let ufunc = TestCustomUfunc::new("squared_diff");

    assert_eq!(ufunc.name(), "squared_diff");
    assert_eq!(ufunc.nin(), 2);
    assert_eq!(ufunc.nout(), 1);

    let metadata = ufunc.metadata();
    assert_eq!(metadata.name, "squared_diff");
    assert!(metadata.doc.is_some());
}

#[test]
fn test_gufunc_trait_impl() {
    let gufunc = TestMatrixVectorGufunc::new().unwrap();

    assert_eq!(gufunc.name(), "matvec");
    assert_eq!(gufunc.signature().signature, "(m,n),(n)->(m)");

    let metadata = gufunc.metadata();
    assert_eq!(metadata.name, "matvec");
    assert!(metadata.doc.is_some());
}

#[test]
fn test_gufunc_calculate_loop_dims() {
    let mut sig = GufuncSignature::parse("(m,n),(n)->(m)").unwrap();

    // Test with arrays: (3, 4) matrix and (4,) vector
    // These exactly match the core dims, so no loop dims
    let shapes: Vec<&[usize]> = vec![&[3, 4], &[4]];
    let loop_dims = sig.calculate_loop_dims(&shapes).unwrap();

    // Should have 0 loop dimensions (shapes exactly match core dims)
    assert_eq!(loop_dims, 0);

    // Test with mismatched input count
    let shapes2: Vec<&[usize]> = vec![&[3_usize, 4]]; // Missing second input
    let result = sig.calculate_loop_dims(&shapes2);
    assert!(result.is_err());
}

#[test]
fn test_ufunc_metadata_clone() {
    let metadata1 = UfuncMetadata::new("test".to_string(), 2, 1)
        .with_doc("Test doc".to_string());

    let metadata2 = metadata1.clone();
    assert_eq!(metadata1.name, metadata2.name);
    assert_eq!(metadata1.doc, metadata2.doc);
}

#[test]
fn test_performance_metrics_clone() {
    let mut metrics1 = UfuncPerformanceMetrics::new();
    metrics1.call_count = 42;
    metrics1.total_time_us = 1000;

    let metrics2 = metrics1.clone();
    assert_eq!(metrics1.call_count, metrics2.call_count);
    assert_eq!(metrics1.total_time_us, metrics2.total_time_us);
}

#[test]
fn test_custom_ufunc_registry_thread_safety() {
    // The registry uses Arc and RwLock, so it should be thread-safe
    let registry = Arc::new(CustomUfuncRegistry::new());
    let registry_clone = Arc::clone(&registry);

    // Enable profiling from one "thread"
    registry.set_profiling_enabled(true);
    assert!(registry_clone.is_profiling_enabled());

    // Check we can read from both "threads"
    assert!(registry.is_profiling_enabled());
    assert!(registry_clone.is_profiling_enabled());
}

#[test]
fn test_gufunc_signature_error_handling() {
    // Test missing arrow
    let result = GufuncSignature::parse("(i,j),(j,k)");
    assert!(result.is_err());

    // Test empty input
    let result = GufuncSignature::parse("");
    assert!(result.is_err());

    // Test empty dimension parsing
    let sig = GufuncSignature::parse("()->()").unwrap();
    assert_eq!(sig.input_core_dims.len(), 1);
    assert_eq!(sig.output_core_dims.len(), 1);
    assert_eq!(sig.input_core_dims[0].len(), 0);
}

#[test]
fn test_multiple_custom_ufuncs() {
    let ufunc1 = TestCustomUfunc::new("ufunc1");
    let ufunc2 = TestCustomUfunc::new("ufunc2");

    // Verify they have independent metadata
    assert_ne!(ufunc1.name(), ufunc2.name());
    assert_eq!(ufunc1.metadata().name, "ufunc1");
    assert_eq!(ufunc2.metadata().name, "ufunc2");
}

#[test]
fn test_ufunc_metadata_builder_pattern() {
    let metadata = UfuncMetadata::new("builder_test".to_string(), 3, 2)
        .with_doc("Testing builder pattern".to_string())
        .with_dtype(DtypeKind::Float)
        .with_dtype(DtypeKind::Complex)
        .with_dtype(DtypeKind::Unsigned)
        .with_attribute("author".to_string(), "test".to_string())
        .with_attribute("version".to_string(), "1.0".to_string());

    assert_eq!(metadata.nin, 3);
    assert_eq!(metadata.nout, 2);
    assert_eq!(metadata.supported_dtypes.len(), 3);
    assert_eq!(metadata.attributes.len(), 2);
    assert_eq!(metadata.attributes.get("author"), Some(&"test".to_string()));
    assert_eq!(metadata.attributes.get("version"), Some(&"1.0".to_string()));
}
