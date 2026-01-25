//! Tests for dynamic kernel registry system

use crate::array::Array;
use crate::dtype::{Dtype, DtypeKind};
use crate::dynamic_kernel_registry::{
    get_best_kernel, get_registry_stats, list_registered_ufuncs, register_kernel, KernelFunction,
    PerformanceTier,
};
use crate::error::Result;
use crate::example_kernels::{create_example_kernels, register_example_kernels};
use std::sync::Arc;

/// Test kernel implementation
struct TestAddKernel;

impl KernelFunction for TestAddKernel {
    fn execute(
        &self,
        inputs: &[&dyn crate::ufunc::ArrayView],
        outputs: &mut [&mut dyn crate::ufunc::ArrayViewMut],
        _where_mask: Option<&Array<bool>>,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(crate::error::NumPyError::ufunc_error(
                "test_add",
                format!(
                    "Expected 2 inputs, 1 output, got {} inputs, {} outputs",
                    inputs.len(),
                    outputs.len()
                ),
            ));
        }

        let in0 = unsafe { &*(inputs[0] as *const _ as *const Array<i32>) };
        let in1 = unsafe { &*(inputs[1] as *const _ as *const Array<i32>) };
        let output = unsafe { &mut *(outputs[0] as *mut _ as *mut Array<i32>) };

        let broadcasted = crate::broadcasting::broadcast_arrays(&[in0, in1])?;
        for i in 0..output.size() {
            if let (Some(a), Some(b)) = (broadcasted[0].get(i), broadcasted[1].get(i)) {
                output.set(i, a + b)?;
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "test_add"
    }

    fn supported_dtypes(&self) -> &[DtypeKind] {
        &[DtypeKind::Integer]
    }

    fn performance_tier(&self) -> PerformanceTier {
        PerformanceTier::Generic
    }
}

#[test]
fn test_dynamic_kernel_registration() -> Result<()> {
    let kernel = Arc::new(TestAddKernel);
    let input_dtypes = vec![Dtype::from_type::<i32>(), Dtype::from_type::<i32>()];

    register_kernel("test_add", Box::new(kernel), input_dtypes.clone())?;
    let retrieved = get_best_kernel("test_add", &input_dtypes);
    assert!(retrieved.is_some());

    let retrieved_entry = retrieved.unwrap();
    assert_eq!(retrieved_entry.kernel().name(), "test_add");
    assert_eq!(retrieved_entry.tier(), PerformanceTier::Generic);

    Ok(())
}

#[test]
fn test_kernel_selection_by_performance_tier() -> Result<()> {
    register_example_kernels()?;

    let f64_dtypes = vec![Dtype::from_type::<f64>(), Dtype::from_type::<f64>()];
    let best_kernel = get_best_kernel("add", &f64_dtypes);
    assert!(best_kernel.is_some());

    let entry = best_kernel.unwrap();
    assert_eq!(entry.tier(), PerformanceTier::Vectorized);

    Ok(())
}

#[test]
fn test_dtype_matching() -> Result<()> {
    register_example_kernels()?;
    let f32_dtypes = vec![Dtype::from_type::<f32>(), Dtype::from_type::<f32>()];

    let best_kernel = get_best_kernel("add", &f32_dtypes);
    assert!(best_kernel.is_some());

    let entry = best_kernel.unwrap();
    assert_eq!(entry.tier(), PerformanceTier::Generic);

    Ok(())
}

#[test]
fn test_list_registered_ufuncs() -> Result<()> {
    register_example_kernels()?;

    let ufuncs = list_registered_ufuncs();
    assert!(ufuncs.contains(&"add".to_string()));
    assert!(ufuncs.contains(&"multiply".to_string()));

    Ok(())
}

#[test]
fn test_registry_stats() -> Result<()> {
    register_example_kernels()?;

    let stats = get_registry_stats();
    assert!(stats.is_some());

    let stats = stats.unwrap();
    assert!(stats.total_kernels > 0);
    assert!(stats.ufunc_count > 0);

    Ok(())
}

#[test]
fn test_kernel_matching_score() -> Result<()> {
    use crate::dynamic_kernel_registry::KernelEntry;

    let kernel = Arc::new(TestAddKernel);
    let entry = KernelEntry::new(
        kernel,
        vec![Dtype::from_type::<i32>(), Dtype::from_type::<i32>()],
    );

    let exact_match_dtypes = vec![Dtype::from_type::<i32>(), Dtype::from_type::<i32>()];

    let castable_dtypes = vec![Dtype::from_type::<i64>(), Dtype::from_type::<i64>()];

    let incompatible_dtypes = vec![Dtype::from_type::<f32>(), Dtype::from_type::<f32>()];
    let exact_score = entry.matching_score(&exact_match_dtypes);
    let castable_score = entry.matching_score(&castable_dtypes);
    let incompatible_score = entry.matching_score(&incompatible_dtypes);

    assert!(exact_score > castable_score);
    assert!(castable_score > incompatible_score);
    assert_eq!(incompatible_score, 0);

    Ok(())
}

#[test]
fn test_runtime_registration_workflow() -> Result<()> {
    let custom_kernel = Arc::new(TestAddKernel);
    let custom_dtypes = vec![Dtype::from_type::<i32>(), Dtype::from_type::<i32>()];
    register_kernel("custom_add", custom_kernel, custom_dtypes)?;

    let ufuncs = list_registered_ufuncs();
    assert!(ufuncs.contains(&"custom_add".to_string()));
    let retrieved_kernel = get_best_kernel("custom_add", &custom_dtypes);
    assert!(retrieved_kernel.is_some());

    let entry = retrieved_kernel.unwrap();
    assert_eq!(entry.tier(), PerformanceTier::Generic);

    Ok(())
}
