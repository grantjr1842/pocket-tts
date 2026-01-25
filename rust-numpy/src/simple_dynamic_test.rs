//! Simple test of dynamic kernel registry core functionality

use rust_numpy::dynamic_kernel_registry::{get_best_kernel, register_kernel};

#[test]
fn test_basic_registration() -> Result<(), Box<dyn std::any::Any + Send + Sync>> {
    use rust_numpy::dtype::Dtype;

    struct SimpleAdd;

    impl rust_numpy::dynamic_kernel_registry::KernelFunction for SimpleAdd {
        fn execute(
            &self,
            _inputs: &[&dyn rust_numpy::ufunc::ArrayView],
            _outputs: &mut [&mut dyn rust_numpy::ufunc::ArrayViewMut],
            _where_mask: Option<&rust_numpy::array::Array<bool>>,
        ) -> rust_numpy::error::Result<()> {
            Ok(())
        }

        fn name(&self) -> &str {
            "simple_add"
        }

        fn supported_dtypes(&self) -> &[rust_numpy::dtype::DtypeKind] {
            &[rust_numpy::dtype::DtypeKind::Integer]
        }

        fn performance_tier(&self) -> rust_numpy::dynamic_kernel_registry::PerformanceTier {
            rust_numpy::dynamic_kernel_registry::PerformanceTier::Generic
        }
    }

    let kernel = std::sync::Arc::new(SimpleAdd);
    let dtypes = vec![Dtype::from_type::<i32>(), Dtype::from_type::<i32>()];

    register_kernel("simple_add", kernel, dtypes)?;

    // Test retrieval
    let retrieved = get_best_kernel("simple_add", &dtypes);
    assert!(retrieved.is_some());

    Ok(())
}
