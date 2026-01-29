// use rust_numpy as numpy;

#[test]
fn test_exports_availability() {
    // These were marked as namespace_mismatch in the CSV
    let arr = numpy::Array::from_vec(vec![0.5]);
    let _ = numpy::arctanh(&arr);

    let arr2 = numpy::Array::from_vec(vec![1.5]);
    let _ = numpy::around(&arr2, 0);

    // Window functions
    let _ = numpy::bartlett::<f64>(10);

    // Broadcasting
    // Note: broadcast_arrays takes &[&Array]
    let _f: for<'a, 'b> fn(
        &'a [&'b numpy::Array<f64>],
    ) -> Result<Vec<numpy::Array<f64>>, numpy::NumPyError> = numpy::broadcast_arrays;

    println!("All symbols valid!");
}
