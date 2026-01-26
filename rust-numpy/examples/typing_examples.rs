//! Examples demonstrating the usage of the typing module
//!
//! This file shows how to use the NDArray, ArrayLike, and DtypeLike
//! type annotations for better type safety and IDE support.

use rust_numpy::array;
use rust_numpy::array::Array;
use rust_numpy::error::NumPyError;
use rust_numpy::typing::prelude::*;

/// Example function using type annotations for arrays
fn process_numeric_data(
    integers: Int32Array,
    floats: Float64Array,
    mask: BoolArray,
) -> Float64Array {
    println!(
        "Processing {} integers, {} floats, {} boolean mask values",
        integers.size(),
        floats.size(),
        mask.size()
    );

    // Simple example: multiply floats by 2 where mask is true
    let result = floats.clone();
    for (i, &is_valid) in mask.iter().enumerate() {
        if is_valid && i < result.size() {
            // Note: This is a simplified example - in real code you'd use proper array operations
            println!(
                "  Processing index {}: {} -> {}",
                i,
                result[i],
                result[i] * 2.0
            );
        }
    }
    result
}

/// Example function using ArrayLike trait
fn convert_to_array<T, A>(data: A) -> Result<Array<T>, NumPyError>
where
    T: Clone + Default + 'static,
    A: ArrayLike<T>,
{
    data.to_array()
}

/// Example function using DtypeLike trait
fn create_dtype_description<D: DtypeLike>(dtype_like: D) -> String {
    let dtype = dtype_like.to_dtype();
    format!("Dtype: {:?}", dtype)
}

/// Example using generic NDArray type annotations
fn generic_array_operations<T>(arr: NDArray<T>) -> usize
where
    T: Clone + Default + 'static,
{
    println!("Array shape: {:?}", arr.shape());
    println!("Array dtype: {:?}", arr.dtype);
    arr.size()
}

fn main() -> Result<(), NumPyError> {
    println!("=== Typing Module Examples ===\n");

    // 1. Basic NDArray type annotations
    println!("1. Basic NDArray Type Annotations:");
    let int_array: Int32Array = array![1, 2, 3, 4, 5];
    let float_array: Float64Array = array![1.1, 2.2, 3.3, 4.4, 5.5];
    let bool_array: BoolArray = array![true, false, true, false, true];

    println!("  Int32Array: {:?}", int_array);
    println!("  Float64Array: {:?}", float_array);
    println!("  BoolArray: {:?}", bool_array);

    // 2. Process data with type annotations
    println!("\n2. Processing Data with Type Annotations:");
    let processed =
        process_numeric_data(int_array.clone(), float_array.clone(), bool_array.clone());
    println!("  Processed array length: {}", processed.size());

    // 3. ArrayLike trait examples
    println!("\n3. ArrayLike Trait Examples:");

    // Convert from Vec
    let vec_data = vec![10, 20, 30];
    let array_from_vec = convert_to_array(vec_data)?;
    println!("  From Vec: {:?}", array_from_vec);

    // Convert from array
    let array_data = [100, 200, 300];
    let array_from_array = convert_to_array(array_data)?;
    println!("  From array: {:?}", array_from_array);

    // Convert from slice
    let slice_data: [i32; 3] = [1000, 2000, 3000];
    let array_from_slice = convert_to_array(slice_data)?;
    println!("  From slice: {:?}", array_from_slice);

    // 4. DtypeLike trait examples
    println!("\n4. DtypeLike Trait Examples:");

    let dtype1 = create_dtype_description("int32");
    println!("  String dtype: {}", dtype1);

    let dtype2 = create_dtype_description(String::from("float64"));
    println!("  String dtype: {}", dtype2);

    let dtype3 = create_dtype_description(42i32);
    println!("  Primitive dtype: {}", dtype3);

    let dtype4 = create_dtype_description(3.14f64);
    println!("  Primitive dtype: {}", dtype4);

    let dtype5 = create_dtype_description(true);
    println!("  Primitive dtype: {}", dtype5);

    // 5. Generic NDArray operations
    println!("\n5. Generic NDArray Operations:");
    let length1 = generic_array_operations(int_array.clone());
    let length2 = generic_array_operations(float_array.clone());
    let length3 = generic_array_operations(bool_array.clone());
    println!(
        "  Generic processing lengths: {}, {}, {}",
        length1, length2, length3
    );

    // 6. Complex array types
    println!("\n6. Complex Array Types:");
    use num_complex::Complex;

    let complex64: Complex64Array = array![
        Complex::new(1.0f32, 2.0f32),
        Complex::new(3.0f32, 4.0f32),
        Complex::new(5.0f32, 6.0f32)
    ];
    println!("  Complex64Array: {:?}", complex64);

    let complex128: Complex128Array = array![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0)
    ];
    println!("  Complex128Array: {:?}", complex128);

    // 7. Unsigned integer types
    println!("\n7. Unsigned Integer Types:");
    let uint8_arr: UInt8Array = array![1u8, 2u8, 3u8];
    let uint16_arr: UInt16Array = array![1u16, 2u16, 3u16];
    let uint32_arr: UInt32Array = array![1u32, 2u32, 3u32];
    let uint64_arr: UInt64Array = array![1u64, 2u64, 3u64];

    println!("  UInt8Array: {:?}", uint8_arr);
    println!("  UInt16Array: {:?}", uint16_arr);
    println!("  UInt32Array: {:?}", uint32_arr);
    println!("  UInt64Array: {:?}", uint64_arr);

    // 8. Different float types
    println!("\n8. Different Float Types:");
    let float32_arr: Float32Array = array![1.0f32, 2.0f32, 3.0f32];
    let float64_arr: Float64Array = array![1.0, 2.0, 3.0];

    println!("  Float32Array: {:?}", float32_arr);
    println!("  Float64Array: {:?}", float64_arr);

    // 9. String dtype parsing examples
    println!("\n9. String Dtype Parsing:");
    let string_dtypes = vec![
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "bool",
        "complex64",
        "complex128",
        "np.int32",
        "np.float64", // with np. prefix
        "i4",
        "f8", // short aliases
    ];

    for dtype_str in string_dtypes {
        let dtype = dtype_str.to_dtype();
        println!("  '{}' -> {:?}", dtype_str, dtype);
    }

    // 10. Invalid dtype handling
    println!("\n10. Invalid Dtype Handling:");
    let invalid_dtypes = vec!["invalid_type", "unknown", "not_a_dtype"];
    for invalid in invalid_dtypes {
        let dtype = invalid.to_dtype();
        println!(
            "  '{}' (invalid) -> {:?} (fallback to Float64)",
            invalid, dtype
        );
    }

    println!("\n=== Examples Complete ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typing_examples() {
        // Test that all examples compile and run without panicking
        main().unwrap();
    }

    #[test]
    fn test_type_annotations() {
        // Test that type annotations work correctly
        let _: Int32Array = array![1, 2, 3];
        let _: Float64Array = array![1.0, 2.0, 3.0];
        let _: BoolArray = array![true, false, true];

        // Test ArrayLike
        let vec_data = vec![1, 2, 3];
        let _ = convert_to_array(vec_data).unwrap();

        // Test DtypeLike
        let _ = create_dtype_description("int32");
        let _ = create_dtype_description(42i32);
    }

    #[test]
    fn test_generic_operations() {
        let int_arr: Int32Array = array![1, 2, 3, 4, 5];
        let float_arr: Float64Array = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(generic_array_operations(int_arr), 5);
        assert_eq!(generic_array_operations(float_arr), 5);
    }
}
