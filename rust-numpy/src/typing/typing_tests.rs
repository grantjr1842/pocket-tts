/// Tests for typing module organization and type aliases
///
/// This module tests all the type aliases and functionality provided by the
/// typing module to ensure they work correctly and match NumPy's typing API.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::dtype::Dtype;
    use crate::typing::{
        NDArray, Int8Array, Int16Array, Int32Array, Int64Array,
        UInt8Array, UInt16Array, UInt32Array, UInt64Array,
        Float32Array, Float64Array, BoolArray, Complex64Array, Complex128Array,
        ArrayLike, DtypeLike, ShapeLike, SupportsIndex,
    };
    use crate::typing::bitwidth::{Int8Bit, Int32Bit, Float64Bit, Complex128Bit};
    use crate::typing::dtype_getter::{DtypeGetter, ToDtype, dtype};

    #[test]
    fn test_ndarray_type_alias() {
        // Test that NDArray works as a type alias for Array
        let arr: NDArray<f64> = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(&1.0));
    }

    #[test]
    fn test_ndarray_type_aliases() {
        // Test all the specific ndarray type aliases
        let int8_arr: Int8Array = Array::from_data(vec![1i8, 2i8, 3i8], vec![3]);
        let int16_arr: Int16Array = Array::from_data(vec![1i16, 2i16, 3i16], vec![3]);
        let int32_arr: Int32Array = Array::from_data(vec![1i32, 2i32, 3i32], vec![3]);
        let int64_arr: Int64Array = Array::from_data(vec![1i64, 2i64, 3i64], vec![3]);

        let uint8_arr: UInt8Array = Array::from_data(vec![1u8, 2u8, 3u8], vec![3]);
        let uint16_arr: UInt16Array = Array::from_data(vec![1u16, 2u16, 3u16], vec![3]);
        let uint32_arr: UInt32Array = Array::from_data(vec![1u32, 2u32, 3u32], vec![3]);
        let uint64_arr: UInt64Array = Array::from_data(vec![1u64, 2u64, 3u64], vec![3]);

        let float32_arr: Float32Array = Array::from_data(vec![1.0f32, 2.0f32, 3.0f32], vec![3]);
        let float64_arr: Float64Array = Array::from_data(vec![1.0f64, 2.0f64, 3.0f64], vec![3]);

        let bool_arr: BoolArray = Array::from_data(vec![true, false, true], vec![3]);

        // Verify that all arrays have the correct length
        assert_eq!(int8_arr.len(), 3);
        assert_eq!(float64_arr.len(), 3);
        assert_eq!(bool_arr.len(), 3);
    }

    #[test]
    fn test_array_like_trait() {
        // Test ArrayLike trait implementations
        let vec_data = vec![1, 2, 3];
        let array_data = Array::from_data(vec![1, 2, 3], vec![3]);
        let array_data_ref = &array_data;

        // All should implement ArrayLike
        fn process_array_like<T: Clone + Default + 'static>(data: &dyn ArrayLike<T>) -> Array<T> {
            data.to_array().unwrap()
        }

        let _result1 = process_array_like(&vec_data);
        let _result2 = process_array_like(&array_data);
        let _result3 = process_array_like(array_data_ref);
    }

    #[test]
    fn test_dtype_like_trait() {
        // Test DtypeLike trait implementations
        let dtype = Dtype::Int32 { byteorder: None };
        let str_dtype = "int32";
        let string_dtype = String::from("float64");
        let int_val: i32 = 42;
        let float_val: f64 = 3.14;
        let bool_val: bool = true;

        // All should implement DtypeLike
        fn process_dtype_like<T: DtypeLike>(data: &T) -> Dtype {
            data.to_dtype()
        }

        let _result1 = process_dtype_like(&dtype);
        let _result2 = process_dtype_like(&str_dtype);
        let _result3 = process_dtype_like(&string_dtype);
        let _result4 = process_dtype_like(&int_val);
        let _result5 = process_dtype_like(&float_val);
        let _result6 = process_dtype_like(&bool_val);
    }

    #[test]
    fn test_shape_like_type() {
        // Test ShapeLike type alias
        let shape: ShapeLike = vec![10, 20, 30];
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0], 10);
    }

    #[test]
    fn test_supports_index_type() {
        // Test SupportsIndex type alias
        let index: SupportsIndex = 42isize;
        assert_eq!(index, 42);
    }

    #[test]
    fn test_prelude_exports() {
        // Test that prelude exports work correctly
        use super::super::prelude::*;

        // These should all be available from prelude
        let _arr: NDArray<f64> = Array::from_data(vec![1.0, 2.0], vec![2]);
        let _int_arr: Int32Array = Array::from_data(vec![1, 2], vec![2]);
        let _bool_arr: BoolArray = Array::from_data(vec![true, false], vec![2]);

        // Test that traits are available
        fn test_traits<T: ArrayLike<f64> + DtypeLike>(_: T) {}
        // This would require a type that implements both traits
    }

    #[test]
    fn test_bitwidth_types() {
        use super::super::bitwidth::*;

        // Test that all bit-width types implement NBitBase
        fn test_nbit_base<T: NBitBase>() -> (u8, bool, bool, bool) {
            (T::BITS as u8, T::SIGNED, T::FLOAT, T::COMPLEX)
        }

        // Test integer types
        let (bits, signed, float, complex) = test_nbit_base::<Int8Bit>();
        assert_eq!(bits, 8);
        assert!(signed);
        assert!(!float);
        assert!(!complex);

        let (bits, signed, float, complex) = test_nbit_base::<UInt32Bit>();
        assert_eq!(bits, 32);
        assert!(!signed);
        assert!(!float);
        assert!(!complex);

        // Test float types
        let (bits, signed, float, complex) = test_nbit_base::<Float64Bit>();
        assert_eq!(bits, 64);
        assert!(!signed);
        assert!(float);
        assert!(!complex);

        // Test complex types
        let (bits, signed, float, complex) = test_nbit_base::<Complex128Bit>();
        assert_eq!(bits, 128);
        assert!(!signed);
        assert!(!float);
        assert!(complex);
    }

    #[test]
    fn test_dtype_getter() {
        use super::super::dtype_getter::*;

        // Test DtypeGetter functionality
        let dtype = DtypeGetter::get::<Int32Bit>();
        match dtype {
            Dtype::Int32 { .. } => {} // Expected
            _ => panic!("Expected Int32 dtype"),
        }

        let dtype = DtypeGetter::get::<Float64Bit>();
        match dtype {
            Dtype::Float64 { .. } => {} // Expected
            _ => panic!("Expected Float64 dtype"),
        }

        let dtype = DtypeGetter::get::<Complex128Bit>();
        match dtype {
            Dtype::Complex128 { .. } => {} // Expected
            _ => panic!("Expected Complex128 dtype"),
        }
    }

    #[test]
    fn test_to_dtype_trait() {
        // Test ToDtype trait implementations
        let dtype = Int32Bit::to_dtype();
        match dtype {
            Dtype::Int32 { .. } => {} // Expected
            _ => panic!("Expected Int32 dtype"),
        }

        let dtype = Float64Bit::to_dtype();
        match dtype {
            Dtype::Float64 { .. } => {} // Expected
            _ => panic!("Expected Float64 dtype"),
        }
    }

    #[test]
    fn test_dtype_function() {
        // Test the convenience dtype function
        let dt = dtype::<Int32Bit>();
        match dt {
            Dtype::Int32 { .. } => {} // Expected
            _ => panic!("Expected Int32 dtype"),
        }

        let dt = dtype::<Float64Bit>();
        match dt {
            Dtype::Float64 { .. } => {} // Expected
            _ => panic!("Expected Float64 dtype"),
        }
    }

    #[test]
    fn test_type_aliases_compatibility() {
        // Test that type aliases work with generic functions
        fn process_ndarray<T>(arr: NDArray<T>) -> usize {
            arr.len()
        }

        let arr = Array::from_data(vec![1, 2, 3, 4], vec![4]);
        let length = process_ndarray(arr);
        assert_eq!(length, 4);

        // Test with specific array types
        fn process_int_array(arr: Int32Array) -> i32 {
            arr.get(0).copied().unwrap_or(0)
        }

        let int_arr = Array::from_data(vec![10, 20, 30], vec![3]);
        let first_val = process_int_array(int_arr);
        assert_eq!(first_val, 10);
    }

    #[test]
    fn test_complex_array_types() {
        // Test complex array types
        {
            use num_complex::Complex;

            let complex64_arr: Complex64Array = Array::from_data(
                vec![Complex::new(1.0f32, 2.0f32), Complex::new(3.0f32, 4.0f32)],
                vec![2],
            );
            assert_eq!(complex64_arr.len(), 2);

            let complex128_arr: Complex128Array = Array::from_data(
                vec![Complex::new(1.0f64, 2.0f64), Complex::new(3.0f64, 4.0f64)],
                vec![2],
            );
            assert_eq!(complex128_arr.len(), 2);
        }
    }

    #[test]
    fn test_module_organization() {
        // Test that the module organization matches NumPy's structure

        // 1. Core types should be available
        let _ndarray_type: std::any::TypeId = std::any::TypeId::of::<NDArray<f64>>();
        let _array_like_trait: std::any::TypeId = std::any::TypeId::of::<dyn ArrayLike<f64>>();
        let _dtype_like_trait: std::any::TypeId = std::any::TypeId::of::<dyn DtypeLike>();

        // 2. Bit-width types should be available
        let _int8_bit: std::any::TypeId = std::any::TypeId::of::<Int8Bit>();
        let _float64_bit: std::any::TypeId = std::any::TypeId::of::<Float64Bit>();
        let _complex128_bit: std::any::TypeId = std::any::TypeId::of::<Complex128Bit>();

        // 3. Dtype getter should be available
        let _dtype_getter: std::any::TypeId = std::any::TypeId::of::<DtypeGetter>();

        // 4. Type aliases should be available
        let _shape_like: std::any::TypeId = std::any::TypeId::of::<ShapeLike>();
        let _supports_index: std::any::TypeId = std::any::TypeId::of::<SupportsIndex>();

        // 5. Prelude should re-export commonly used types
        use crate::typing::prelude::*;
        let _prelude_ndarray: std::any::TypeId = std::any::TypeId::of::<NDArray<f64>>();
        let _prelude_array_like: std::any::TypeId = std::any::TypeId::of::<dyn ArrayLike<f64>>();
    }

    #[test]
    fn test_typing_module_completeness() {
        // This test ensures that all major NumPy typing components are available

        // ArrayLike - objects that can be converted to arrays
        fn test_array_like_completeness() {
            let vec_data = vec![1, 2, 3];
            let array_data = Array::from_data(vec![1, 2, 3], vec![3]);

            // Both should work with ArrayLike
            let _result1 = vec_data.to_array().unwrap();
            let _result2 = array_data.to_array().unwrap();
        }

        // DtypeLike - objects that can be converted to dtypes
        fn test_dtype_like_completeness() {
            let dtype = Dtype::Float64 { byteorder: None };
            let str_dtype = "float64";

            // Both should work with DtypeLike
            let _result1 = dtype.to_dtype();
            let _result2 = str_dtype.to_dtype();
        }

        // NDArray - runtime type annotations
        fn test_ndarray_completeness() {
            let arr: NDArray<f64> = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
            assert_eq!(arr.len(), 3);
        }

        // NBitBase - bit-width type hierarchy
        fn test_nbit_base_completeness() {
            use super::super::bitwidth::*;

            fn check_nbit_base<T: NBitBase>() {
                assert!(T::BITS > 0);
            }

            check_nbit_base::<Int8Bit>();
            check_nbit_base::<Float64Bit>();
            check_nbit_base::<Complex128Bit>();
        }

        test_array_like_completeness();
        test_dtype_like_completeness();
        test_ndarray_completeness();
        test_nbit_base_completeness();
    }
}
