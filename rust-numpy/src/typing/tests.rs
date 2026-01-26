#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::dtype::Dtype;
    use crate::typing::{
        NDArray, Int32Array, Float64Array, BoolArray, Complex64Array,
        DtypeLike, ArrayLike,
    };

    #[test]
    fn test_ndarray_type_alias() {
        // Test that NDArray is just an alias for Array
        let arr: NDArray<f64> = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn test_specific_ndarray_types() {
        // Test specific dtype array types
        let int_arr: Int32Array = Array::from_data(vec![1, 2, 3], vec![3]);
        assert_eq!(int_arr.shape(), &[3]);

        let float_arr: Float64Array = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(float_arr.shape(), &[3]);

        let bool_arr: BoolArray = Array::from_data(vec![true, false, true], vec![3]);
        assert_eq!(bool_arr.shape(), &[3]);
    }

    #[test]
    fn test_array_like_trait() {
        // Test ArrayLike implementations

        // Test with Array
        let arr = Array::from_data(vec![1, 2, 3], vec![3]);
        let converted = arr.to_array().unwrap();
        assert_eq!(converted.shape(), &[3]);

        // Test with Vec
        let vec_data = vec![1, 2, 3];
        let converted = vec_data.to_array().unwrap();
        assert_eq!(converted.shape(), &[3]);

        // Test with array
        let array_data = [1, 2, 3];
        let converted = array_data.to_array().unwrap();
        assert_eq!(converted.shape(), &[3]);

        // Test with slice
        let slice_data = &[1, 2, 3];
        let converted = slice_data.to_array().unwrap();
        assert_eq!(converted.shape(), &[3]);
    }

    #[test]
    fn test_dtype_like_trait() {
        // Test DtypeLike implementations

        // Test with Dtype
        let dtype = Dtype::Int32 { byteorder: None };
        let converted = dtype.to_dtype();
        assert_eq!(dtype, converted);

        // Test with string
        let str_dtype = "int32";
        let converted = str_dtype.to_dtype();
        assert_eq!(converted, Dtype::Int32 { byteorder: None });

        // Test with String
        let string_dtype = String::from("float64");
        let converted = string_dtype.to_dtype();
        assert_eq!(converted, Dtype::Float64 { byteorder: None });

        // Test with primitive types
        let int_dtype: i32 = 42;
        let converted = int_dtype.to_dtype();
        assert_eq!(converted, Dtype::Int32 { byteorder: None });

        let float_dtype: f64 = 3.14;
        let converted = float_dtype.to_dtype();
        assert_eq!(converted, Dtype::Float64 { byteorder: None });

        let bool_dtype: bool = true;
        let converted = bool_dtype.to_dtype();
        assert_eq!(converted, Dtype::Bool);
    }

    #[test]
    fn test_dtype_like_invalid_string() {
        // Test that invalid strings fall back to Float64
        let invalid_dtype = "invalid_type";
        let converted = invalid_dtype.to_dtype();
        assert_eq!(converted, Dtype::Float64 { byteorder: None });
    }

    #[test]
    fn test_prelude_exports() {
        // Test that prelude exports work
        use crate::typing::prelude::*;

        // These should all be available
        let _: NDArray<f64> = Array::from_data(vec![1.0], vec![1]);
        let _: Int32Array = Array::from_data(vec![1], vec![1]);
        let _: Float64Array = Array::from_data(vec![1.0], vec![1]);
        let _: BoolArray = Array::from_data(vec![true], vec![1]);

        // Test that traits are available
        fn test_array_like<T: Clone + Default + 'static, A: ArrayLike<T>>(data: A) -> Array<T> {
            data.to_array().unwrap()
        }

        fn test_dtype_like<D: DtypeLike>(dtype: D) -> Dtype {
            dtype.to_dtype()
        }

        // These should compile
        let _ = test_array_like(vec![1, 2, 3]);
        let _ = test_dtype_like(42i32);
    }

    #[test]
    fn test_complex_array_types() {
        // Test complex array types
        use num_complex::Complex;

        let complex32_arr: Complex64Array = Array::from_data(
            vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
            vec![2],
        );
        assert_eq!(complex32_arr.shape(), &[2]);
    }

    #[test]
    fn test_comprehensive_dtype_coverage() {
        // Test all primitive dtype implementations
        assert_eq!(DtypeLike::to_dtype(&0i8), Dtype::Int8 { byteorder: None });
        assert_eq!(DtypeLike::to_dtype(&0i16), Dtype::Int16 { byteorder: None });
        assert_eq!(DtypeLike::to_dtype(&0i32), Dtype::Int32 { byteorder: None });
        assert_eq!(DtypeLike::to_dtype(&0i64), Dtype::Int64 { byteorder: None });
        assert_eq!(DtypeLike::to_dtype(&0u8), Dtype::UInt8 { byteorder: None });
        assert_eq!(
            DtypeLike::to_dtype(&0u16),
            Dtype::UInt16 { byteorder: None }
        );
        assert_eq!(
            DtypeLike::to_dtype(&0u32),
            Dtype::UInt32 { byteorder: None }
        );
        assert_eq!(
            DtypeLike::to_dtype(&0u64),
            Dtype::UInt64 { byteorder: None }
        );
        assert_eq!(
            DtypeLike::to_dtype(&0.0f32),
            Dtype::Float32 { byteorder: None }
        );
        assert_eq!(
            DtypeLike::to_dtype(&0.0f64),
            Dtype::Float64 { byteorder: None }
        );
        assert_eq!(DtypeLike::to_dtype(&true), Dtype::Bool);
    }
}
