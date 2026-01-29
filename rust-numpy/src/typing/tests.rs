#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::dtype::Dtype;
    use crate::typing::prelude::*;
    use crate::typing::DtypeLike;

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

    #[test]
    fn test_shape_trait() {
        // Test shape trait implementations
        use super::shape::*;

        // Test Dynamic shape
        let dynamic = Dynamic;
        assert_eq!(dynamic.NDIM, 0);
        assert_eq!(dynamic.SIZE, None);
        assert_eq!(dynamic.as_slice(), &[]);

        // Test Shape1
        let shape1 = Shape1::<10>;
        assert_eq!(shape1.NDIM, 1);
        assert_eq!(shape1.SIZE, Some(10));
        assert_eq!(shape1.as_slice(), &[10]);

        // Test Shape2
        let shape2 = Shape2::<3, 4>;
        assert_eq!(shape2.NDIM, 2);
        assert_eq!(shape2.SIZE, Some(12));
        assert_eq!(shape2.as_slice(), &[3, 4]);

        // Test Shape3
        let shape3 = Shape3::<2, 3, 4>;
        assert_eq!(shape3.NDIM, 3);
        assert_eq!(shape3.SIZE, Some(24));
        assert_eq!(shape3.as_slice(), &[2, 3, 4]);

        // Test Shape4
        let shape4 = Shape4::<2, 3, 4, 5>;
        assert_eq!(shape4.NDIM, 4);
        assert_eq!(shape4.SIZE, Some(120));
        assert_eq!(shape4.as_slice(), &[2, 3, 4, 5]);

        // Test RuntimeShape
        let runtime = RuntimeShape::new(vec![3, 4, 5]);
        assert_eq!(runtime.ndim(), 3);
        assert_eq!(runtime.size(), 60);
        assert_eq!(runtime.as_slice(), &[3, 4, 5]);
    }

    #[test]
    fn test_typed_array() {
        // Test TypedArray creation and methods
        use super::shape::*;

        // Test with runtime shape
        let arr = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let typed: TypedArray<f64, RuntimeShape> = TypedArray::new(
            arr.clone(),
            RuntimeShape::new(vec![3]),
        );
        assert_eq!(typed.ndim(), 0);
        assert_eq!(typed.size(), None);
        assert_eq!(typed.shape_slice(), &[3]);

        // Test with fixed shape
        let fixed: TypedArray<f64, Shape1<10>> = TypedArray::new(
            arr.clone(),
            Shape1::<10>,
        );
        assert_eq!(fixed.ndim(), 1);
        assert_eq!(fixed.size(), Some(10));
        assert_eq!(fixed.shape_slice(), &[10]);

        // Test 2D fixed shape
        let arr2d = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let matrix: TypedArray<f64, Shape2<2, 2>> = TypedArray::new(
            arr2d.clone(),
            Shape2::<2, 2>,
        );
        assert_eq!(matrix.ndim(), 2);
        assert_eq!(matrix.size(), Some(4));
        assert_eq!(matrix.shape_slice(), &[2, 2]);
    }

    #[test]
    fn test_typed_array_conversions() {
        // Test TypedArray conversions
        use super::shape::*;

        let arr = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let typed: TypedArray<f64, RuntimeShape> = TypedArray::new(
            arr.clone(),
            RuntimeShape::new(vec![3]),
        );

        // Test as_array
        assert_eq!(typed.as_array().shape(), &[3]);

        // Test as_array_mut
        let mut typed_mut = typed.clone();
        assert_eq!(typed_mut.as_array_mut().shape(), &[3]);

        // Test into_array
        let back_to_array: Array<f64> = typed.into_array();
        assert_eq!(back_to_array.shape(), &[3]);

        // Test From<Array<T>>
        let typed_from: TypedArray<f64, RuntimeShape> = Array::from_data(
            vec![1.0, 2.0],
            vec![2],
        ).into();
        assert_eq!(typed_from.as_array().shape(), &[2]);

        // Test From<TypedArray<T, S>>
        let arr_from: Array<f64> = typed.clone().into();
        assert_eq!(arr_from.shape(), &[3]);
    }

    #[test]
    fn test_typed_array_reshape() {
        // Test TypedArray reshape
        use super::shape::*;

        let arr = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let typed: TypedArray<f64, RuntimeShape> = TypedArray::new(
            arr.clone(),
            RuntimeShape::new(vec![4]),
        );

        // Reshape to 2D
        let reshaped = typed.reshape(Shape2::<2, 2>);
        assert_eq!(reshaped.ndim(), 2);
        assert_eq!(reshaped.size(), Some(4));
        assert_eq!(reshaped.shape_slice(), &[2, 2]);

        // Reshape to 1D
        let back_to_1d = reshaped.reshape(Shape1::<4>);
        assert_eq!(back_to_1d.ndim(), 1);
        assert_eq!(back_to_1d.size(), Some(4));
        assert_eq!(back_to_1d.shape_slice(), &[4]);
    }

    #[test]
    fn test_typed_array_clone() {
        // Test TypedArray Clone implementation
        use super::shape::*;

        let arr = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let typed: TypedArray<f64, RuntimeShape> = TypedArray::new(
            arr.clone(),
            RuntimeShape::new(vec![3]),
        );

        let cloned = typed.clone();
        assert_eq!(cloned.as_array().shape(), &[3]);
        assert_eq!(cloned.shape_slice(), &[3]);
    }

    #[test]
    fn test_enhanced_type_aliases() {
        // Test enhanced type aliases
        use super::shape::*;
        use super::*;

        // Test Array1 and Array2
        let arr1: Array1<f64> = Array::from_data(vec![1.0, 2.0], vec![2]).into();
        assert_eq!(arr1.as_array().shape(), &[2]);

        let arr2: Array2<f64> = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).into();
        assert_eq!(arr2.as_array().shape(), &[2, 2]);

        // Test Vector and Matrix
        let vec: Vector<f64> = Array::from_data(vec![1.0, 2.0, 3.0], vec![3]).into();
        assert_eq!(vec.as_array().shape(), &[3]);

        let mat: Matrix<f64> = Array::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).into();
        assert_eq!(mat.as_array().shape(), &[2, 2]);

        // Test FixedArray1
        let fixed1: FixedArray1<f64, 10> = Array::from_data(
            vec![1.0; 10],
            vec![10],
        ).into();
        assert_eq!(fixed1.ndim(), 1);
        assert_eq!(fixed1.size(), Some(10));

        // Test FixedArray2
        let fixed2: FixedArray2<f64, 2, 3> = Array::from_data(
            vec![1.0; 6],
            vec![2, 3],
        ).into();
        assert_eq!(fixed2.ndim(), 2);
        assert_eq!(fixed2.size(), Some(6));

        // Test dtype-specific aliases
        let float64_arr: Float64Array<RuntimeShape> = Array::from_data(
            vec![1.0, 2.0],
            vec![2],
        ).into();
        assert_eq!(float64_arr.as_array().shape(), &[2]);

        let int32_arr: Int32Array<RuntimeShape> = Array::from_data(
            vec![1, 2],
            vec![2],
        ).into();
        assert_eq!(int32_arr.as_array().shape(), &[2]);

        let bool_arr: BoolArray<RuntimeShape> = Array::from_data(
            vec![true, false],
            vec![2],
        ).into();
        assert_eq!(bool_arr.as_array().shape(), &[2]);
    }

    #[test]
    fn test_prelude_with_new_types() {
        // Test that prelude exports new types
        use super::prelude::*;

        // Shape types should be available
        let _ = Dynamic;
        let _ = Shape1::<10>;
        let _ = Shape2::<3, 4>;
        let _ = RuntimeShape::new(vec![3, 4]);

        // TypedArray should be available
        let arr = Array::from_data(vec![1.0, 2.0], vec![2]);
        let _ = TypedArray::<f64, RuntimeShape>::new(
            arr.clone(),
            RuntimeShape::new(vec![2]),
        );

        // Type aliases should be available
        let _ = Array1::<f64>;
        let _ = Array2::<f64>;
        let _ = Vector::<f64>;
        let _ = Matrix::<f64>;
        let _ = FixedArray1::<f64, 10>;
        let _ = FixedArray2::<f64, 2, 3>;
    }
}
