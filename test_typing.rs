// Simple test to verify ArrayLike and DtypeLike are implemented
use rust_numpy::{Array, ArrayLike, DtypeLike, Dtype};

fn main() {
    // Test ArrayLike
    let vec_data = vec![1, 2, 3];
    let array_data = Array::from_data(vec![1, 2, 3], vec![3]);
    
    // Test that these implement ArrayLike
    fn test_array_like<T: Clone + Default + 'static>(data: &dyn ArrayLike<T>) -> Array<T> {
        data.to_array().unwrap()
    }
    
    let _result1 = test_array_like(&vec_data);
    let _result2 = test_array_like(&array_data);
    
    // Test DtypeLike
    let dtype = Dtype::Int32 { byteorder: None };
    let str_dtype = "int32";
    let int_val: i32 = 42;
    
    fn test_dtype_like<T: DtypeLike>(data: &T) -> Dtype {
        data.to_dtype()
    }
    
    let _result1 = test_dtype_like(&dtype);
    let _result2 = test_dtype_like(&str_dtype);
    let _result3 = test_dtype_like(&int_val);
    
    println!("All typing tests passed!");
}
