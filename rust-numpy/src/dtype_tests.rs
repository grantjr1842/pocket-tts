#[cfg(test)]
mod tests {
    use crate::dtype::*;
    use crate::typing::*;

    #[test]
    fn test_bitwidth_from_str() {
        assert_eq!(Dtype::from_str("np.int8").unwrap(), Dtype::Int8 { byteorder: None });
        assert_eq!(Dtype::from_str("np.float64").unwrap(), Dtype::Float64 { byteorder: None });
    }

    #[test]
    fn test_dtype_getter_syntax() {
        assert_eq!(Dtype::from_str("dtype[int32]").unwrap(), Dtype::Int32 { byteorder: None });
        assert_eq!(Dtype::from_str("dtype[f8]").unwrap(), Dtype::Float64 { byteorder: None });
    }

    #[test]
    fn test_bitwidth_aliases() {
        // Just verify they compile
        let _i8: Int8 = nbit_8;
        let _f64: Float64 = nbit_64;
    }
}
