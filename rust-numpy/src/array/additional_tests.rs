//! Additional tests for the array module

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::dtype::Dtype;
    use crate::error::NumPyError;

    #[test]
    fn test_array_creation() -> Result<(), NumPyError> {
        let arr = Array::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(arr.size(), 4);
        Ok(())
    }

    #[test]
    fn test_array_reshape() -> Result<(), NumPyError> {
        let arr = Array::from_vec(vec![1, 2, 3, 4]);
        let reshaped = arr.reshape(&[2, 2])?;
        assert_eq!(reshaped.shape(), &[2, 2]);
        Ok(())
    }
}
