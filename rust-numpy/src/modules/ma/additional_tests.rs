//! Additional tests for the masked array module

#[cfg(test)]
mod tests {
    use crate::array::Array;
    use crate::error::NumPyError;
    use crate::modules::ma::MaskedArray;

    #[test]
    fn test_masked_array_creation() -> Result<(), NumPyError> {
        let data = Array::from_vec(vec![1, 2, 3, 4]);
        let mask = Array::from_vec(vec![false, true, false, true]);
        let masked = MaskedArray::new(data, mask)?;
        assert_eq!(masked.size(), 4);
        Ok(())
    }

    #[test]
    fn test_masked_array_flatten() -> Result<(), NumPyError> {
        let data = Array::from_vec(vec![1, 2, 3, 4]);
        let mask = Array::from_vec(vec![false, true, false, true]);
        let masked = MaskedArray::new(data, mask)?;
        let flattened = masked.flatten();
        assert_eq!(flattened.size(), 4);
        Ok(())
    }
}
