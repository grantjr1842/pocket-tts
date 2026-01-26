#[cfg(test)]
mod tests {
    use rust_numpy::*;

    #[test]
    fn test_expand_dims_basic() {
        let a = array![1, 2, 3];
        let expanded = expand_dims(&a, 0).unwrap();
        assert_eq!(expanded.shape(), &[1, 3]);
        assert_eq!(expanded.get(0), Some(&1));
        assert_eq!(expanded.get(1), Some(&2));
        assert_eq!(expanded.get(2), Some(&3));
    }

    #[test]
    fn test_expand_dims_axis_1() {
        let a = array![1, 2, 3];
        let expanded = expand_dims(&a, 1).unwrap();
        assert_eq!(expanded.shape(), &[3, 1]);
        assert_eq!(expanded.get(0), Some(&1));
        assert_eq!(expanded.get(1), Some(&2));
        assert_eq!(expanded.get(2), Some(&3));
    }

    #[test]
    fn test_expand_dims_negative_axis() {
        let a = array![1, 2, 3];
        let expanded = expand_dims(&a, -1).unwrap();
        assert_eq!(expanded.shape(), &[3, 1]);
    }

    #[test]
    fn test_expand_dims_2d() {
        let a = array2![[1, 2], [3, 4]];
        let expanded = expand_dims(&a, 0).unwrap();
        assert_eq!(expanded.shape(), &[1, 2, 2]);
    }

    #[test]
    fn test_expand_dims_2d_axis_1() {
        let a = array2![[1, 2], [3, 4]];
        let expanded = expand_dims(&a, 1).unwrap();
        assert_eq!(expanded.shape(), &[2, 1, 2]);
    }

    #[test]
    fn test_expand_dims_3d() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let a = Array::from_shape_vec(vec![2, 2, 2], data);
        let expanded = expand_dims(&a, 0).unwrap();
        assert_eq!(expanded.shape(), &[1, 2, 2, 2]);
    }

    #[test]
    fn test_expand_dims_multiple() {
        let a = array![1, 2];
        let expanded1 = expand_dims(&a, 0).unwrap();
        let expanded2 = expand_dims(&expanded1, 2).unwrap();
        assert_eq!(expanded2.shape(), &[1, 2, 1]);
    }

    #[test]
    fn test_expand_dims_negative_2() {
        let a = array2![[1, 2], [3, 4]];
        let expanded = expand_dims(&a, -2).unwrap();
        assert_eq!(expanded.shape(), &[2, 1, 2]);
    }

    #[test]
    fn test_expand_dims_scalar() {
        let a = Array::from_vec(vec![1]);
        let expanded = expand_dims(&a, 0).unwrap();
        assert_eq!(expanded.shape(), &[1, 1]);
    }

    #[test]
    fn test_expand_dims_invalid_axis() {
        let a = array![1, 2, 3];
        let result = expand_dims(&a, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_expand_dims_invalid_negative_axis() {
        let a = array![1, 2, 3];
        let result = expand_dims(&a, -5);
        assert!(result.is_err());
    }
}
