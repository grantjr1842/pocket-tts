#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    fn create_test_array() -> Array<i32> {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        Array::from_data(data, vec![data.len()])
    }

    #[test]
    fn test_sort_basic() {
        let mut array = create_test_array();
        let result = sort(&mut array, None, "quicksort", "asc").unwrap();
        let sorted = result.to_vec();
        assert_eq!(sorted, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_argsort_basic() {
        let array = create_test_array();
        let indices = argsort(&array, None, "quicksort", "asc").unwrap();
        assert_eq!(indices.to_vec(), vec![1, 3, 6, 0, 2, 5, 7, 4]);
    }

    #[test]
    fn test_sort_kinds() {
        let mut array = create_test_array();

        let result1 = sort(&mut array.clone(), None, "quicksort", "asc").unwrap();
        assert_eq!(result1.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);

        let result2 = sort(&mut array.clone(), None, "mergesort", "asc").unwrap();
        assert_eq!(result2.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);

        let result3 = sort(&mut array.clone(), None, "heapsort", "asc").unwrap();
        assert_eq!(result3.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn test_sort_orders() {
        let mut array = create_test_array();

        let result1 = sort(&mut array.clone(), None, "quicksort", "asc").unwrap();
        assert_eq!(result1.to_vec(), vec![1, 1, 2, 3, 4, 5, 6, 9]);

        let result2 = sort(&mut array.clone(), None, "quicksort", "desc").unwrap();
        assert_eq!(result2.to_vec(), vec![9, 6, 5, 4, 3, 2, 1, 1]);
    }

    #[test]
    fn test_sort_invalid_kind() {
        let mut array = create_test_array();
        let result = sort(&mut array, None, "invalid", "asc");
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_invalid_order() {
        let mut array = create_test_array();
        let result = sort(&mut array, None, "quicksort", "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_empty() {
        let data: Vec<i32> = vec![];
        let mut array = Array::from_data(data.clone(), vec![0]);
        let result = sort(&mut array, None, "quicksort", "asc").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_searchsorted() {
        let sorted_data = vec![1, 2, 2, 4, 5, 6, 9];
        let sorted_array = Array::from_data(sorted_data.clone(), vec![sorted_data.len()]);
        let search_data = vec![2, 3, 3, 7];
        let search_array = Array::from_data(search_data.clone(), vec![search_data.len()]);

        let indices_left = searchsorted(&sorted_array, &search_array, "left", None).unwrap();
        assert_eq!(indices_left.to_vec(), vec![1, 1, 1, 4, 5]);

        let indices_right = searchsorted(&sorted_array, &search_array, "right", None).unwrap();
        assert_eq!(indices_right.to_vec(), vec![2, 2, 2, 5, 6]);
    }

    #[test]
    fn test_searchsorted_invalid_side() {
        let sorted_data = vec![1, 2, 4];
        let sorted_array = Array::from_data(sorted_data.clone(), vec![sorted_data.len()]);
        let search_data = vec![3];
        let search_array = Array::from_data(search_data.clone(), vec![search_data.len()]);

        let result = searchsorted(&sorted_array, &search_array, "invalid", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract() {
        let condition_data = vec![true, false, true, false];
        let condition = Array::from_data(condition_data.clone(), vec![condition_data.len()]);
        let array_data = vec![1, 2, 3, 4];
        let array = Array::from_data(array_data.clone(), vec![array_data.len()]);

        let result = extract(&condition, &array).unwrap();
        assert_eq!(result.to_vec(), vec![1, 3]);
    }

    #[test]
    fn test_extract_shape_mismatch() {
        let condition_data = vec![true, false];
        let condition = Array::from_data(condition_data.clone(), vec![condition_data.len()]);
        let array_data = vec![1, 2, 3];
        let array = Array::from_data(array_data.clone(), vec![array_data.len()]);

        let result = extract(&condition, &array);
        assert!(result.is_err());
    }

    #[test]
    fn test_count_nonzero() {
        let data = vec![0, 1, 0, 2, 0, 3];
        let array = Array::from_data(data.clone(), vec![data.len()]);
        let count = count_nonzero(&array).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_count_nonzero_empty() {
        let data: Vec<i32> = vec![];
        let array = Array::from_data(data.clone(), vec![0]);
        let count = count_nonzero(&array).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_flatnonzero() {
        let data = vec![0, 1, 0, 2, 0, 3];
        let array = Array::from_data(data.clone(), vec![data.len()]);
        let indices = flatnonzero(&array).unwrap();
        assert_eq!(indices.to_vec(), vec![1, 3]);
    }

    #[test]
    fn test_flatnonzero_empty() {
        let data: Vec<i32> = vec![];
        let array = Array::from_data(data.clone(), vec![0]);
        let indices = flatnonzero(&array).unwrap();
        assert!(indices.is_empty());
    }

    #[test]
    fn test_argwhere_1d() {
        let data = vec![0, 1, 0, 2, 0, 3];
        let array = Array::from_data(data.clone(), vec![data.len()]);
        let result = argwhere(&array).unwrap();
        // Shape should be (3, 1) for 3 non-zero elements in 1D array
        assert_eq!(result.shape(), &[3, 1]);
        // Each row should contain the index
        assert_eq!(result.to_vec(), vec![1, 3, 5]);
    }

    #[test]
    fn test_argwhere_2d() {
        let data = vec![1, 0, 0, 0, 2, 0, 0, 0, 3];
        let array = Array::from_data(data.clone(), vec![3, 3]);
        let result = argwhere(&array).unwrap();
        // Shape should be (3, 2) for 3 non-zero elements in 2D array
        assert_eq!(result.shape(), &[3, 2]);
        // Each row should be [row, col] coordinates
        // Non-zero elements at: (0,0), (1,1), (2,2)
        assert_eq!(result.to_vec(), vec![0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn test_argwhere_3d() {
        let data = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let array = Array::from_data(data.clone(), vec![3, 3, 3]);
        let result = argwhere(&array).unwrap();
        // Shape should be (1, 3) for 1 non-zero element in 3D array
        assert_eq!(result.shape(), &[1, 3]);
        // Row should be [dim0, dim1, dim2] coordinate
        assert_eq!(result.to_vec(), vec![0, 0, 0]);
    }

    #[test]
    fn test_argwhere_empty() {
        let data = vec![0, 0, 0];
        let array = Array::from_data(data.clone(), vec![3]);
        let result = argwhere(&array).unwrap();
        // Shape should be (0, 1) for empty result in 1D array
        assert_eq!(result.shape(), &[0, 1]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_argwhere_2d_empty() {
        let data = vec![0, 0, 0, 0, 0, 0, 0, 0, 0];
        let array = Array::from_data(data.clone(), vec![3, 3]);
        let result = argwhere(&array).unwrap();
        // Shape should be (0, 2) for empty result in 2D array
        assert_eq!(result.shape(), &[0, 2]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_argwhere_bool() {
        let data = vec![true, false, true, false];
        let array = Array::from_data(data.clone(), vec![4]);
        let result = argwhere(&array).unwrap();
        // Shape should be (2, 1) for 2 true values in 1D array
        assert_eq!(result.shape(), &[2, 1]);
        // True values at indices 0 and 2
        assert_eq!(result.to_vec(), vec![0, 2]);
    }

    #[test]
    fn test_argwhere_all_nonzero() {
        let data = vec![1, 2, 3];
        let array = Array::from_data(data.clone(), vec![3]);
        let result = argwhere(&array).unwrap();
        // Shape should be (3, 1) for 3 non-zero elements
        assert_eq!(result.shape(), &[3, 1]);
        // All indices present
        assert_eq!(result.to_vec(), vec![0, 1, 2]);
    }

    #[test]
    fn test_argmax() {
        let data = vec![1, 5, 3, 9, 2, 6];
        let array = Array::from_data(data.clone(), vec![data.len()]);
        let index = argmax(&array, None, None, false).unwrap();
        assert_eq!(index.to_vec()[0], 3); // index of 9
    }

    #[test]
    fn test_argmin() {
        let data = vec![1, 5, 3, 9, 2, 6];
        let array = Array::from_data(data.clone(), vec![data.len()]);
        let index = argmin(&array, None, None, false).unwrap();
        assert_eq!(index.to_vec()[0], 0);
    }

    #[test]
    fn test_argmax_empty() {
        let data: Vec<i32> = vec![];
        let array = Array::from_data(data.clone(), vec![0]);
        let result = argmax(&array, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_argmin_empty() {
        let data: Vec<i32> = vec![];
        let array = Array::from_data(data.clone(), vec![0]);
        let result = argmin(&array, None, None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_where_condition_only() {
        let condition_data = vec![true, false, true];
        let condition = Array::from_data(condition_data.clone(), vec![condition_data.len()]);

        let result = where_(&condition, None, None).unwrap();
        assert_eq!(result.to_vec(), vec![1, 1]);
    }

    // ===== argwhere tests =====

    #[test]
    fn test_argwhere_1d() {
        // 1D array: [0, 1, 0, 2, 0, 3]
        let data = vec![0, 1, 0, 2, 0, 3];
        let array = Array::from_data(data.clone(), vec![data.len()]);

        let result = argwhere(&array).unwrap();

        // Should return shape (3, 1) with indices [[1], [3], [5]]
        assert_eq!(result.shape(), &[3, 1]);
        assert_eq!(result.to_vec(), vec![1, 3, 5]);
    }

    #[test]
    fn test_argwhere_2d() {
        // 2D array: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        let data = vec![1, 0, 0, 0, 2, 0, 0, 0, 3];
        let array = Array::from_data(data.clone(), vec![3, 3]);

        let result = argwhere(&array).unwrap();

        // Should return shape (3, 2) with coordinates:
        // [[0, 0], [1, 1], [2, 2]]
        assert_eq!(result.shape(), &[3, 2]);
        let coords = result.to_vec();
        // First non-zero at (0, 0)
        assert_eq!(coords[0], 0); // row 0
        assert_eq!(coords[1], 0); // col 0
                                  // Second non-zero at (1, 1)
        assert_eq!(coords[2], 1); // row 1
        assert_eq!(coords[3], 1); // col 1
                                  // Third non-zero at (2, 2)
        assert_eq!(coords[4], 2); // row 2
        assert_eq!(coords[5], 2); // col 2
    }

    #[test]
    fn test_argwhere_3d() {
        // 3D array with shape (2, 2, 2)
        // [[[1, 0], [0, 0]], [[0, 0], [0, 2]]]
        let data = vec![1, 0, 0, 0, 0, 0, 0, 2];
        let array = Array::from_data(data.clone(), vec![2, 2, 2]);

        let result = argwhere(&array).unwrap();

        // Should return shape (2, 3) with coordinates:
        // [[0, 0, 0], [1, 1, 1]]
        assert_eq!(result.shape(), &[2, 3]);
        let coords = result.to_vec();
        // First non-zero at (0, 0, 0)
        assert_eq!(coords[0], 0);
        assert_eq!(coords[1], 0);
        assert_eq!(coords[2], 0);
        // Second non-zero at (1, 1, 1)
        assert_eq!(coords[3], 1);
        assert_eq!(coords[4], 1);
        assert_eq!(coords[5], 1);
    }

    #[test]
    fn test_argwhere_boolean() {
        // Boolean array: [true, false, true, false, true]
        let data = vec![true, false, true, false, true];
        let array = Array::from_data(data.clone(), vec![data.len()]);

        let result = argwhere(&array).unwrap();

        // Should return shape (3, 1) with indices [[0], [2], [4]]
        assert_eq!(result.shape(), &[3, 1]);
        assert_eq!(result.to_vec(), vec![0, 2, 4]);
    }

    #[test]
    fn test_argwhere_all_zeros() {
        // Array with all zeros
        let data = vec![0, 0, 0, 0];
        let array = Array::from_data(data.clone(), vec![data.len()]);

        let result = argwhere(&array).unwrap();

        // Should return empty array with shape (0, 1)
        assert_eq!(result.shape(), &[0, 1]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_argwhere_empty() {
        // Empty array
        let data: Vec<i32> = vec![];
        let array = Array::from_data(data.clone(), vec![0]);

        let result = argwhere(&array).unwrap();

        // Should return empty array with shape (0, 1)
        assert_eq!(result.shape(), &[0, 1]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_argwhere_all_nonzero() {
        // Array with all non-zero elements
        let data = vec![1, 2, 3];
        let array = Array::from_data(data.clone(), vec![data.len()]);

        let result = argwhere(&array).unwrap();

        // Should return shape (3, 1) with indices [[0], [1], [2]]
        assert_eq!(result.shape(), &[3, 1]);
        assert_eq!(result.to_vec(), vec![0, 1, 2]);
    }
}
