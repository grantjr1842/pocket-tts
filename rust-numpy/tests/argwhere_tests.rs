use rust_numpy::sorting::argwhere;
use rust_numpy::Array;

#[test]
fn test_argwhere_1d() {
    let data = vec![0, 1, 0, 2, 0, 3];
    let array = Array::from_shape_vec(vec![data.len()], data);
    let result = argwhere(&array).unwrap();
    // Shape should be (3, 1) for 3 non-zero elements in 1D array
    assert_eq!(result.shape(), &[3, 1]);
    // Each row should contain the index
    assert_eq!(result.to_vec(), vec![1, 3, 5]);
}

#[test]
fn test_argwhere_2d() {
    let data = vec![1, 0, 0, 0, 2, 0, 0, 0, 3];
    let array = Array::from_shape_vec(vec![3, 3], data);
    let result = argwhere(&array).unwrap();
    // Shape should be (3, 2) for 3 non-zero elements in 2D array
    assert_eq!(result.shape(), &[3, 2]);
    // Each row should be [row, col] coordinates
    // Non-zero elements at: (0,0), (1,1), (2,2)
    assert_eq!(result.to_vec(), vec![0, 0, 1, 1, 2, 2]);
}

#[test]
fn test_argwhere_3d() {
    let data = vec![
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let array = Array::from_shape_vec(vec![3, 3, 3], data);
    let result = argwhere(&array).unwrap();
    // Shape should be (1, 3) for 1 non-zero element in 3D array
    assert_eq!(result.shape(), &[1, 3]);
    // Row should be [dim0, dim1, dim2] coordinate
    assert_eq!(result.to_vec(), vec![0, 0, 0]);
}

#[test]
fn test_argwhere_empty() {
    let data = vec![0, 0, 0];
    let array = Array::from_shape_vec(vec![3], data);
    let result = argwhere(&array).unwrap();
    // Shape should be (0, 1) for empty result in 1D array
    assert_eq!(result.shape(), &[0, 1]);
    assert!(result.is_empty());
}

#[test]
fn test_argwhere_2d_empty() {
    let data = vec![0, 0, 0, 0, 0, 0, 0, 0, 0];
    let array = Array::from_shape_vec(vec![3, 3], data);
    let result = argwhere(&array).unwrap();
    // Shape should be (0, 2) for empty result in 2D array
    assert_eq!(result.shape(), &[0, 2]);
    assert!(result.is_empty());
}

#[test]
fn test_argwhere_all_nonzero() {
    let data = vec![1, 2, 3];
    let array = Array::from_shape_vec(vec![3], data);
    let result = argwhere(&array).unwrap();
    // Shape should be (3, 1) for 3 non-zero elements
    assert_eq!(result.shape(), &[3, 1]);
    // All indices present
    assert_eq!(result.to_vec(), vec![0, 1, 2]);
}
