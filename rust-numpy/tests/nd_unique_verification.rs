use numpy::{array2, array3, set_ops::unique};

#[test]
fn test_unique_2d_axis_0() {
    let a = array2![[1, 2], [3, 4], [1, 2], [5, 6]];
    let res = unique(&a, true, true, true, Some(&[0])).unwrap();

    assert_eq!(res.values.shape(), &[3, 2]);
    assert_eq!(res.values.to_vec(), vec![1, 2, 3, 4, 5, 6]);

    assert_eq!(res.indices.unwrap().to_vec(), vec![0, 1, 3]);
    assert_eq!(res.inverse.unwrap().to_vec(), vec![0, 1, 0, 2]);
    assert_eq!(res.counts.unwrap().to_vec(), vec![2, 1, 1]);
}

#[test]
fn test_unique_2d_axis_1() {
    let a = array2![[1, 3, 1], [2, 4, 2]];
    let res = unique(&a, true, true, true, Some(&[1])).unwrap();

    // Unique columns are [1, 2] and [3, 4]
    assert_eq!(res.values.shape(), &[2, 2]);
    // Note: lexicographical sort of columns.
    // Col 0: [1, 2], Col 1: [3, 4], Col 2: [1, 2]
    // Unique: [1, 2], [3, 4]
    // result.values should be [[1, 3], [2, 4]] in row-major order flattened is [1, 3, 2, 4]
    assert_eq!(res.values.to_vec(), vec![1, 3, 2, 4]);

    assert_eq!(res.indices.unwrap().to_vec(), vec![0, 1]);
    assert_eq!(res.counts.unwrap().to_vec(), vec![2, 1]);
}

#[test]
fn test_unique_3d_axis_2() {
    // 2x2x2 array
    let a = array3![[[1, 1], [2, 2]], [[3, 3], [4, 4]]];
    let res = unique(&a, false, false, false, Some(&[2])).unwrap();

    // Along axis 2, we have items of shape 2x2.
    // Item 0: [[1], [2]], [[3], [4]] (flattened 1,2,3,4)
    // Item 1: [[1], [2]], [[3], [4]] (flattened 1,2,3,4)
    // Both items are identical.

    // Values shape should be (2, 2, 1)
    assert_eq!(res.values.shape(), &[2, 2, 1]);
    assert_eq!(res.values.to_vec(), vec![1, 2, 3, 4]);
}
