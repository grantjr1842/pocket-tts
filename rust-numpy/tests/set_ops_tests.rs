use numpy::{
    array, array2,
    set_ops::{in1d, intersect1d, isin, setdiff1d, setxor1d, union1d, unique, SetOps},
};

#[test]
fn test_intersect1d() {
    let a = array![1, 2, 3, 2, 1];
    let b = array![2, 3, 4];
    let res = intersect1d(&a, &b).unwrap();
    assert_eq!(res.to_vec(), vec![2, 3]);
}

#[test]
fn test_union1d() {
    let a = array![1, 2, 3];
    let b = array![2, 3, 4];
    let res = union1d(&a, &b).unwrap();
    assert_eq!(res.to_vec(), vec![1, 2, 3, 4]);
}

#[test]
fn test_setdiff1d() {
    let a = array![1, 2, 3, 2, 4, 1];
    let b = array![2, 3];
    let res = setdiff1d(&a, &b).unwrap();
    assert_eq!(res.to_vec(), vec![1, 4]);
}

#[test]
fn test_setxor1d() {
    let a = array![1, 2, 3];
    let b = array![2, 3, 4];
    let res = setxor1d(&a, &b).unwrap();
    assert_eq!(res.to_vec(), vec![1, 4]);
}

#[test]
fn test_in1d() {
    let a = array![1, 2, 3, 4, 5];
    let b = array![2, 4, 6];
    let res = in1d(&a, &b, false).unwrap();
    assert_eq!(res.to_vec(), vec![false, true, false, true, false]);
}

#[test]
fn test_isin() {
    let a = array![1, 2, 3, 4, 5];
    let b = array![2, 4, 6];
    let res = isin(&a, &b).unwrap();
    assert_eq!(res.to_vec(), vec![false, true, false, true, false]);
}

#[test]
fn test_unique_axis_0() {
    let a = array2![[1, 1], [2, 3], [1, 1], [2, 3], [4, 5]];
    let res = unique(&a, false, false, false, Some(&[0])).unwrap();
    assert_eq!(res.values.shape(), vec![3, 2]);
    let flat_vals = res.values.to_vec();
    // Sorted unique rows: [1,1], [2,3], [4,5]
    assert_eq!(flat_vals, vec![1, 1, 2, 3, 4, 5]);
}

#[test]
fn test_unique_axis_1() {
    let a = array2![[1, 2, 1, 2], [1, 3, 1, 3]];
    let res = unique(&a, false, false, false, Some(&[1])).unwrap();
    assert_eq!(res.values.shape(), vec![2, 2]);
    let flat_vals = res.values.to_vec();
    // Unique columns: [[1,1], [2,3]]
    // flat: [1, 2, 1, 3]
    assert_eq!(flat_vals, vec![1, 2, 1, 3]);
}

#[test]
fn test_unique_rows() {
    let a = array2![[1, 1], [2, 3], [1, 1]];
    let res = SetOps::unique_rows(&a).unwrap();
    assert_eq!(res.shape(), vec![2, 2]);
    assert_eq!(res.to_vec(), vec![1, 1, 2, 3]);
}

#[test]
fn test_unique_axis_with_counts() {
    let a = array2![[1, 1], [2, 3], [1, 1], [4, 5]];
    let res = unique(&a, false, false, true, Some(&[0])).unwrap();
    assert_eq!(res.values.shape(), vec![3, 2]);
    assert_eq!(res.counts.unwrap().to_vec(), vec![2, 1, 1]); // [1,1] appears twice
}

#[test]
fn test_unique_axis_with_index() {
    let a = array2![[1, 1], [2, 3], [1, 1], [4, 5]];
    // rows at index 0 and 2 are identical. first occurrence is at 0.
    // sorted unique rows: [1,1] (from index 0), [2,3] (from index 1), [4,5] (from index 3)
    let res = unique(&a, true, false, false, Some(&[0])).unwrap();
    assert_eq!(res.indices.unwrap().to_vec(), vec![0, 1, 3]);
}

#[test]
fn test_unique_axis_with_inverse() {
    let a = array2![[1, 1], [2, 3], [1, 1], [4, 5]];
    // row 0 -> unique row 0
    // row 1 -> unique row 1
    // row 2 -> unique row 0
    // row 3 -> unique row 2
    let res = unique(&a, false, true, false, Some(&[0])).unwrap();
    assert_eq!(res.inverse.unwrap().to_vec(), vec![0, 1, 0, 2]);
}
