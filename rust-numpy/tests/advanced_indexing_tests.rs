use numpy::*;

#[test]
fn test_fancy_indexing_1d() {
    let arr = array![10, 20, 30, 40, 50];
    let idx = array![0, 4, 1];

    let res = arr.fancy_index(&[&idx]).unwrap();
    assert_eq!(res.shape(), &[3]);
    assert_eq!(res.to_vec(), vec![10, 50, 20]);
}

#[test]
fn test_fancy_indexing_2d_coord() {
    // 3x3 array
    // [[0, 1, 2],
    //  [3, 4, 5],
    //  [6, 7, 8]]
    let arr = array2![[0, 1, 2], [3, 4, 5], [6, 7, 8]];

    // Select (0,0) and (2,2)
    let idx0 = array![0, 2];
    let idx1 = array![0, 2];

    let res = arr.fancy_index(&[&idx0, &idx1]).unwrap();
    assert_eq!(res.shape(), &[2]);
    assert_eq!(res.to_vec(), vec![0, 8]);
}

#[test]
fn test_fancy_indexing_2d_broadcast() {
    let arr = array2![[0, 1, 2], [3, 4, 5], [6, 7, 8]];

    // idx0: [0, 1] broadcasted to [2, 2] -> [[0, 1], [0, 1]]
    // idx1: [[0], [1]] broadcasted to [2, 2] -> [[0, 0], [1, 1]]
    // Pairs: (0,0), (1,0), (0,1), (1,1)

    let idx0 = array![0, 1];
    let idx1 = array2![[0], [1]]; // Shape [2, 1]

    let res = arr.fancy_index(&[&idx0, &idx1]).unwrap();
    assert_eq!(res.shape(), &[2, 2]);
    // (0,0)=0, (1,0)=3, (0,1)=1, (1,1)=4
    assert_eq!(res.to_vec(), vec![0, 3, 1, 4]);
}

#[test]
fn test_fancy_indexing_out_of_bounds() {
    let arr = array![1, 2, 3];
    let idx = array![3];

    let res = arr.fancy_index(&[&idx]);
    assert!(res.is_err());
}
