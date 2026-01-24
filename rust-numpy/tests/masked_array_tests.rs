use numpy::*;

#[test]
fn test_masked_array_basic() {
    let data = array![1.0, 2.0, 3.0, 4.0];
    let mask = array![false, true, false, true];

    let ma = MaskedArray::new(data.clone(), mask.clone()).unwrap();

    assert_eq!(ma.shape(), &[4]);
    assert_eq!(ma.size(), 4);
    assert_eq!(ma.mask().data(), &[false, true, false, true]);
}

#[test]
fn test_masked_array_filled() {
    let data = array![1.0, 2.0, 3.0, 4.0];
    let mask = array![false, true, false, true];

    let mut ma = MaskedArray::new(data, mask).unwrap();
    ma.set_fill_value(99.0);

    let filled = ma.filled();
    assert_eq!(filled.data(), &[1.0, 99.0, 3.0, 99.0]);
}

#[test]
fn test_masked_array_sum() {
    let data = array![1.0, 2.0, 3.0, 4.0];
    let mask = array![false, true, false, true];

    let ma = MaskedArray::new(data, mask).unwrap();

    // 1.0 + 3.0 = 4.0
    let total = ma.sum().unwrap();
    assert_eq!(total, 4.0);
}

#[test]
fn test_masked_array_binary_op() {
    let data1 = array![1.0, 2.0, 3.0, 4.0];
    let mask1 = array![false, true, false, false];
    let ma1 = MaskedArray::new(data1, mask1).unwrap();

    let data2 = array![10.0, 20.0, 30.0, 40.0];
    let mask2 = array![false, false, true, false];
    let ma2 = MaskedArray::new(data2, mask2).unwrap();

    // ma1 + ma2
    // Result mask should be mask1 | mask2 = [false, true, true, false]
    // Result data (at unmasked) should be [11.0, _, _, 44.0]

    let res = ma1.binary_op(&ma2, |a, b, w, c| a.add(b, w, c)).unwrap();

    assert_eq!(res.mask().data(), &[false, true, true, false]);
    assert_eq!(res.data().get_multi(&[0]).unwrap(), 11.0);
    assert_eq!(res.data().get_multi(&[3]).unwrap(), 44.0);
}

#[test]
fn test_masked_array_mean() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let mask = array![false, true, false, true, false];

    let ma = MaskedArray::new(data, mask).unwrap();

    // Mean of [1, 3, 5] = 9/3 = 3.0
    let mean = ma.mean().unwrap();
    assert_eq!(mean, 3.0);
}

#[test]
fn test_masked_array_median() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mask = array![false, true, false, true, false, true, false];

    let ma = MaskedArray::new(data, mask).unwrap();

    // Unmasked values: [1, 3, 5, 7], median = (3 + 5) / 2 = 4.0
    let median = ma.median().unwrap();
    assert_eq!(median, 4.0);
}

#[test]
fn test_masked_array_median_odd() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let mask = array![false, true, false, true, false];

    let ma = MaskedArray::new(data, mask).unwrap();

    // Unmasked values: [1, 3, 5], median = 3.0
    let median = ma.median().unwrap();
    assert_eq!(median, 3.0);
}

#[test]
fn test_masked_array_var() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let mask = array![false, true, false, true, false];

    let ma = MaskedArray::new(data, mask).unwrap();

    // Unmasked values: [1, 3, 5], mean = 3.0
    // variance = [(1-3)^2 + (3-3)^2 + (5-3)^2] / 3 = (4 + 0 + 4) / 3 = 8/3
    let var = ma.var(Some(0)).unwrap();
    assert!((var - 8.0_f64 / 3.0).abs() < 1e-6);
}

#[test]
fn test_masked_array_std() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let mask = array![false, true, false, true, false];

    let ma = MaskedArray::new(data, mask).unwrap();

    // std = sqrt(8/3) ≈ 1.63299
    let std = ma.std(Some(0)).unwrap();
    assert!((std - 1.63299_f64).abs() < 1e-4);
}

#[test]
fn test_masked_array_compress() {
    let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let mask = array![false, true, false, true, false];
    let condition = array![true, false, true, false, true];

    let ma = MaskedArray::new(data, mask).unwrap();
    let compressed = ma.compress(&condition, None).unwrap();

    // Unmasked AND true in condition: indices 0, 2, 4
    // Values: 1, 3, 5
    assert_eq!(compressed.shape(), &[3]);

    let comp_data = compressed.data().data();
    let mut comp_vec: Vec<_> = comp_data.to_vec();
    comp_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(comp_vec[0], 1.0);
    assert_eq!(comp_vec[1], 3.0);
    assert_eq!(comp_vec[2], 5.0);
}
