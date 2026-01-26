//! Test that namespace exports work correctly
//! 
use rust_numpy::*;

fn test_exports() {
    // Test basic mathematical functions
    let a = array![1, 2, 3];
    let b = array![4, 5, 6];
    
    // These should now be accessible via numpy namespace
    let result = add(&a, &b);
    assert_eq!(result[[0, 0], [5, 7]);
    
    let result = multiply(&a, &b);
    assert_eq!(result[[0, 0], [4, 10]]);
    
    // Test array creation functions
    let zeros = zeros(&[3, 2]);
    assert_eq!(zeros.shape(), vec![3, 2]);
    
    let ones = ones(&[2, 3]);
    assert_eq!(ones.shape(), vec![2, 3]);
}

#[test]
fn main() {
    test_exports();
    println!("✅ All namespace exports work correctly!");
}