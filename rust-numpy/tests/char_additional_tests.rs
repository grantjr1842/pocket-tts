// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use numpy::char::*;
use numpy::{array, Array};

#[test]
fn test_ljust() {
    let arr: Array<String> = array!["hello", "world"].map(|s| s.to_string());
    let result = ljust(&arr, 10, Some('-')).unwrap();

    assert_eq!(result.get(0).unwrap(), "hello-----");
    assert_eq!(result.get(1).unwrap(), "world-----");
}

#[test]
fn test_rjust() {
    let arr: Array<String> = array!["hello", "world"].map(|s| s.to_string());
    let result = rjust(&arr, 10, Some('-')).unwrap();

    assert_eq!(result.get(0).unwrap(), "-----hello");
    assert_eq!(result.get(1).unwrap(), "-----world");
}

#[test]
fn test_swapcase() {
    let arr: Array<String> = array!["HeLLo", "WoRLD"].map(|s| s.to_string());
    let result = swapcase(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), "hEllO");
    assert_eq!(result.get(1).unwrap(), "wOrld");
}

#[test]
fn test_title() {
    let arr: Array<String> = array!["hello world", "NUMPY RUST"].map(|s| s.to_string());
    let result = title(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), "Hello World");
    assert_eq!(result.get(1).unwrap(), "Numpy Rust");
}

#[test]
fn test_rsplit() {
    let arr: Array<String> = array!["a:b:c", "x:y:z"].map(|s| s.to_string());
    let result = rsplit(&arr, ":", Some(1)).unwrap();

    assert_eq!(result.get(0).unwrap(), "a:b c");
    assert_eq!(result.get(1).unwrap(), "x:y z");
}

#[test]
fn test_partition() {
    let arr: Array<String> = array!["hello-world", "test"].map(|s| s.to_string());
    let result = partition(&arr, "-").unwrap();

    let expected_0: Vec<String> = vec!["hello".to_string(), "-".to_string(), "world".to_string()];
    let expected_1: Vec<String> = vec!["test".to_string(), "".to_string(), "".to_string()];

    assert_eq!(result.get(0).unwrap(), &expected_0);
    assert_eq!(result.get(1).unwrap(), &expected_1);
}

#[test]
fn test_rpartition() {
    let arr: Array<String> = array!["a-b-c", "test"].map(|s| s.to_string());
    let result = rpartition(&arr, "-").unwrap();

    let expected_0: Vec<String> = vec!["a-b".to_string(), "-".to_string(), "c".to_string()];
    let expected_1: Vec<String> = vec!["".to_string(), "".to_string(), "test".to_string()];

    assert_eq!(result.get(0).unwrap(), &expected_0);
    assert_eq!(result.get(1).unwrap(), &expected_1);
}

#[test]
fn test_splitlines() {
    let arr: Array<String> = array!["line1\nline2", "single"].map(|s| s.to_string());
    let result = splitlines(&arr, Some(false)).unwrap();

    assert_eq!(result.get(0).unwrap().len(), 2);
    assert_eq!(result.get(1).unwrap().len(), 1);
}

#[test]
fn test_str_len() {
    let arr: Array<String> = array!["hello", "world", "test"].map(|s| s.to_string());
    let result = str_len(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), &5);
    assert_eq!(result.get(1).unwrap(), &5);
    assert_eq!(result.get(2).unwrap(), &4);
}

#[test]
fn test_equal() {
    let a: Array<String> = array!["hello", "world"].map(|s| s.to_string());
    let b: Array<String> = array!["hello", "rust"].map(|s| s.to_string());
    let result = equal(&a, &b).unwrap();

    assert_eq!(result.get(0).unwrap(), &true);
    assert_eq!(result.get(1).unwrap(), &false);
}

#[test]
fn test_greater() {
    let a: Array<String> = array!["zebra", "apple"].map(|s| s.to_string());
    let b: Array<String> = array!["apple", "zebra"].map(|s| s.to_string());
    let result = greater(&a, &b).unwrap();

    assert_eq!(result.get(0).unwrap(), &true);
    assert_eq!(result.get(1).unwrap(), &false);
}

#[test]
fn test_less() {
    let a: Array<String> = array!["apple", "zebra"].map(|s| s.to_string());
    let b: Array<String> = array!["zebra", "apple"].map(|s| s.to_string());
    let result = less(&a, &b).unwrap();

    assert_eq!(result.get(0).unwrap(), &true);
    assert_eq!(result.get(1).unwrap(), &false);
}

#[test]
fn test_invalid_fillchar() {
    let arr: Array<String> = array!["test"].map(|s| s.to_string());
    let result = ljust(&arr, 10, Some('\n'));
    assert!(result.is_err());
}

#[test]
fn test_just_longer_than_width() {
    let arr: Array<String> = array!["very long string"].map(|s| s.to_string());
    let result = ljust(&arr, 5, None).unwrap();
    assert_eq!(result.get(0).unwrap(), "very long string");
}

// Tests for newly added functions (issue #384)

#[test]
fn test_isdecimal() {
    use numpy::char::isdecimal;
    
    let a = Array::from_vec(vec!["123".to_string(), "abc".to_string(), "12.3".to_string()]);
    let result = isdecimal(&a).unwrap();
    
    assert_eq!(result[0], true);  // "123" is all decimal
    assert_eq!(result[1], false); // "abc" is not decimal
    assert_eq!(result[2], false); // "12.3" has a dot
}

#[test]
fn test_islower() {
    use numpy::char::islower;
    
    let a = Array::from_vec(vec!["hello".to_string(), "Hello".to_string(), "".to_string()]);
    let result = islower(&a).unwrap();
    
    assert_eq!(result[0], true);  // "hello" is all lowercase
    assert_eq!(result[1], false); // "Hello" has uppercase
    assert_eq!(result[2], false); // empty string
}

#[test]
fn test_isupper() {
    use numpy::char::isupper;
    
    let a = Array::from_vec(vec!["HELLO".to_string(), "Hello".to_string(), "".to_string()]);
    let result = isupper(&a).unwrap();
    
    assert_eq!(result[0], true);  // "HELLO" is all uppercase
    assert_eq!(result[1], false); // "Hello" has lowercase
    assert_eq!(result[2], false); // empty string
}

#[test]
fn test_istitle() {
    use numpy::char::istitle;
    
    let a = Array::from_vec(vec!["Hello World".to_string(), "HELLO WORLD".to_string(), "hello world".to_string()]);
    let result = istitle(&a).unwrap();
    
    assert_eq!(result[0], true);  // "Hello World" is titlecased
    assert_eq!(result[1], false); // "HELLO WORLD" is all uppercase
    assert_eq!(result[2], false); // "hello world" is all lowercase
}

#[test]
fn test_translate() {
    use numpy::char::translate;
    
    let a = Array::from_vec(vec!["hello".to_string(), "world".to_string()]);
    let mut table = std::collections::HashMap::new();
    table.insert('h', "H".to_string());
    table.insert('e', "3".to_string());
    table.insert('l', "1".to_string());
    table.insert('o', "0".to_string());
    
    let result = translate(&a, &table).unwrap();
    
    assert_eq!(result[0], "H3110"); // "hello" -> "H3110"
    assert_eq!(result[1], "w0r1d"); // "world" -> "w0r1d" (partial)
}
