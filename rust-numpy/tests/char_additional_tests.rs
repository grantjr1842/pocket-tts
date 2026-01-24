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

#[test]
fn test_isdecimal() {
    let arr: Array<String> = array!["123", "abc", "12.34", "456"].map(|s| s.to_string());
    let result = isdecimal(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), &true);
    assert_eq!(result.get(1).unwrap(), &false);
    assert_eq!(result.get(2).unwrap(), &false); // Contains dot
    assert_eq!(result.get(3).unwrap(), &true);
}

#[test]
fn test_islower() {
    let arr: Array<String> = array!["hello", "WORLD", "test123", ""].map(|s| s.to_string());
    let result = islower(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), &true);
    assert_eq!(result.get(1).unwrap(), &false);
    assert_eq!(result.get(2).unwrap(), &true);
    assert_eq!(result.get(3).unwrap(), &false); // Empty string
}

#[test]
fn test_isupper() {
    let arr: Array<String> = array!["HELLO", "world", "TEST123", ""].map(|s| s.to_string());
    let result = isupper(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), &true);
    assert_eq!(result.get(1).unwrap(), &false);
    assert_eq!(result.get(2).unwrap(), &true);
    assert_eq!(result.get(3).unwrap(), &false); // Empty string
}

#[test]
fn test_istitle() {
    let arr: Array<String> =
        array!["Hello World", "HELLO WORLD", "hello world", "Test"].map(|s| s.to_string());
    let result = istitle(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), &true);
    assert_eq!(result.get(1).unwrap(), &false);
    assert_eq!(result.get(2).unwrap(), &false);
    assert_eq!(result.get(3).unwrap(), &true);
}

#[test]
fn test_translate() {
    let arr: Array<String> = array!["abc", "xyz"].map(|s| s.to_string());
    // Table maps a->b, b->c, etc. (each char maps to the next one in table)
    let table = "abcdefghijklmnopqrstuvwxyz";
    let delete = None;
    let result = translate(&arr, table, delete).unwrap();

    // "abc" becomes "bcd" (a->b, b->c, c->d)
    assert_eq!(result.get(0).unwrap(), "bcd");
    // "xyz" becomes "yzz" (x->y, y->z, z->z as z is last in table)
    assert_eq!(result.get(1).unwrap(), "yzz");
}

#[test]
fn test_translate_with_delete() {
    let arr: Array<String> = array!["hello world", "test"].map(|s| s.to_string());
    let table = "";
    let delete = Some("aeiou ");
    let result = translate(&arr, table, delete).unwrap();

    assert_eq!(result.get(0).unwrap(), "hllwrld");
    assert_eq!(result.get(1).unwrap(), "tst");
}

#[test]
fn test_mod() {
    let format_arr: Array<String> = array!["Hello %s", "Test: %s"].map(|s| s.to_string());
    let values: Array<String> = array!["World", "Value"].map(|s| s.to_string());
    let result = mod_(&format_arr, &values).unwrap();

    assert_eq!(result.get(0).unwrap(), "Hello World");
    assert_eq!(result.get(1).unwrap(), "Test: Value");
}

#[test]
fn test_array() {
    let arr: Array<String> = array!["hello", "world"].map(|s| s.to_string());
    let result = array(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), "hello");
    assert_eq!(result.get(1).unwrap(), "world");
}

#[test]
fn test_asarray() {
    let arr: Array<String> = array!["test", "data"].map(|s| s.to_string());
    let result = asarray(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), "test");
    assert_eq!(result.get(1).unwrap(), "data");
}

#[test]
fn test_chararray() {
    let arr: Array<String> = array!["sample", "text"].map(|s| s.to_string());
    let result = chararray(&arr).unwrap();

    assert_eq!(result.get(0).unwrap(), "sample");
    assert_eq!(result.get(1).unwrap(), "text");
}

#[test]
fn test_compare_chararrays() {
    let a: Array<String> = array!["apple", "zebra"].map(|s| s.to_string());
    let b: Array<String> = array!["banana", "apple"].map(|s| s.to_string());
    let result = compare_chararrays(&a, &b, "<", None).unwrap();

    assert_eq!(result.get(0).unwrap(), &true);
    assert_eq!(result.get(1).unwrap(), &false);
}

#[test]
fn test_compare_chararrays_with_rstrip() {
    let a: Array<String> = array!["test  ", "hello"].map(|s| s.to_string());
    let b: Array<String> = array!["test", "hello  "].map(|s| s.to_string());
    let result = compare_chararrays(&a, &b, "==", Some(true)).unwrap();

    assert_eq!(result.get(0).unwrap(), &true);
    assert_eq!(result.get(1).unwrap(), &true);
}

#[test]
fn test_decode() {
    let arr: Array<String> = array!["hello", "world"].map(|s| s.to_string());
    let result = decode(&arr, Some("utf-8")).unwrap();

    assert_eq!(result.get(0).unwrap(), "hello");
    assert_eq!(result.get(1).unwrap(), "world");
}

#[test]
fn test_encode() {
    let arr: Array<String> = array!["hi", "AB"].map(|s| s.to_string());
    let result = encode(&arr, Some("utf-8")).unwrap();

    // "hi" in UTF-8 hex: 68 69
    // "AB" in UTF-8 hex: 41 42
    assert_eq!(result.get(0).unwrap(), "6869");
    assert_eq!(result.get(1).unwrap(), "4142");
}
