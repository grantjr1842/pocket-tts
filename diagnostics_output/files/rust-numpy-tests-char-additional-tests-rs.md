# Diagnostics: rust-numpy/tests/char_additional_tests.rs

**Errors:** 34 | **Warnings:** 0

---

## Errors

### Line 13 (Col 55)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
  --> tests/char_additional_tests.rs:13:55
   |
13 |     let arr: Array<String> = array!["hello", "world"].map(|s| s.to_string());
   |                                                       ^^^ `rust_numpy::Array<&str>` is not an iterator
   |
  ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `rust_numpy::Array<&str>: Iterator`
           which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 13 (Col 60)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
  --> tests/char_additional_tests.rs:13:60
   |
13 |     let arr: Array<String> = array!["hello", "world"].map(|s| s.to_string());
   |                                                            ^  - type must be known at this point
   |
```

### Line 22 (Col 55)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
  --> tests/char_additional_tests.rs:22:55
   |
22 |     let arr: Array<String> = array!["hello", "world"].map(|s| s.to_string());
   |                                                       ^^^ `rust_numpy::Array<&str>` is not an iterator
   |
  ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `rust_numpy::Array<&str>: Iterator`
           which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 22 (Col 60)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
  --> tests/char_additional_tests.rs:22:60
   |
22 |     let arr: Array<String> = array!["hello", "world"].map(|s| s.to_string());
   |                                                            ^  - type must be known at this point
   |
```

### Line 31 (Col 55)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
  --> tests/char_additional_tests.rs:31:55
   |
31 |     let arr: Array<String> = array!["HeLLo", "WoRLD"].map(|s| s.to_string());
   |                                                       ^^^ `rust_numpy::Array<&str>` is not an iterator
   |
  ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `rust_numpy::Array<&str>: Iterator`
           which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 31 (Col 60)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
  --> tests/char_additional_tests.rs:31:60
   |
31 |     let arr: Array<String> = array!["HeLLo", "WoRLD"].map(|s| s.to_string());
   |                                                            ^  - type must be known at this point
   |
```

### Line 40 (Col 66)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
  --> tests/char_additional_tests.rs:40:66
   |
40 |     let arr: Array<String> = array!["hello world", "NUMPY RUST"].map(|s| s.to_string());
   |                                                                  ^^^ `rust_numpy::Array<&str>` is not an iterator
   |
  ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `rust_numpy::Array<&str>: Iterator`
           which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 40 (Col 71)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
  --> tests/char_additional_tests.rs:40:71
   |
40 |     let arr: Array<String> = array!["hello world", "NUMPY RUST"].map(|s| s.to_string());
   |                                                                       ^  - type must be known at this point
   |
```

### Line 49 (Col 55)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
  --> tests/char_additional_tests.rs:49:55
   |
49 |     let arr: Array<String> = array!["a:b:c", "x:y:z"].map(|s| s.to_string());
   |                                                       ^^^ `rust_numpy::Array<&str>` is not an iterator
   |
  ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `rust_numpy::Array<&str>: Iterator`
           which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 49 (Col 60)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
  --> tests/char_additional_tests.rs:49:60
   |
49 |     let arr: Array<String> = array!["a:b:c", "x:y:z"].map(|s| s.to_string());
   |                                                            ^  - type must be known at this point
   |
```

### Line 58 (Col 60)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
  --> tests/char_additional_tests.rs:58:60
   |
58 |     let arr: Array<String> = array!["hello-world", "test"].map(|s| s.to_string());
   |                                                            ^^^ `rust_numpy::Array<&str>` is not an iterator
   |
  ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `rust_numpy::Array<&str>: Iterator`
           which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 58 (Col 65)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
  --> tests/char_additional_tests.rs:58:65
   |
58 |     let arr: Array<String> = array!["hello-world", "test"].map(|s| s.to_string());
   |                                                                 ^  - type must be known at this point
   |
```

### Line 70 (Col 54)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
  --> tests/char_additional_tests.rs:70:54
   |
70 |     let arr: Array<String> = array!["a-b-c", "test"].map(|s| s.to_string());
   |                                                      ^^^ `rust_numpy::Array<&str>` is not an iterator
   |
  ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `rust_numpy::Array<&str>: Iterator`
           which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 70 (Col 59)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
  --> tests/char_additional_tests.rs:70:59
   |
70 |     let arr: Array<String> = array!["a-b-c", "test"].map(|s| s.to_string());
   |                                                           ^  - type must be known at this point
   |
```

### Line 82 (Col 63)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
  --> tests/char_additional_tests.rs:82:63
   |
82 |     let arr: Array<String> = array!["line1\nline2", "single"].map(|s| s.to_string());
   |                                                               ^^^ `rust_numpy::Array<&str>` is not an iterator
   |
  ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `rust_numpy::Array<&str>: Iterator`
           which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 82 (Col 68)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
  --> tests/char_additional_tests.rs:82:68
   |
82 |     let arr: Array<String> = array!["line1\nline2", "single"].map(|s| s.to_string());
   |                                                                    ^  - type must be known at this point
   |
```

### Line 91 (Col 63)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
  --> tests/char_additional_tests.rs:91:63
   |
91 |     let arr: Array<String> = array!["hello", "world", "test"].map(|s| s.to_string());
   |                                                               ^^^ `rust_numpy::Array<&str>` is not an iterator
   |
  ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
   |
22 | pub struct Array<T> {
   | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
   |
   = note: the following trait bounds were not satisfied:
           `rust_numpy::Array<&str>: Iterator`
           which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 91 (Col 68)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
  --> tests/char_additional_tests.rs:91:68
   |
91 |     let arr: Array<String> = array!["hello", "world", "test"].map(|s| s.to_string());
   |                                                                    ^  - type must be known at this point
   |
```

### Line 101 (Col 53)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
   --> tests/char_additional_tests.rs:101:53
    |
101 |     let a: Array<String> = array!["hello", "world"].map(|s| s.to_string());
    |                                                     ^^^ `rust_numpy::Array<&str>` is not an iterator
    |
   ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
    |
    = note: the following trait bounds were not satisfied:
            `rust_numpy::Array<&str>: Iterator`
            which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 101 (Col 58)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> tests/char_additional_tests.rs:101:58
    |
101 |     let a: Array<String> = array!["hello", "world"].map(|s| s.to_string());
    |                                                          ^  - type must be known at this point
    |
```

### Line 102 (Col 52)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
   --> tests/char_additional_tests.rs:102:52
    |
102 |     let b: Array<String> = array!["hello", "rust"].map(|s| s.to_string());
    |                                                    ^^^ `rust_numpy::Array<&str>` is not an iterator
    |
   ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
    |
    = note: the following trait bounds were not satisfied:
            `rust_numpy::Array<&str>: Iterator`
            which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 102 (Col 57)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> tests/char_additional_tests.rs:102:57
    |
102 |     let b: Array<String> = array!["hello", "rust"].map(|s| s.to_string());
    |                                                         ^  - type must be known at this point
    |
```

### Line 111 (Col 53)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
   --> tests/char_additional_tests.rs:111:53
    |
111 |     let a: Array<String> = array!["zebra", "apple"].map(|s| s.to_string());
    |                                                     ^^^ `rust_numpy::Array<&str>` is not an iterator
    |
   ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
    |
    = note: the following trait bounds were not satisfied:
            `rust_numpy::Array<&str>: Iterator`
            which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 111 (Col 58)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> tests/char_additional_tests.rs:111:58
    |
111 |     let a: Array<String> = array!["zebra", "apple"].map(|s| s.to_string());
    |                                                          ^  - type must be known at this point
    |
```

### Line 112 (Col 53)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
   --> tests/char_additional_tests.rs:112:53
    |
112 |     let b: Array<String> = array!["apple", "zebra"].map(|s| s.to_string());
    |                                                     ^^^ `rust_numpy::Array<&str>` is not an iterator
    |
   ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
    |
    = note: the following trait bounds were not satisfied:
            `rust_numpy::Array<&str>: Iterator`
            which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 112 (Col 58)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> tests/char_additional_tests.rs:112:58
    |
112 |     let b: Array<String> = array!["apple", "zebra"].map(|s| s.to_string());
    |                                                          ^  - type must be known at this point
    |
```

### Line 121 (Col 53)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
   --> tests/char_additional_tests.rs:121:53
    |
121 |     let a: Array<String> = array!["apple", "zebra"].map(|s| s.to_string());
    |                                                     ^^^ `rust_numpy::Array<&str>` is not an iterator
    |
   ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
    |
    = note: the following trait bounds were not satisfied:
            `rust_numpy::Array<&str>: Iterator`
            which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 121 (Col 58)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> tests/char_additional_tests.rs:121:58
    |
121 |     let a: Array<String> = array!["apple", "zebra"].map(|s| s.to_string());
    |                                                          ^  - type must be known at this point
    |
```

### Line 122 (Col 53)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
   --> tests/char_additional_tests.rs:122:53
    |
122 |     let b: Array<String> = array!["zebra", "apple"].map(|s| s.to_string());
    |                                                     ^^^ `rust_numpy::Array<&str>` is not an iterator
    |
   ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
    |
    = note: the following trait bounds were not satisfied:
            `rust_numpy::Array<&str>: Iterator`
            which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 122 (Col 58)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> tests/char_additional_tests.rs:122:58
    |
122 |     let b: Array<String> = array!["zebra", "apple"].map(|s| s.to_string());
    |                                                          ^  - type must be known at this point
    |
```

### Line 131 (Col 45)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
   --> tests/char_additional_tests.rs:131:45
    |
131 |     let arr: Array<String> = array!["test"].map(|s| s.to_string());
    |                                             ^^^ `rust_numpy::Array<&str>` is not an iterator
    |
   ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
    |
    = note: the following trait bounds were not satisfied:
            `rust_numpy::Array<&str>: Iterator`
            which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 131 (Col 50)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> tests/char_additional_tests.rs:131:50
    |
131 |     let arr: Array<String> = array!["test"].map(|s| s.to_string());
    |                                                  ^  - type must be known at this point
    |
```

### Line 138 (Col 57)
**ERROR[E0599]**: `rust_numpy::Array<&str>` is not an iterator

```
error[E0599]: `rust_numpy::Array<&str>` is not an iterator
   --> tests/char_additional_tests.rs:138:57
    |
138 |     let arr: Array<String> = array!["very long string"].map(|s| s.to_string());
    |                                                         ^^^ `rust_numpy::Array<&str>` is not an iterator
    |
   ::: /home/grant/github/pocket-tts/rust-numpy/src/array.rs:22:1
    |
 22 | pub struct Array<T> {
    | ------------------- doesn't satisfy `rust_numpy::Array<&str>: Iterator`
    |
    = note: the following trait bounds were not satisfied:
            `rust_numpy::Array<&str>: Iterator`
            which is required by `&mut rust_numpy::Array<&str>: Iterator`
```

### Line 138 (Col 62)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> tests/char_additional_tests.rs:138:62
    |
138 |     let arr: Array<String> = array!["very long string"].map(|s| s.to_string());
    |                                                              ^  - type must be known at this point
    |
```
