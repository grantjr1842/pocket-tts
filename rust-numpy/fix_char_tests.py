#!/usr/bin/env python3
import re
import sys

with open('tests/char_additional_tests.rs', 'r') as f:
    content = f.read()

# Replace all array!["..."].map(|s| s.to_string()) patterns
patterns = [
    (r'array!\["hello", "world"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["hello".to_string(), "world".to_string()])'),
    (r'array!\["HeLLo", "WoRLD"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["HeLLo".to_string(), "WoRLD".to_string()])'),
    (r'array!\["hello world", "NUMPY RUST"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["hello world".to_string(), "NUMPY RUST".to_string()])'),
    (r'array!\["a:b:c", "x:y:z"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["a:b:c".to_string(), "x:y:z".to_string()])'),
    (r'array!\["hello-world", "test"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["hello-world".to_string(), "test".to_string()])'),
    (r'array!\["a-b-c", "test"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["a-b-c".to_string(), "test".to_string()])'),
    (r'array!\["line1\\nline2", "single"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["line1\\nline2".to_string(), "single".to_string()])'),
    (r'array!\["hello", "world", "test"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["hello".to_string(), "world".to_string(), "test".to_string()])'),
    (r'array!\["hello", "rust"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["hello".to_string(), "rust".to_string()])'),
    (r'array!\["zebra", "apple"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["zebra".to_string(), "apple".to_string()])'),
    (r'array!\["apple", "zebra"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["apple".to_string(), "zebra".to_string()])'),
    (r'array!\["test"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["test".to_string()])'),
    (r'array!\["very long string"\]\.map\(\|s\| s\.to_string\(\)\)',
     'Array::from_vec(vec!["very long string".to_string()])'),
]

for pattern, replacement in patterns:
    content = re.sub(pattern, replacement, content)

with open('tests/char_additional_tests.rs', 'w') as f:
    f.write(content)

print("Fixed char_additional_tests.rs")
