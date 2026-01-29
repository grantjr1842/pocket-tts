# Diagnostics Prioritized by Error Count

Files sorted by errors (descending), then warnings.

---

## 1. [rust-numpy/src/random/tests.rs](files/rust-numpy-src-random-tests-rs.md)
**Errors:** 40 | **Warnings:** 1

**Top Errors:**

- **Line 9:** `E0425` - cannot find function `default_rng` in this scope
- **Line 23:** `E0425` - cannot find function `default_rng_with_seed` in this scope
- **Line 24:** `E0425` - cannot find function `default_rng_with_seed` in this scope
- **Line 40:** `E0425` - cannot find function `random` in this scope
- **Line 45:** `E0425` - cannot find function `randint` in this scope

## 2. [rust-numpy/examples/generator_examples.rs](files/rust-numpy-examples-generator-examples-rs.md)
**Errors:** 38 | **Warnings:** 0

**Top Errors:**

- **Line 21:** `E0533` - expected value, found struct variant `Dtype::Float64`
- **Line 50:** `E0533` - expected value, found struct variant `Dtype::Float64`
- **Line 51:** `E0533` - expected value, found struct variant `Dtype::Float64`
- **Line 60:** `E0533` - expected value, found struct variant `Dtype::Float64`
- **Line 127:** `E0614` - type `f64` cannot be dereferenced

## 3. [rust-numpy/tests/char_additional_tests.rs](files/rust-numpy-tests-char-additional-tests-rs.md)
**Errors:** 34 | **Warnings:** 0

**Top Errors:**

- **Line 13:** `E0599` - `rust_numpy::Array<&str>` is not an iterator
- **Line 13:** `E0282` - type annotations needed
- **Line 22:** `E0599` - `rust_numpy::Array<&str>` is not an iterator
- **Line 22:** `E0282` - type annotations needed
- **Line 31:** `E0599` - `rust_numpy::Array<&str>` is not an iterator

## 4. [rust-numpy/src/typing/typing_tests.rs](files/rust-numpy-src-typing-typing-tests-rs.md)
**Errors:** 32 | **Warnings:** 3

**Top Errors:**

- **Line 15:** `E0412` - cannot find type `NDArray` in this scope
- **Line 23:** `E0412` - cannot find type `Int8Array` in this scope
- **Line 24:** `E0412` - cannot find type `Int16Array` in this scope
- **Line 25:** `E0412` - cannot find type `Int32Array` in this scope
- **Line 26:** `E0412` - cannot find type `Int64Array` in this scope

## 5. [rust-numpy/examples/random_examples.rs](files/rust-numpy-examples-random-examples-rs.md)
**Errors:** 21 | **Warnings:** 9

**Top Errors:**

- **Line 7:** `E0432` - unresolved import `rust_numpy::prelude`
- **Line 214:** `E0308` - mismatched types
- **Line 215:** `E0308` - mismatched types
- **Line 216:** `E0308` - mismatched types
- **Line 217:** `E0308` - mismatched types

## 6. [rust-numpy/src/typing/tests.rs](files/rust-numpy-src-typing-tests-rs.md)
**Errors:** 16 | **Warnings:** 1

**Top Errors:**

- **Line 10:** `E0412` - cannot find type `NDArray` in this scope
- **Line 18:** `E0412` - cannot find type `Int32Array` in this scope
- **Line 21:** `E0412` - cannot find type `Float64Array` in this scope
- **Line 24:** `E0412` - cannot find type `BoolArray` in this scope
- **Line 124:** `E0412` - cannot find type `Complex64Array` in this scope

## 7. [rust-numpy/tests/reduction_tests.rs](files/rust-numpy-tests-reduction-tests-rs.md)
**Errors:** 6 | **Warnings:** 0

**Top Errors:**

- **Line 84:** `E0689` - can't call method `abs` on ambiguous numeric type `{float}`
- **Line 92:** `E0689` - can't call method `abs` on ambiguous numeric type `{float}`
- **Line 93:** `E0689` - can't call method `abs` on ambiguous numeric type `{float}`
- **Line 94:** `E0689` - can't call method `abs` on ambiguous numeric type `{float}`
- **Line 102:** `E0689` - can't call method `abs` on ambiguous numeric type `{float}`

## 8. [rust-numpy/tests/random_tests.rs](files/rust-numpy-tests-random-tests-rs.md)
**Errors:** 4 | **Warnings:** 0

**Top Errors:**

- **Line 26:** `E0283` - type annotations needed
- **Line 29:** `E0283` - type annotations needed
- **Line 200:** `E0425` - cannot find function `shuffle` in this scope
- **Line 220:** `E0425` - cannot find function `shuffle` in this scope

## 9. [rust-numpy/src/kernel_registry.rs](files/rust-numpy-src-kernel-registry-rs.md)
**Errors:** 3 | **Warnings:** 0

**Top Errors:**

- **Line 161:** `E0412` - cannot find type `Array` in this scope
- **Line 162:** `E0412` - cannot find type `Array` in this scope
- **Line 163:** `E0412` - cannot find type `Array` in this scope

## 10. [rust-numpy/src/advanced_reductions.rs](files/rust-numpy-src-advanced-reductions-rs.md)
**Errors:** 2 | **Warnings:** 0

**Top Errors:**

- **Line 478:** `E0433` - failed to resolve: use of undeclared type `Complex32`
- **Line 479:** `E0433` - failed to resolve: use of undeclared type `Complex32`

## 11. [rust-numpy/src/simd_ops.rs](files/rust-numpy-src-simd-ops-rs.md)
**Errors:** 2 | **Warnings:** 0

**Top Errors:**

- **Line 367:** `E0433` - failed to resolve: could not find `simd_intrinsics` in the crate root
- **Line 375:** `E0433` - failed to resolve: use of undeclared type `SimdChunkSize`

## 12. [problems-1.json](files/problems-1.json.md)
**Errors:** 1 | **Warnings:** 0

**Top Errors:**

- **Line 1:** `unknown` - Expected a JSON object, array or literal.

## 13. [rust-numpy/examples/bitgenerator_examples.rs](files/rust-numpy-examples-bitgenerator-examples-rs.md)
**Errors:** 1 | **Warnings:** 0

**Top Errors:**

- **Line 1:** `E0601` - `main` function not found in crate `bitgenerator_examples`

## 14. [rust-numpy/src/char_tests.rs](files/rust-numpy-src-char-tests-rs.md)
**Errors:** 1 | **Warnings:** 0

**Top Errors:**

- **Line 108:** `E0425` - cannot find function `strip_chars` in this scope

## 15. [rust-numpy/tests/comprehensive_tests.rs](files/rust-numpy-tests-comprehensive-tests-rs.md)
**Errors:** 1 | **Warnings:** 0

**Top Errors:**

- **Line 129:** `E0277` - the trait bound `{integer}: std::convert::From<f64>` is not satisfied

## 16. [rust-numpy/tests/kernel_registry_tests.rs](files/rust-numpy-tests-kernel-registry-tests-rs.md)
**Errors:** 1 | **Warnings:** 0

**Top Errors:**

- **Line 307:** `Click for full compiler diagnostic` - error: unexpected closing delimiter: `}`

## 17. [rust-numpy/tests/polynomial_verification.rs](files/rust-numpy-tests-polynomial-verification-rs.md)
**Errors:** 0 | **Warnings:** 3


## 18. [rust-numpy/tests/test_broadcasting_edge_cases.rs](files/rust-numpy-tests-test-broadcasting-edge-cases-rs.md)
**Errors:** 0 | **Warnings:** 2


## 19. [rust-numpy/examples/typing_examples.rs](files/rust-numpy-examples-typing-examples-rs.md)
**Errors:** 0 | **Warnings:** 1


## 20. [rust-numpy/src/dynamic_kernel_registry.rs](files/rust-numpy-src-dynamic-kernel-registry-rs.md)
**Errors:** 0 | **Warnings:** 1


## 21. [rust-numpy/src/typing/dtype_getter.rs](files/rust-numpy-src-typing-dtype-getter-rs.md)
**Errors:** 0 | **Warnings:** 1


## 22. [rust-numpy/tests/broadcasting_edge_cases.rs](files/rust-numpy-tests-broadcasting-edge-cases-rs.md)
**Errors:** 0 | **Warnings:** 1


## 23. [rust-numpy/tests/ndarray_conversion_tests.rs](files/rust-numpy-tests-ndarray-conversion-tests-rs.md)
**Errors:** 0 | **Warnings:** 1


## 24. [rust-numpy/tests/property_tests.rs](files/rust-numpy-tests-property-tests-rs.md)
**Errors:** 0 | **Warnings:** 1


## 25. [rust-numpy/tests/random_architecture_tests.rs](files/rust-numpy-tests-random-architecture-tests-rs.md)
**Errors:** 0 | **Warnings:** 1


## 26. [rust-numpy/tests/type_promotion_rules_tests.rs](files/rust-numpy-tests-type-promotion-rules-tests-rs.md)
**Errors:** 0 | **Warnings:** 1

