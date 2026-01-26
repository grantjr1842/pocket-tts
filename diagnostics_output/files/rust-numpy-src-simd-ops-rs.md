# Diagnostics: rust-numpy/src/simd_ops.rs

**Errors:** 2 | **Warnings:** 0

---

## Errors

### Line 367 (Col 31)
**ERROR[E0433]**: failed to resolve: could not find `simd_intrinsics` in the crate root

```
error[E0433]: failed to resolve: could not find `simd_intrinsics` in the crate root
   --> src/simd_ops.rs:367:31
    |
367 |         let features = crate::simd_intrinsics::CpuFeatures::detect();
    |                               ^^^^^^^^^^^^^^^ could not find `simd_intrinsics` in the crate root
    |
```

### Line 375 (Col 26)
**ERROR[E0433]**: failed to resolve: use of undeclared type `SimdChunkSize`

```
error[E0433]: failed to resolve: use of undeclared type `SimdChunkSize`
   --> src/simd_ops.rs:375:26
    |
375 |         let chunk_size = SimdChunkSize::new();
    |                          ^^^^^^^^^^^^^ use of undeclared type `SimdChunkSize`
```
