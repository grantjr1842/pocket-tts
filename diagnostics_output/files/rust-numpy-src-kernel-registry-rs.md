# Diagnostics: rust-numpy/src/kernel_registry.rs

**Errors:** 3 | **Warnings:** 0

---

## Errors

### Line 161 (Col 58)
**ERROR[E0412]**: cannot find type `Array` in this scope

```
error[E0412]: cannot find type `Array` in this scope
   --> src/kernel_registry.rs:161:58
    |
161 |                 let in0 = unsafe { &*(input[0] as *const Array<f64>) };
    |                                                          ^^^^^ not found in this scope
    |
```

### Line 162 (Col 58)
**ERROR[E0412]**: cannot find type `Array` in this scope

```
error[E0412]: cannot find type `Array` in this scope
   --> src/kernel_registry.rs:162:58
    |
162 |                 let in1 = unsafe { &*(input[1] as *const Array<f64>) };
    |                                                          ^^^^^ not found in this scope
    |
```

### Line 163 (Col 61)
**ERROR[E0412]**: cannot find type `Array` in this scope

```
error[E0412]: cannot find type `Array` in this scope
   --> src/kernel_registry.rs:163:61
    |
163 |                 let out = unsafe { &mut *(output[0] as *mut Array<f64>) };
    |                                                             ^^^^^ not found in this scope
    |
```
