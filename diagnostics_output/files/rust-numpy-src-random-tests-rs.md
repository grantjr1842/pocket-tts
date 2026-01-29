# Diagnostics: rust-numpy/src/random/tests.rs

**Errors:** 40 | **Warnings:** 1

---

## Errors

### Line 9 (Col 23)
**ERROR[E0425]**: cannot find function `default_rng` in this scope

```
error[E0425]: cannot find function `default_rng` in this scope
  |
9 |         let mut rng = default_rng();
  |                       ^^^^^^^^^^^ not found in this scope
  |
```

### Line 23 (Col 24)
**ERROR[E0425]**: cannot find function `default_rng_with_seed` in this scope

```
error[E0425]: cannot find function `default_rng_with_seed` in this scope
  --> src/random/tests.rs:23:24
   |
23 |         let mut rng1 = default_rng_with_seed(seed);
   |                        ^^^^^^^^^^^^^^^^^^^^^ not found in this scope
   |
```

### Line 24 (Col 24)
**ERROR[E0425]**: cannot find function `default_rng_with_seed` in this scope

```
error[E0425]: cannot find function `default_rng_with_seed` in this scope
  --> src/random/tests.rs:24:24
   |
24 |         let mut rng2 = default_rng_with_seed(seed);
   |                        ^^^^^^^^^^^^^^^^^^^^^ not found in this scope
   |
```

### Line 40 (Col 19)
**ERROR[E0425]**: cannot find function `random` in this scope

```
error[E0425]: cannot find function `random` in this scope
  --> src/random/tests.rs:40:19
   |
40 |         let arr = random::<f64>(&[2, 3], Dtype::Float64).unwrap();
   |                   ^^^^^^ not found in this scope
   |
```

### Line 45 (Col 23)
**ERROR[E0425]**: cannot find function `randint` in this scope

```
error[E0425]: cannot find function `randint` in this scope
  --> src/random/tests.rs:45:23
   |
45 |         let int_arr = randint::<i32>(0, 10, &[2, 2]).unwrap();
   |                       ^^^^^^^ not found in this scope
   |
```

### Line 50 (Col 27)
**ERROR[E0425]**: cannot find function `uniform` in this scope

```
error[E0425]: cannot find function `uniform` in this scope
  --> src/random/tests.rs:50:27
   |
50 |         let uniform_arr = uniform::<f64>(0.0, 1.0, &[3, 2]).unwrap();
   |                           ^^^^^^^ not found in this scope
   |
```

### Line 55 (Col 26)
**ERROR[E0425]**: cannot find function `normal` in this scope

```
error[E0425]: cannot find function `normal` in this scope
  --> src/random/tests.rs:55:26
   |
55 |         let normal_arr = normal::<f64>(0.0, 1.0, &[2, 2]).unwrap();
   |                          ^^^^^^ not found in this scope
   |
```

### Line 60 (Col 30)
**ERROR[E0425]**: cannot find function `standard_normal` in this scope

```
error[E0425]: cannot find function `standard_normal` in this scope
  --> src/random/tests.rs:60:30
   |
60 |         let std_normal_arr = standard_normal::<f64>(&[2, 3]).unwrap();
   |                              ^^^^^^^^^^^^^^^ not found in this scope
   |
```

### Line 70 (Col 23)
**ERROR[E0425]**: cannot find function `binomial` in this scope

```
error[E0425]: cannot find function `binomial` in this scope
  --> src/random/tests.rs:70:23
   |
70 |         let bin_arr = binomial::<f64>(10, 0.5, &[2, 2]).unwrap();
   |                       ^^^^^^^^ not found in this scope
   |
```

### Line 74 (Col 24)
**ERROR[E0425]**: cannot find function `poisson` in this scope

```
error[E0425]: cannot find function `poisson` in this scope
  --> src/random/tests.rs:74:24
   |
74 |         let pois_arr = poisson::<f64>(5.0, &[2, 2]).unwrap();
   |                        ^^^^^^^ not found in this scope
   |
```

### Line 78 (Col 23)
**ERROR[E0425]**: cannot find function `exponential` in this scope

```
error[E0425]: cannot find function `exponential` in this scope
  --> src/random/tests.rs:78:23
   |
78 |         let exp_arr = exponential::<f64>(1.0, &[2, 2]).unwrap();
   |                       ^^^^^^^^^^^ not found in this scope
   |
```

### Line 82 (Col 25)
**ERROR[E0425]**: cannot find function `gamma` in this scope

```
error[E0425]: cannot find function `gamma` in this scope
  --> src/random/tests.rs:82:25
   |
82 |         let gamma_arr = gamma::<f64>(2.0, 2.0, &[2, 2]).unwrap();
   |                         ^^^^^ not found in this scope
   |
```

### Line 86 (Col 24)
**ERROR[E0425]**: cannot find function `beta` in this scope

```
error[E0425]: cannot find function `beta` in this scope
  --> src/random/tests.rs:86:24
   |
86 |         let beta_arr = beta::<f64>(2.0, 2.0, &[2, 2]).unwrap();
   |                        ^^^^ not found in this scope
   |
```

### Line 95 (Col 23)
**ERROR[E0425]**: cannot find function `legacy_rng` in this scope

```
error[E0425]: cannot find function `legacy_rng` in this scope
  --> src/random/tests.rs:95:23
   |
95 |         let _legacy = legacy_rng();
   |                       ^^^^^^^^^^ not found in this scope
   |
```

### Line 98 (Col 26)
**ERROR[E0425]**: cannot find function `legacy_random` in this scope

```
error[E0425]: cannot find function `legacy_random` in this scope
  --> src/random/tests.rs:98:26
   |
98 |         let legacy_arr = legacy_random::<f64>(&[2, 2], Dtype::Float64).unwrap();
   |                          ^^^^^^^^^^^^^ not found in this scope
   |
```

### Line 102 (Col 30)
**ERROR[E0425]**: cannot find function `legacy_randint` in this scope

```
error[E0425]: cannot find function `legacy_randint` in this scope
   --> src/random/tests.rs:102:30
    |
102 |         let legacy_int_arr = legacy_randint::<i32>(0, 10, &[2, 2]).unwrap();
    |                              ^^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 106 (Col 9)
**ERROR[E0425]**: cannot find function `seed` in this scope

```
error[E0425]: cannot find function `seed` in this scope
   --> src/random/tests.rs:106:9
    |
106 |         seed(12345);
    |         ^^^^ not found in this scope
    |
```

### Line 107 (Col 26)
**ERROR[E0425]**: cannot find function `legacy_random` in this scope

```
error[E0425]: cannot find function `legacy_random` in this scope
   --> src/random/tests.rs:107:26
    |
107 |         let seeded_arr = legacy_random::<f64>(&[2, 2], Dtype::Float64).unwrap();
    |                          ^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 117 (Col 23)
**ERROR[E0425]**: cannot find function `default_rng` in this scope

```
error[E0425]: cannot find function `default_rng` in this scope
   --> src/random/tests.rs:117:23
    |
117 |         let mut rng = default_rng();
    |                       ^^^^^^^^^^^ not found in this scope
    |
```

### Line 122 (Col 30)
**ERROR[E0425]**: cannot find function `default_rng_with_seed` in this scope

```
error[E0425]: cannot find function `default_rng_with_seed` in this scope
   --> src/random/tests.rs:122:30
    |
 20 |     fn test_default_rng_with_seed() {
    |     ------------------------------- similarly named constant `test_default_rng_with_seed` defined here
...
122 |         let mut seeded_rng = default_rng_with_seed(42);
    |                              ^^^^^^^^^^^^^^^^^^^^^
    |
```

### Line 142 (Col 23)
**ERROR[E0425]**: cannot find function `legacy_rng` in this scope

```
error[E0425]: cannot find function `legacy_rng` in this scope
   --> src/random/tests.rs:142:23
    |
142 |         let _legacy = legacy_rng();
    |                       ^^^^^^^^^^ not found in this scope
    |
```

### Line 145 (Col 9)
**ERROR[E0425]**: cannot find function `seed` in this scope

```
error[E0425]: cannot find function `seed` in this scope
   --> src/random/tests.rs:145:9
    |
145 |         seed(54321);
    |         ^^^^ not found in this scope
    |
```

### Line 148 (Col 26)
**ERROR[E0425]**: cannot find function `legacy_random` in this scope

```
error[E0425]: cannot find function `legacy_random` in this scope
   --> src/random/tests.rs:148:26
    |
148 |         let legacy_arr = legacy_random::<f64>(&[2, 2], Dtype::Float64).unwrap();
    |                          ^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 152 (Col 30)
**ERROR[E0425]**: cannot find function `legacy_randint` in this scope

```
error[E0425]: cannot find function `legacy_randint` in this scope
   --> src/random/tests.rs:152:30
    |
152 |         let legacy_int_arr = legacy_randint::<i32>(0, 10, &[2, 2]).unwrap();
    |                              ^^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 159 (Col 23)
**ERROR[E0425]**: cannot find function `default_rng` in this scope

```
error[E0425]: cannot find function `default_rng` in this scope
   --> src/random/tests.rs:159:23
    |
159 |         let mut rng = default_rng();
    |                       ^^^^^^^^^^^ not found in this scope
    |
```

### Line 186 (Col 23)
**ERROR[E0433]**: failed to resolve: use of undeclared type `PCG64`

```
error[E0433]: failed to resolve: use of undeclared type `PCG64`
   --> src/random/tests.rs:186:23
    |
186 |         let mut pcg = PCG64::new();
    |                       ^^^^^ use of undeclared type `PCG64`
    |
```

### Line 195 (Col 26)
**ERROR[E0433]**: failed to resolve: use of undeclared type `PCG64`

```
error[E0433]: failed to resolve: use of undeclared type `PCG64`
   --> src/random/tests.rs:195:26
    |
195 |         let seeded_pcg = PCG64::seed_from_u64(54321);
    |                          ^^^^^ use of undeclared type `PCG64`
    |
```

### Line 199 (Col 23)
**ERROR[E0433]**: failed to resolve: use of undeclared type `Generator`

```
error[E0433]: failed to resolve: use of undeclared type `Generator`
   --> src/random/tests.rs:199:23
    |
199 |         let mut rng = Generator::new(Box::new(seeded_pcg));
    |                       ^^^^^^^^^ use of undeclared type `Generator`
    |
```

### Line 209 (Col 20)
**ERROR[E0425]**: cannot find function `random` in this scope

```
error[E0425]: cannot find function `random` in this scope
   --> src/random/tests.rs:209:20
    |
209 |         let arr1 = random::<f64>(&[2, 2], Dtype::Float64).unwrap();
    |                    ^^^^^^ not found in this scope
    |
```

### Line 212 (Col 20)
**ERROR[E0425]**: cannot find function `random` in this scope

```
error[E0425]: cannot find function `random` in this scope
   --> src/random/tests.rs:212:20
    |
212 |         let arr2 = random::<f64>(&[2, 2], Dtype::Float64).unwrap();
    |                    ^^^^^^ not found in this scope
    |
```

### Line 224 (Col 26)
**ERROR[E0425]**: cannot find function `random` in this scope

```
error[E0425]: cannot find function `random` in this scope
   --> src/random/tests.rs:224:26
    |
224 |         let modern_arr = random::<f64>(&[2, 2], Dtype::Float64).unwrap();
    |                          ^^^^^^ not found in this scope
    |
```

### Line 228 (Col 26)
**ERROR[E0425]**: cannot find function `legacy_random` in this scope

```
error[E0425]: cannot find function `legacy_random` in this scope
   --> src/random/tests.rs:228:26
    |
228 |         let legacy_arr = legacy_random::<f64>(&[2, 2], Dtype::Float64).unwrap();
    |                          ^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 238 (Col 23)
**ERROR[E0425]**: cannot find function `default_rng` in this scope

```
error[E0425]: cannot find function `default_rng` in this scope
   --> src/random/tests.rs:238:23
    |
238 |         let mut rng = default_rng();
    |                       ^^^^^^^^^^^ not found in this scope
    |
```

### Line 277 (Col 21)
**ERROR[E0425]**: cannot find function `legacy_random` in this scope

```
error[E0425]: cannot find function `legacy_random` in this scope
   --> src/random/tests.rs:277:21
    |
277 |         let _arr1 = legacy_random::<f64>(shape, Dtype::Float64).unwrap();
    |                     ^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 278 (Col 21)
**ERROR[E0425]**: cannot find function `legacy_randint` in this scope

```
error[E0425]: cannot find function `legacy_randint` in this scope
   --> src/random/tests.rs:278:21
    |
278 |         let _arr2 = legacy_randint::<i32>(0, 10, shape).unwrap();
    |                     ^^^^^^^^^^^^^^ not found in this scope
    |
```

### Line 281 (Col 21)
**ERROR[E0425]**: cannot find function `random` in this scope

```
error[E0425]: cannot find function `random` in this scope
   --> src/random/tests.rs:281:21
    |
281 |         let _arr3 = random::<f64>(shape, Dtype::Float64).unwrap();
    |                     ^^^^^^ not found in this scope
    |
```

### Line 282 (Col 21)
**ERROR[E0425]**: cannot find function `randint` in this scope

```
error[E0425]: cannot find function `randint` in this scope
   --> src/random/tests.rs:282:21
    |
282 |         let _arr4 = randint::<i32>(0, 10, shape).unwrap();
    |                     ^^^^^^^ not found in this scope
    |
```

### Line 296 (Col 24)
**ERROR[E0425]**: cannot find function `default_rng_with_seed` in this scope

```
error[E0425]: cannot find function `default_rng_with_seed` in this scope
   --> src/random/tests.rs:296:24
    |
 20 |     fn test_default_rng_with_seed() {
    |     ------------------------------- similarly named constant `test_default_rng_with_seed` defined here
...
296 |         let mut rng1 = default_rng_with_seed(seed);
    |                        ^^^^^^^^^^^^^^^^^^^^^
    |
```

### Line 300 (Col 19)
**ERROR[E0433]**: failed to resolve: use of undeclared type `PCG64`

```
error[E0433]: failed to resolve: use of undeclared type `PCG64`
   --> src/random/tests.rs:300:19
    |
300 |         let pcg = PCG64::seed_from_u64(seed);
    |                   ^^^^^ use of undeclared type `PCG64`
    |
```

### Line 301 (Col 24)
**ERROR[E0433]**: failed to resolve: use of undeclared type `Generator`

```
error[E0433]: failed to resolve: use of undeclared type `Generator`
   --> src/random/tests.rs:301:24
    |
301 |         let mut rng2 = Generator::new(Box::new(pcg));
    |                        ^^^^^^^^^ use of undeclared type `Generator`
    |
```

## Warnings

### Line 3 (Col 9)
**WARNING[unused_imports]**: warning: unused import: `super::*`

```
warning: unused import: `super::*`
  |
3 |     use super::*;
  |         ^^^^^^^^
  |
  = note: `#[warn(unused_imports)]` (part of `#[warn(unused)]`) on by default
```
