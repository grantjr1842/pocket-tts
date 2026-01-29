# Diagnostics: rust-numpy/examples/random_examples.rs

**Errors:** 21 | **Warnings:** 9

---

## Errors

### Line 7 (Col 17)
**ERROR[E0432]**: unresolved import `rust_numpy::prelude`

```
error[E0432]: unresolved import `rust_numpy::prelude`
  |
7 | use rust_numpy::prelude::*;
  |                 ^^^^^^^ could not find `prelude` in `rust_numpy`
```

### Line 214 (Col 21)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:214:21
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
214 |         ("randint", || rng.randint::<i32>(0, 10, &[2, 2])),
    |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:214:21: 214:23}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 215 (Col 21)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:215:21
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
214 |         ("randint", || rng.randint::<i32>(0, 10, &[2, 2])),
215 |         ("uniform", || rng.uniform::<f64>(0.0, 1.0, &[2, 2])),
    |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:215:21: 215:23}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 216 (Col 20)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:216:20
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
216 |         ("normal", || rng.normal::<f64>(0.0, 1.0, &[2, 2])),
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:216:20: 216:22}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 217 (Col 29)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:217:29
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
217 |         ("standard_normal", || rng.standard_normal::<f64>(&[2, 2])),
    |                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:217:29: 217:31}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 218 (Col 22)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:218:22
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
218 |         ("binomial", || rng.binomial::<f64>(10, 0.5, &[2, 2])),
    |                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:218:22: 218:24}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 219 (Col 21)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:219:21
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
219 |         ("poisson", || rng.poisson::<f64>(5.0, &[2, 2])),
    |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:219:21: 219:23}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 220 (Col 25)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:220:25
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
220 |         ("exponential", || rng.exponential::<f64>(1.0, &[2, 2])),
    |                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:220:25: 220:27}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 221 (Col 19)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:221:19
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
221 |         ("gamma", || rng.gamma::<f64>(2.0, 2.0, &[2, 2])),
    |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:221:19: 221:21}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 222 (Col 18)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:222:18
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
222 |         ("beta", || rng.beta::<f64>(2.0, 2.0, &[2, 2])),
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:222:18: 222:20}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 223 (Col 23)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:223:23
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
223 |         ("chisquare", || rng.chisquare::<f64>(2.0, &[2, 2])),
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:223:23: 223:25}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 224 (Col 23)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:224:23
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
224 |         ("bernoulli", || rng.bernoulli::<f64>(0.5, &[2, 2])),
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:224:23: 224:25}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 225 (Col 23)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:225:23
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
225 |         ("lognormal", || rng.lognormal::<f64>(0.0, 1.0, &[2, 2])),
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:225:23: 225:25}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 226 (Col 29)
**ERROR[E0599]**: no method named `logistic` found for struct `Generator` in the current scope

```
error[E0599]: no method named `logistic` found for struct `Generator` in the current scope
   --> examples/random_examples.rs:226:29
    |
226 |         ("logistic", || rng.logistic::<f64>(0.0, 1.0, &[2, 2])),
    |                             ^^^^^^^^ method not found in `Generator`
```

### Line 227 (Col 23)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:227:23
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
227 |         ("geometric", || rng.geometric::<f64>(0.5, &[2, 2])),
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:227:23: 227:25}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 228 (Col 29)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:228:29
    |
213 |         ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                    -- the expected closure
...
228 |         ("standard_cauchy", || rng.standard_cauchy::<f64>(&[2, 2])),
    |                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:228:29: 228:31}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 229 (Col 34)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/random_examples.rs:229:34
    |
213 |           ("random", || rng.random::<f64>(&[2, 2], Dtype::Float64)),
    |                      -- the expected closure
...
229 |           ("standard_exponential", || {
    |  __________________________________^
230 | |             rng.standard_exponential::<f64>(&[2, 2])
231 | |         }),
    | |_________^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/random_examples.rs:213:20: 213:22}`
               found closure `{closure@examples/random_examples.rs:229:34: 229:36}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 297 (Col 9)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> examples/random_examples.rs:297:9
    |
297 |     let arr1 = handle1.join().unwrap()?;
    |         ^^^^
...
300 |     println!("Thread 1 array shape: {:?}", arr1.shape());
    |                                            ---- type must be known at this point
    |
```

### Line 297 (Col 16)
**ERROR[E0277]**: the `?` operator can only be applied to values that implement `Try`

```
error[E0277]: the `?` operator can only be applied to values that implement `Try`
   --> examples/random_examples.rs:297:16
    |
297 |     let arr1 = handle1.join().unwrap()?;
    |                ^^^^^^^^^^^^^^^^^^^^^^^^ the `?` operator cannot be applied to type `rust_numpy::Array<f64>`
    |
    = help: the trait `Try` is not implemented for `rust_numpy::Array<f64>`
```

### Line 298 (Col 9)
**ERROR[E0282]**: type annotations needed

```
error[E0282]: type annotations needed
   --> examples/random_examples.rs:298:9
    |
298 |     let arr2 = handle2.join().unwrap()?;
    |         ^^^^
...
301 |     println!("Thread 2 array shape: {:?}", arr2.shape());
    |                                            ---- type must be known at this point
    |
```

### Line 298 (Col 16)
**ERROR[E0277]**: the `?` operator can only be applied to values that implement `Try`

```
error[E0277]: the `?` operator can only be applied to values that implement `Try`
   --> examples/random_examples.rs:298:16
    |
298 |     let arr2 = handle2.join().unwrap()?;
    |                ^^^^^^^^^^^^^^^^^^^^^^^^ the `?` operator cannot be applied to type `rust_numpy::Array<f64>`
    |
    = help: the trait `Try` is not implemented for `rust_numpy::Array<f64>`
```

## Warnings

### Line 9 (Col 25)
**WARNING[deprecated]**: warning: use of deprecated module `rust_numpy::random::legacy`: Use modern Generator API instead

```
warning: use of deprecated module `rust_numpy::random::legacy`: Use modern Generator API instead
  |
9 | use rust_numpy::random::legacy;
  |                         ^^^^^^
  |
  = note: `#[warn(deprecated)]` on by default
```

### Line 159 (Col 30)
**WARNING[deprecated]**: warning: use of deprecated function `rust_numpy::random::legacy_random`: Use random() function which uses modern Generator API

```
warning: use of deprecated function `rust_numpy::random::legacy_random`: Use random() function which uses modern Generator API
   --> examples/random_examples.rs:159:30
    |
159 |     let legacy_arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64)?;
    |                              ^^^^^^^^^^^^^
```

### Line 162 (Col 34)
**WARNING[deprecated]**: warning: use of deprecated function `rust_numpy::random::legacy_randint`: Use randint() function which uses modern Generator API

```
warning: use of deprecated function `rust_numpy::random::legacy_randint`: Use randint() function which uses modern Generator API
   --> examples/random_examples.rs:162:34
    |
162 |     let legacy_int_arr = legacy::legacy_randint::<i32>(0, 10, &[2, 2])?;
    |                                  ^^^^^^^^^^^^^^
```

### Line 166 (Col 13)
**WARNING[deprecated]**: warning: use of deprecated function `rust_numpy::random::seed`: Use default_rng_with_seed(seed) instead for new code

```
warning: use of deprecated function `rust_numpy::random::seed`: Use default_rng_with_seed(seed) instead for new code
   --> examples/random_examples.rs:166:13
    |
166 |     legacy::seed(12345);
    |             ^^^^
```

### Line 167 (Col 37)
**WARNING[deprecated]**: warning: use of deprecated function `rust_numpy::random::legacy_random`: Use random() function which uses modern Generator API

```
warning: use of deprecated function `rust_numpy::random::legacy_random`: Use random() function which uses modern Generator API
   --> examples/random_examples.rs:167:37
    |
167 |     let seeded_legacy_arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64)?;
    |                                     ^^^^^^^^^^^^^
```

### Line 171 (Col 30)
**WARNING[deprecated]**: warning: use of deprecated function `rust_numpy::random::legacy_rng`: Use default_rng() instead for new code

```
warning: use of deprecated function `rust_numpy::random::legacy_rng`: Use default_rng() instead for new code
   --> examples/random_examples.rs:171:30
    |
171 |     let legacy_rng = legacy::legacy_rng();
    |                              ^^^^^^^^^^
```

### Line 186 (Col 30)
**WARNING[deprecated]**: warning: use of deprecated function `rust_numpy::random::legacy_random`: Use random() function which uses modern Generator API

```
warning: use of deprecated function `rust_numpy::random::legacy_random`: Use random() function which uses modern Generator API
   --> examples/random_examples.rs:186:30
    |
186 |     let legacy_arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64)?;
    |                              ^^^^^^^^^^^^^
```

### Line 253 (Col 13)
**WARNING[deprecated]**: warning: use of deprecated function `rust_numpy::random::seed`: Use default_rng_with_seed(seed) instead for new code

```
warning: use of deprecated function `rust_numpy::random::seed`: Use default_rng_with_seed(seed) instead for new code
   --> examples/random_examples.rs:253:13
    |
253 |     legacy::seed(42);
    |             ^^^^
```

### Line 254 (Col 27)
**WARNING[deprecated]**: warning: use of deprecated function `rust_numpy::random::legacy_random`: Use random() function which uses modern Generator API

```
warning: use of deprecated function `rust_numpy::random::legacy_random`: Use random() function which uses modern Generator API
   --> examples/random_examples.rs:254:27
    |
254 |     let old_arr = legacy::legacy_random::<f64>(&[2, 2], Dtype::Float64)?;
    |                           ^^^^^^^^^^^^^
```
