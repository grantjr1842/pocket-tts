# Diagnostics: rust-numpy/examples/generator_examples.rs

**Errors:** 38 | **Warnings:** 0

---

## Errors

### Line 21 (Col 49)
**ERROR[E0533]**: expected value, found struct variant `Dtype::Float64`

```
error[E0533]: expected value, found struct variant `Dtype::Float64`
  --> examples/generator_examples.rs:21:49
   |
21 |     let random_arr = rng.random::<f64>(&[3, 4], Dtype::Float64)?;
   |                                                 ^^^^^^^^^^^^^^ not a value
   |
```

### Line 50 (Col 44)
**ERROR[E0533]**: expected value, found struct variant `Dtype::Float64`

```
error[E0533]: expected value, found struct variant `Dtype::Float64`
  --> examples/generator_examples.rs:50:44
   |
50 |     let arr1 = rng1.random::<f64>(&[2, 2], Dtype::Float64)?;
   |                                            ^^^^^^^^^^^^^^ not a value
   |
```

### Line 51 (Col 44)
**ERROR[E0533]**: expected value, found struct variant `Dtype::Float64`

```
error[E0533]: expected value, found struct variant `Dtype::Float64`
  --> examples/generator_examples.rs:51:44
   |
51 |     let arr2 = rng2.random::<f64>(&[2, 2], Dtype::Float64)?;
   |                                            ^^^^^^^^^^^^^^ not a value
   |
```

### Line 60 (Col 56)
**ERROR[E0533]**: expected value, found struct variant `Dtype::Float64`

```
error[E0533]: expected value, found struct variant `Dtype::Float64`
  --> examples/generator_examples.rs:60:56
   |
60 |     let manual_arr = manual_rng.random::<f64>(&[2, 2], Dtype::Float64)?;
   |                                                        ^^^^^^^^^^^^^^ not a value
   |
```

### Line 127 (Col 71)
**ERROR[E0614]**: type `f64` cannot be dereferenced

```
error[E0614]: type `f64` cannot be dereferenced
   --> examples/generator_examples.rs:127:71
    |
127 |     let bern_true_rate = bern_arr.data.as_slice().iter().filter(|&&x| *x == 1.0).count() as f64 / 1000.0;
    |                                                                       ^^ can't be dereferenced
```

### Line 222 (Col 22)
**ERROR[E0308]**: expected {closure#340993}, found {closure#340994}

### Line 222 (Col 29)
**ERROR[E0599]**: no method named `logistic` found for struct `Generator` in the current scope

```
error[E0599]: no method named `logistic` found for struct `Generator` in the current scope
   --> examples/generator_examples.rs:222:29
    |
222 |         ("Logistic", || rng.logistic::<f64>(0.0, 1.0, &[1000])),
    |                             ^^^^^^^^ method not found in `Generator`
```

### Line 223 (Col 20)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:223:20
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
222 |         ("Logistic", || rng.logistic::<f64>(0.0, 1.0, &[1000])),
223 |         ("Gumbel", || rng.gumbel::<f64>(0.0, 1.0, &[1000])),
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:223:20: 223:22}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 224 (Col 18)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:224:18
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
...
224 |         ("Wald", || rng.wald::<f64>(1.0, 1.0, &[1000])),
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:224:18: 224:20}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 225 (Col 21)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:225:21
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
...
225 |         ("Weibull", || rng.weibull::<f64>(2.0, &[1000])),
    |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:225:21: 225:23}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 226 (Col 24)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:226:24
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
...
226 |         ("Triangular", || rng.triangular::<f64>(0.0, 0.5, 1.0, &[1000])),
    |                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:226:24: 226:26}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 227 (Col 20)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:227:20
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
...
227 |         ("Pareto", || rng.pareto::<f64>(2.0, &[1000])),
    |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:227:20: 227:22}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 228 (Col 18)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:228:18
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
...
228 |         ("Zipf", || rng.zipf::<f64>(3.0, &[1000])),
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:228:18: 228:20}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 229 (Col 29)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:229:29
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
...
229 |         ("Standard Cauchy", || rng.standard_cauchy::<f64>(&[1000])),
    |                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:229:29: 229:31}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 230 (Col 34)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:230:34
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
...
230 |         ("Standard Exponential", || rng.standard_exponential::<f64>(&[1000])),
    |                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:230:34: 230:36}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 231 (Col 28)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:231:28
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
...
231 |         ("Standard Gamma", || rng.standard_gamma::<f64>(2.0, &[1000])),
    |                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:231:28: 231:30}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 232 (Col 28)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:232:28
    |
221 |         ("Log-Normal", || rng.lognormal::<f64>(0.0, 1.0, &[1000])),
    |                        -- the expected closure
...
232 |         ("F-Distribution", || rng.f::<f64>(2.0, 2.0, &[1000])),
    |                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:221:24: 221:26}`
               found closure `{closure@examples/generator_examples.rs:232:28: 232:30}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 233 (Col 19)
**ERROR[E0308]**: expected {closure#340993}, found {closure#341005}

### Line 233 (Col 26)
**ERROR[E0599]**: no method named `power` found for struct `Generator` in the current scope

```
error[E0599]: no method named `power` found for struct `Generator` in the current scope
   --> examples/generator_examples.rs:233:26
    |
233 |         ("Power", || rng.power::<f64>(2.0, &[1000])),
    |                          ^^^^^ method not found in `Generator`
```

### Line 234 (Col 23)
**ERROR[E0308]**: expected {closure#340993}, found {closure#341006}

### Line 234 (Col 30)
**ERROR[E0599]**: no method named `vonmises` found for struct `Generator` in the current scope

```
error[E0599]: no method named `vonmises` found for struct `Generator` in the current scope
   --> examples/generator_examples.rs:234:30
    |
234 |         ("von Mises", || rng.vonmises::<f64>(0.0, 1.0, &[1000])),
    |                              ^^^^^^^^ method not found in `Generator`
```

### Line 261 (Col 36)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:261:36
    |
260 |         ("Negative scale (exponential)", || rng.exponential::<f64>(-1.0, &[2, 2])),
    |                                          -- the expected closure
261 |         ("Negative shape (gamma)", || rng.gamma::<f64>(-1.0, 1.0, &[2, 2])),
    |                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:260:42: 260:44}`
               found closure `{closure@examples/generator_examples.rs:261:36: 261:38}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 262 (Col 44)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:262:44
    |
260 |         ("Negative scale (exponential)", || rng.exponential::<f64>(-1.0, &[2, 2])),
    |                                          -- the expected closure
261 |         ("Negative shape (gamma)", || rng.gamma::<f64>(-1.0, 1.0, &[2, 2])),
262 |         ("Invalid probability (binomial)", || rng.binomial::<f64>(10, -0.5, &[2, 2])),
    |                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:260:42: 260:44}`
               found closure `{closure@examples/generator_examples.rs:262:44: 262:46}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 263 (Col 38)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:263:38
    |
260 |         ("Negative scale (exponential)", || rng.exponential::<f64>(-1.0, &[2, 2])),
    |                                          -- the expected closure
...
263 |         ("Invalid range (integers)", || rng.integers(10, 5, 5)),
    |                                      ^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:260:42: 260:44}`
               found closure `{closure@examples/generator_examples.rs:263:38: 263:40}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 264 (Col 36)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:264:36
    |
260 |         ("Negative scale (exponential)", || rng.exponential::<f64>(-1.0, &[2, 2])),
    |                                          -- the expected closure
...
264 |         ("Invalid range (floats)", || rng.random_floats_range(10.0, 5.0, 5)),
    |                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:260:42: 260:44}`
               found closure `{closure@examples/generator_examples.rs:264:36: 264:38}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 265 (Col 27)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:265:27
    |
260 |         ("Negative scale (exponential)", || rng.exponential::<f64>(-1.0, &[2, 2])),
    |                                          -- the expected closure
...
265 |         ("Empty choices", || rng.choice(&Vec::<i32>::new(), 3, false)),
    |                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:260:42: 260:44}`
               found closure `{closure@examples/generator_examples.rs:265:27: 265:29}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 266 (Col 50)
**ERROR[E0308]**: mismatched types

```
error[E0308]: mismatched types
   --> examples/generator_examples.rs:266:50
    |
260 |           ("Negative scale (exponential)", || rng.exponential::<f64>(-1.0, &[2, 2])),
    |                                            -- the expected closure
...
266 |           ("Too large sample without replacement", || {
    |  __________________________________________________^
267 | |             let small_choices = vec![1, 2];
268 | |             rng.choice(&small_choices, 5, false)
269 | |         }),
    | |_________^ expected closure, found a different closure
    |
    = note: expected closure `{closure@examples/generator_examples.rs:260:42: 260:44}`
               found closure `{closure@examples/generator_examples.rs:266:50: 266:52}`
    = note: no two closures, even if identical, have the same type
    = help: consider boxing your closure and/or using it as a trait object
```

### Line 299 (Col 44)
**ERROR[E0533]**: expected value, found struct variant `Dtype::Float64`

```
error[E0533]: expected value, found struct variant `Dtype::Float64`
   --> examples/generator_examples.rs:299:44
    |
299 |         let _ = rng.random::<f64>(&[size], Dtype::Float64)?;
    |                                            ^^^^^^^^^^^^^^ not a value
    |
```

### Line 317 (Col 50)
**ERROR[E0533]**: expected value, found struct variant `Dtype::Float64`

```
error[E0533]: expected value, found struct variant `Dtype::Float64`
   --> examples/generator_examples.rs:317:50
    |
317 |             let arr = rng.random::<f64>(&[1000], Dtype::Float64)?;
    |                                                  ^^^^^^^^^^^^^^ not a value
    |
```

### Line 319 (Col 13)
**ERROR[E0107]**: enum takes 2 generic arguments but 1 generic argument was supplied

```
error[E0107]: enum takes 2 generic arguments but 1 generic argument was supplied
   --> examples/generator_examples.rs:319:13
    |
319 |             Ok::<f64>(mean)
    |             ^^   --- supplied 1 generic argument
    |             |
    |             expected 2 generic arguments
    |
```

### Line 365 (Col 65)
**ERROR[E0614]**: type `f64` cannot be dereferenced

```
error[E0614]: type `f64` cannot be dereferenced
   --> examples/generator_examples.rs:365:65
    |
365 |         let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(*b));
    |                                                                 ^^ can't be dereferenced
```

### Line 366 (Col 69)
**ERROR[E0614]**: type `f64` cannot be dereferenced

```
error[E0614]: type `f64` cannot be dereferenced
   --> examples/generator_examples.rs:366:69
    |
366 |         let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(*b));
    |                                                                     ^^ can't be dereferenced
```

### Line 385 (Col 53)
**ERROR[E0405]**: cannot find trait `BitGenerator` in this scope

```
error[E0405]: cannot find trait `BitGenerator` in this scope
   --> examples/generator_examples.rs:385:53
    |
385 |         ("PCG64", Box::new(PCG64::new()) as Box<dyn BitGenerator>),
    |                                                     ^^^^^^^^^^^^ not found in this scope
    |
```

### Line 386 (Col 77)
**ERROR[E0405]**: cannot find trait `BitGenerator` in this scope

```
error[E0405]: cannot find trait `BitGenerator` in this scope
   --> examples/generator_examples.rs:386:77
    |
386 |         ("PCG64 (seeded)", Box::new(PCG64::seed_from_u64(12345)) as Box<dyn BitGenerator>),
    |                                                                             ^^^^^^^^^^^^ not found in this scope
    |
```

### Line 406 (Col 23)
**ERROR[E0599]**: no method named `next_u32` found for struct `Generator` in the current scope

```
error[E0599]: no method named `next_u32` found for struct `Generator` in the current scope
   --> examples/generator_examples.rs:406:23
    |
406 |     let u32_val = rng.next_u32();
    |                       ^^^^^^^^
    |
   ::: /home/grant/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rand_core-0.6.4/src/lib.rs:148:8
    |
148 |     fn next_u32(&mut self) -> u32;
    |        -------- the method is available for `Generator` here
    |
    = help: items from traits can only be used if the trait is in scope
```

### Line 407 (Col 23)
**ERROR[E0599]**: no method named `next_u64` found for struct `Generator` in the current scope

```
error[E0599]: no method named `next_u64` found for struct `Generator` in the current scope
   --> examples/generator_examples.rs:407:23
    |
407 |     let u64_val = rng.next_u64();
    |                       ^^^^^^^^
    |
   ::: /home/grant/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rand_core-0.6.4/src/lib.rs:155:8
    |
155 |     fn next_u64(&mut self) -> u64;
    |        -------- the method is available for `Generator` here
    |
    = help: items from traits can only be used if the trait is in scope
```

### Line 414 (Col 9)
**ERROR[E0599]**: no method named `fill_bytes` found for struct `Generator` in the current scope

```
error[E0599]: no method named `fill_bytes` found for struct `Generator` in the current scope
   --> examples/generator_examples.rs:414:9
    |
414 |     rng.fill_bytes(&mut bytes);
    |         ^^^^^^^^^^
    |
   ::: /home/grant/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rand_core-0.6.4/src/lib.rs:171:8
    |
171 |     fn fill_bytes(&mut self, dest: &mut [u8]);
    |        ---------- the method is available for `Generator` here
    |
    = help: items from traits can only be used if the trait is in scope
```

### Line 420 (Col 29)
**ERROR[E0599]**: no method named `gen` found for struct `Generator` in the current scope

```
error[E0599]: no method named `gen` found for struct `Generator` in the current scope
   --> examples/generator_examples.rs:420:29
    |
420 |         rng_floats.push(rng.gen::<f64>());
    |                             ^^^ method not found in `Generator`
    |
   ::: /home/grant/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rand-0.8.5/src/rng.rs:93:8
    |
 93 |     fn gen<T>(&mut self) -> T
    |        --- the method is available for `Generator` here
    |
    = help: items from traits can only be used if the trait is in scope
```
