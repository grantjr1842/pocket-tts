// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use numpy::dtype::Dtype;
use numpy::random::bit_generator::PCG64;
use numpy::random::generator::Generator;
use numpy::random::RandomState;

#[test]
fn test_generator_creation() {
    let bit_gen = PCG64::new();
    let mut gen = Generator::new(Box::new(bit_gen));
    let res = gen.normal(0.0, 1.0, &[5]).unwrap();
    assert_eq!(res.shape(), vec![5]);
}

#[test]
fn test_random_state_delegation() {
    let mut rs = RandomState::new();
    let res = rs.normal(0.0, 1.0, &[5]).unwrap();
    assert_eq!(res.shape(), vec![5]);
}

#[test]
fn test_global_proxies() {
    use numpy::random::*;
    let res = normal(0.0, 1.0, &[5]).unwrap();
    assert_eq!(res.shape(), vec![5]);
}

#[test]
fn test_randint() {
    use numpy::random::*;
    let res = randint(0, 10, &[10]).unwrap();
    assert_eq!(res.shape(), vec![10]);
    for i in 0..10 {
        let val = *res.get_linear(i).unwrap();
        assert!(val >= 0 && val < 10);
    }
}
