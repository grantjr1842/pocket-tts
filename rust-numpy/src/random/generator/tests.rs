// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
// Tests for the Generator implementation.
//
// Most tests are located in src/random/tests.rs for easier organization
// across the random module.

#[cfg(test)]
mod generator_tests {
    use super::*;
    use crate::array::Array;
    use crate::dtype::Dtype;
    use crate::random::bit_generator::PCG64;

    #[test]
    fn test_generator_creation() {
        let bit_gen = Box::new(PCG64::new());
        let mut gen = Generator::new(bit_gen);

        // Test that we can generate random numbers
        let result = gen.random::<f64>(&[3], Dtype::Float64 { byteorder: None });
        assert!(result.is_ok());
    }

    #[test]
    fn test_shuffle_1d() {
        let bit_gen = Box::new(PCG64::seed_from_u64(42));
        let mut gen = Generator::new(bit_gen);

        let mut arr = Array::from_data(vec![1, 2, 3, 4, 5], vec![5]);
        let original = arr.data.clone();

        let result = gen.shuffle(&mut arr);
        assert!(result.is_ok());

        // After shuffle, the elements should be the same but possibly in different order
        let mut sorted_original = original;
        sorted_original.sort();
        let mut sorted_arr = arr.data.clone();
        sorted_arr.sort();
        assert_eq!(sorted_original, sorted_arr);
    }
}
