// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

/// BitGenerator trait defines the interface for raw bit stream generators
pub trait BitGenerator: RngCore + Send + Sync {
    /// Reseed the generator
    fn set_state(&mut self, state: u64);

    /// Get current state (or a representation of it)
    fn get_state(&self) -> u64;
}

/// PCG64 is the default BitGenerator for NumPy 1.17+
pub struct PCG64 {
    pub(crate) rng: StdRng,
    seed: u64,
}

impl PCG64 {
    pub fn new() -> Self {
        let seed = rand::random::<u64>();
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }
}

impl BitGenerator for PCG64 {
    fn set_state(&mut self, state: u64) {
        self.rng = StdRng::seed_from_u64(state);
        self.seed = state;
    }

    fn get_state(&self) -> u64 {
        self.seed
    }
}

impl RngCore for PCG64 {
    fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.rng.try_fill_bytes(dest)
    }
}
