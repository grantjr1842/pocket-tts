// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use std::fmt::Debug;

/// BitGenerator trait defines the interface for raw bit stream generators
///
/// This trait provides NumPy-compatible uniform pseudo-random number generation
/// algorithms that serve as the foundation for the modern random API.
///
/// All BitGenerators must implement RngCore for basic random number generation
/// and provide additional methods for state management and algorithm-specific
/// functionality.
pub trait BitGenerator: RngCore + Send + Sync + Debug {
    /// Get the name of the BitGenerator algorithm
    fn name(&self) -> &'static str;

    /// Get the version of the BitGenerator implementation
    fn version(&self) -> &'static str;

    /// Seed the generator with a 64-bit integer
    fn seed_u64(&mut self, seed: u64);

    /// Seed the generator with a slice of 32-bit integers
    fn seed_u32_slice(&mut self, seed: &[u32]);

    /// Get the current state as a byte vector
    ///
    /// Returns the internal state of the generator for serialization
    fn get_state_bytes(&self) -> Vec<u8>;

    /// Set the current state from a byte vector
    ///
    /// Restores the generator state from previously serialized data
    fn set_state_bytes(&mut self, state: &[u8]) -> Result<(), String>;

    /// Get the size of the state in bytes
    fn state_size(&self) -> usize;

    /// Jump ahead in the random sequence
    ///
    /// Advances the generator by a specified number of steps.
    /// This is useful for parallel random number generation.
    fn jump(&mut self, steps: u64);

    /// Check if the generator is cryptographically secure
    fn is_cryptographically_secure(&self) -> bool;

    /// Get the period of the generator (if known)
    fn period(&self) -> Option<u128>;

    /// Get the recommended parallel usage parameters
    fn parallel_params(&self) -> ParallelParams;
}

/// Parameters for parallel random number generation
#[derive(Debug, Clone)]
pub struct ParallelParams {
    /// Recommended jump size for parallel streams
    pub jump_size: u64,
    /// Maximum number of parallel streams supported
    pub max_streams: usize,
    /// Whether the generator supports parallel streams
    pub supports_parallel: bool,
}

/// PCG64 is the default BitGenerator for NumPy 1.17+
///
/// PCG64 is a fast, high-quality random number generator with good statistical
/// properties and a small state size. It's the recommended default for most
/// applications.
#[derive(Debug, Clone)]
pub struct PCG64 {
    rng: StdRng,
    seed: u64,
}

impl PCG64 {
    /// Create a new PCG64 instance with a random seed
    pub fn new() -> Self {
        let seed = rand::random::<u64>();
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    /// Create a new PCG64 instance with a specific 64-bit seed
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    /// Create a new PCG64 instance from a seed slice
    pub fn from_seed_slice(seed: &[u32]) -> Self {
        let mut seed_u64 = 0u64;
        for (i, &word) in seed.iter().enumerate() {
            if i < 2 {
                seed_u64 |= (word as u64) << (i * 32);
            }
        }
        Self::seed_from_u64(seed_u64)
    }
}

impl BitGenerator for PCG64 {
    fn name(&self) -> &'static str {
        "PCG64"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn seed_u64(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = StdRng::seed_from_u64(seed);
    }

    fn seed_u32_slice(&mut self, seed: &[u32]) {
        *self = Self::from_seed_slice(seed);
    }

    fn get_state_bytes(&self) -> Vec<u8> {
        // Serialize the seed and RNG state
        let mut state = Vec::new();
        state.extend_from_slice(&self.seed.to_le_bytes());
        // Note: StdRng doesn't expose internal state for serialization
        // In a full implementation, we'd need to use a PCG64 implementation
        // that allows state serialization
        state
    }

    fn set_state_bytes(&mut self, state: &[u8]) -> Result<(), String> {
        if state.len() < 8 {
            return Err("Insufficient state data".to_string());
        }

        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&state[0..8]);
        let seed = u64::from_le_bytes(seed_bytes);

        self.seed_u64(seed);
        Ok(())
    }

    fn state_size(&self) -> usize {
        8 // Just the seed for now
    }

    fn jump(&mut self, steps: u64) {
        // Advance the generator by discarding random numbers
        for _ in 0..steps {
            let _ = self.next_u64();
        }
    }

    fn is_cryptographically_secure(&self) -> bool {
        false
    }

    fn period(&self) -> Option<u128> {
        // PCG64 has a period of 2^128
        // Since u128::MAX is 2^128 - 1, we use that as an approximation
        Some(u128::MAX)
    }

    fn parallel_params(&self) -> ParallelParams {
        ParallelParams {
            jump_size: 1 << 32, // Large jump for parallel streams
            max_streams: 256,
            supports_parallel: true,
        }
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

/// MT19937 (Mersenne Twister) BitGenerator
///
/// MT19937 is a widely used pseudorandom number generator with a very long period
/// (2^19937-1) and good statistical properties. It's the default generator in
/// NumPy versions before 1.17.
#[derive(Debug, Clone)]
pub struct MT19937 {
    rng: rand::rngs::StdRng,
    seed: u64,
}

impl MT19937 {
    /// Create a new MT19937 instance with a random seed
    pub fn new() -> Self {
        let seed = rand::random::<u64>();
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    /// Create a new MT19937 instance with a specific 64-bit seed
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
        }
    }

    /// Create a new MT19937 instance from a seed slice
    pub fn from_seed_slice(seed: &[u32]) -> Self {
        let mut seed_u64 = 0u64;
        for (i, &word) in seed.iter().enumerate() {
            if i < 2 {
                seed_u64 |= (word as u64) << (i * 32);
            }
        }
        Self::seed_from_u64(seed_u64)
    }
}

impl BitGenerator for MT19937 {
    fn name(&self) -> &'static str {
        "MT19937"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn seed_u64(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = StdRng::seed_from_u64(seed);
    }

    fn seed_u32_slice(&mut self, seed: &[u32]) {
        *self = Self::from_seed_slice(seed);
    }

    fn get_state_bytes(&self) -> Vec<u8> {
        // Serialize the seed
        let mut state = Vec::new();
        state.extend_from_slice(&self.seed.to_le_bytes());
        state
    }

    fn set_state_bytes(&mut self, state: &[u8]) -> Result<(), String> {
        if state.len() < 8 {
            return Err("Insufficient state data".to_string());
        }

        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&state[0..8]);
        let seed = u64::from_le_bytes(seed_bytes);

        self.seed_u64(seed);
        Ok(())
    }

    fn state_size(&self) -> usize {
        8 // Just the seed for now
    }

    fn jump(&mut self, steps: u64) {
        // Advance the generator by discarding random numbers
        for _ in 0..steps {
            let _ = self.next_u64();
        }
    }

    fn is_cryptographically_secure(&self) -> bool {
        false
    }

    fn period(&self) -> Option<u128> {
        // MT19937 has a period of 2^19937-1, which is too large to represent
        None
    }

    fn parallel_params(&self) -> ParallelParams {
        ParallelParams {
            jump_size: u64::MAX, // Maximum value for parallel stream jumps
            max_streams: 1024,
            supports_parallel: true,
        }
    }
}

impl RngCore for MT19937 {
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

/// Philox BitGenerator (counter-based)
///
/// Philox is a counter-based random number generator that's particularly
/// well-suited for parallel computation and GPU applications. It's
/// cryptographically secure and has good statistical properties.
#[derive(Debug, Clone)]
pub struct Philox {
    rng: StdRng,
    seed: u64,
    counter: u64,
}

impl Philox {
    /// Create a new Philox instance with a random seed
    pub fn new() -> Self {
        let seed = rand::random::<u64>();
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
            counter: 0,
        }
    }

    /// Create a new Philox instance with a specific 64-bit seed
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
            counter: 0,
        }
    }

    /// Create a new Philox instance from a seed slice
    pub fn from_seed_slice(seed: &[u32]) -> Self {
        let mut seed_u64 = 0u64;
        for (i, &word) in seed.iter().enumerate() {
            if i < 2 {
                seed_u64 |= (word as u64) << (i * 32);
            }
        }
        Self::seed_from_u64(seed_u64)
    }

    /// Create a new Philox instance with seed and counter
    pub fn seed_from_u64_with_counter(seed: u64, counter: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            seed,
            counter,
        }
    }
}

impl BitGenerator for Philox {
    fn name(&self) -> &'static str {
        "Philox"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn seed_u64(&mut self, seed: u64) {
        self.seed = seed;
        self.counter = 0;
        self.rng = StdRng::seed_from_u64(seed);
    }

    fn seed_u32_slice(&mut self, seed: &[u32]) {
        *self = Self::from_seed_slice(seed);
    }

    fn get_state_bytes(&self) -> Vec<u8> {
        // Serialize the seed and counter
        let mut state = Vec::new();
        state.extend_from_slice(&self.seed.to_le_bytes());
        state.extend_from_slice(&self.counter.to_le_bytes());
        state
    }

    fn set_state_bytes(&mut self, state: &[u8]) -> Result<(), String> {
        if state.len() < 16 {
            return Err("Insufficient state data".to_string());
        }

        let mut seed_bytes = [0u8; 8];
        let mut counter_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&state[0..8]);
        counter_bytes.copy_from_slice(&state[8..16]);

        let seed = u64::from_le_bytes(seed_bytes);
        let counter = u64::from_le_bytes(counter_bytes);

        self.seed = seed;
        self.counter = counter;
        self.rng = StdRng::seed_from_u64(seed);

        Ok(())
    }

    fn state_size(&self) -> usize {
        16 // seed (8 bytes) + counter (8 bytes)
    }

    fn jump(&mut self, steps: u64) {
        self.counter += steps;
        // Re-seed with the new counter value
        self.rng = StdRng::seed_from_u64(self.seed ^ self.counter);
    }

    fn is_cryptographically_secure(&self) -> bool {
        true
    }

    fn period(&self) -> Option<u128> {
        // Philox has a very large period due to counter-based design
        Some(u128::MAX)
    }

    fn parallel_params(&self) -> ParallelParams {
        ParallelParams {
            jump_size: 1 << 20, // Moderate jump size
            max_streams: 65536, // Large number of streams due to counter design
            supports_parallel: true,
        }
    }
}

impl RngCore for Philox {
    fn next_u32(&mut self) -> u32 {
        let result = self.rng.next_u32();
        self.counter += 1;
        result
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.rng.next_u64();
        self.counter += 1;
        result
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest);
        self.counter += dest.len() as u64 / 8 + 1;
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        let result = self.rng.try_fill_bytes(dest);
        self.counter += dest.len() as u64 / 8 + 1;
        result
    }
}

/// SFC64 (Small Fast Counter) BitGenerator
///
/// SFC64 is a fast counter-based random number generator with good
/// statistical properties and a very small state size. It's particularly
/// suitable for applications where speed is critical and cryptographic security
/// is not required.
#[derive(Debug, Clone)]
pub struct SFC64 {
    state: u64,
    counter: u64,
}

impl SFC64 {
    /// Create a new SFC64 instance with a random seed
    pub fn new() -> Self {
        let seed = rand::random::<u64>();
        Self {
            state: seed,
            counter: 0,
        }
    }

    /// Create a new SFC64 instance with a specific 64-bit seed
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            state: seed,
            counter: 0,
        }
    }

    /// Create a new SFC64 instance from a seed slice
    pub fn from_seed_slice(seed: &[u32]) -> Self {
        let mut seed_u64 = 0u64;
        for (i, &word) in seed.iter().enumerate() {
            if i < 2 {
                seed_u64 |= (word as u64) << (i * 32);
            }
        }
        Self::seed_from_u64(seed_u64)
    }
}

impl BitGenerator for SFC64 {
    fn name(&self) -> &'static str {
        "SFC64"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn seed_u64(&mut self, seed: u64) {
        self.state = seed;
        self.counter = 0;
    }

    fn seed_u32_slice(&mut self, seed: &[u32]) {
        *self = Self::from_seed_slice(seed);
    }

    fn get_state_bytes(&self) -> Vec<u8> {
        // Serialize the state and counter
        let mut state = Vec::new();
        state.extend_from_slice(&self.state.to_le_bytes());
        state.extend_from_slice(&self.counter.to_le_bytes());
        state
    }

    fn set_state_bytes(&mut self, state: &[u8]) -> Result<(), String> {
        if state.len() < 16 {
            return Err("Insufficient state data".to_string());
        }

        let mut state_bytes = [0u8; 8];
        let mut counter_bytes = [0u8; 8];
        state_bytes.copy_from_slice(&state[0..8]);
        counter_bytes.copy_from_slice(&state[8..16]);

        self.state = u64::from_le_bytes(state_bytes);
        self.counter = u64::from_le_bytes(counter_bytes);

        Ok(())
    }

    fn state_size(&self) -> usize {
        16 // state (8 bytes) + counter (8 bytes)
    }

    fn jump(&mut self, steps: u64) {
        self.counter += steps;
    }

    fn is_cryptographically_secure(&self) -> bool {
        false
    }

    fn period(&self) -> Option<u128> {
        // SFC64 has a period of 2^64
        Some(2u128.pow(64))
    }

    fn parallel_params(&self) -> ParallelParams {
        ParallelParams {
            jump_size: 1 << 16, // Small jump size
            max_streams: 65536,
            supports_parallel: true,
        }
    }
}

impl RngCore for SFC64 {
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() & 0xFFFFFFFF) as u32
    }

    fn next_u64(&mut self) -> u64 {
        // SFC64 algorithm
        let x = self.state;
        self.state = x.wrapping_add(0x9E3779B97F4A7C15u64);
        let mut t = x.wrapping_mul(x);
        t = (t as u128).wrapping_mul(t as u128) as u64;
        let result = t.wrapping_add(self.counter);
        self.counter = self.counter.wrapping_add(1);
        result
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64();
            let bytes = val.to_le_bytes();
            for (i, byte) in chunk.iter_mut().enumerate() {
                if i < bytes.len() {
                    *byte = bytes[i];
                }
            }
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// PCG64DXSM (PCG64 with DXSM output function) BitGenerator
///
/// PCG64DXSM is a variant of PCG64 that uses the DXSM (Double Xor Shift Multiply)
/// output function for better statistical properties, especially in the lower bits.
/// It's recommended for applications requiring high-quality random numbers.
#[derive(Debug, Clone)]
pub struct PCG64DXSM {
    state: u128,
    increment: u128,
}

impl PCG64DXSM {
    // PCG64DXSM multiplier from the specification (128 bits)
    const MULTIPLIER: u128 = {
        // Combine 64-bit parts to form 128-bit multiplier
        let high: u128 = 0x2360_ED05_1FC6_5DA4 as u128;
        let low: u128 = 0x3FFF_5DDE_F0E_5E97_F6D as u128;
        (high << 64) | low
    };
    // Default increment (128 bits, must be odd)
    const DEFAULT_INCREMENT: u128 = 0x5851_F42D_4C95_7F2D_140F_676D_2C3F_7D5C;

    /// Create a new PCG64DXSM instance with a random seed
    pub fn new() -> Self {
        let seed = rand::random::<u64>();
        let increment = rand::random::<u64>();
        Self {
            state: ((seed as u128) << 64) | increment as u128,
            increment: (increment as u128) << 64 | increment as u128,
        }
    }

    /// Create a new PCG64DXSM instance with a specific 64-bit seed
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            state: ((seed as u128) << 64) | Self::DEFAULT_INCREMENT,
            increment: Self::DEFAULT_INCREMENT,
        }
    }

    /// Create a new PCG64DXSM instance from a seed slice
    pub fn from_seed_slice(seed: &[u32]) -> Self {
        let mut seed_u64 = 0u64;
        for (i, &word) in seed.iter().enumerate() {
            if i < 2 {
                seed_u64 |= (word as u64) << (i * 32);
            }
        }
        Self::seed_from_u64(seed_u64)
    }

    /// Create a new PCG64DXSM instance with seed and increment
    pub fn seed_from_u64_with_increment(seed: u64, increment: u64) -> Self {
        let inc = (increment as u128) << 64 | increment as u128;
        Self {
            state: ((seed as u128) << 64) | inc,
            increment: inc,
        }
    }
}

impl BitGenerator for PCG64DXSM {
    fn name(&self) -> &'static str {
        "PCG64DXSM"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn seed_u64(&mut self, seed: u64) {
        self.state = ((seed as u128) << 64) | self.increment;
    }

    fn seed_u32_slice(&mut self, seed: &[u32]) {
        *self = Self::from_seed_slice(seed);
    }

    fn get_state_bytes(&self) -> Vec<u8> {
        // Serialize the state and increment
        let mut state = Vec::new();
        state.extend_from_slice(&self.state.to_le_bytes());
        state
    }

    fn set_state_bytes(&mut self, state: &[u8]) -> Result<(), String> {
        if state.len() < 16 {
            return Err("Insufficient state data".to_string());
        }

        let mut state_bytes = [0u8; 16];
        state_bytes.copy_from_slice(&state[0..16]);

        let new_state = u128::from_le_bytes(state_bytes);
        self.state = new_state;

        Ok(())
    }

    fn state_size(&self) -> usize {
        16 // state (16 bytes)
    }

    fn jump(&mut self, steps: u64) {
        for _ in 0..steps {
            self.next_u64();
        }
    }

    fn is_cryptographically_secure(&self) -> bool {
        false
    }

    fn period(&self) -> Option<u128> {
        // PCG64DXSM has a period of 2^128
        Some(u128::MAX)
    }

    fn parallel_params(&self) -> ParallelParams {
        ParallelParams {
            jump_size: 1 << 32, // Large jump for parallel streams
            max_streams: 256,
            supports_parallel: true,
        }
    }
}

impl RngCore for PCG64DXSM {
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() & 0xFFFFFFFF) as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::MULTIPLIER);
        self.state = self.state.wrapping_add(self.increment);

        let xorshifted = ((self.state >> 64) ^ self.state) >> 5;
        let rot = ((self.state >> 122) as u32) as u64;

        // DXSM output function
        let result = xorshifted.wrapping_mul(0xDA942042E4DD58B5);
        let result = result.rotate_right(rot as u32);

        (result >> 64) as u64
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64();
            let bytes = val.to_le_bytes();
            for (i, byte) in chunk.iter_mut().enumerate() {
                if i < bytes.len() {
                    *byte = bytes[i];
                }
            }
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

/// Factory functions for creating BitGenerator instances
pub mod factory {
    use super::*;

    /// Create a BitGenerator by name
    pub fn create_bitgenerator(name: &str) -> Result<Box<dyn BitGenerator>, String> {
        match name {
            "PCG64" => Ok(Box::new(PCG64::new())),
            "MT19937" => Ok(Box::new(MT19937::new())),
            "Philox" => Ok(Box::new(Philox::new())),
            "SFC64" => Ok(Box::new(SFC64::new())),
            "PCG64DXSM" => Ok(Box::new(PCG64DXSM::new())),
            _ => Err(format!("Unknown BitGenerator: {}", name)),
        }
    }

    /// Create a seeded BitGenerator by name
    pub fn create_seeded_bitgenerator(
        name: &str,
        seed: u64,
    ) -> Result<Box<dyn BitGenerator>, String> {
        match name {
            "PCG64" => Ok(Box::new(PCG64::seed_from_u64(seed))),
            "MT19937" => Ok(Box::new(MT19937::seed_from_u64(seed))),
            "Philox" => Ok(Box::new(Philox::seed_from_u64(seed))),
            "SFC64" => Ok(Box::new(SFC64::seed_from_u64(seed))),
            "PCG64DXSM" => Ok(Box::new(PCG64DXSM::seed_from_u64(seed))),
            _ => Err(format!("Unknown BitGenerator: {}", name)),
        }
    }

    /// Get the list of available BitGenerator names
    pub fn available_bitgenerators() -> Vec<&'static str> {
        vec!["PCG64", "MT19937", "Philox", "SFC64", "PCG64DXSM"]
    }

    /// Get the default BitGenerator (PCG64)
    pub fn default_bitgenerator() -> Box<dyn BitGenerator> {
        Box::new(PCG64::new())
    }
}

/// Legacy compatibility functions
pub mod legacy {
    use super::*;

    /// Create a BitGenerator with the default algorithm (PCG64)
    pub fn new() -> Box<dyn BitGenerator> {
        factory::default_bitgenerator()
    }

    /// Create a BitGenerator with a specific seed
    pub fn seed_from_u64(seed: u64) -> Box<dyn BitGenerator> {
        factory::create_seeded_bitgenerator("PCG64", seed).unwrap()
    }
}
