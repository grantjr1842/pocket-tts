// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! SeedSequence for spawnable random number streams
//!
//! This module provides a SeedSequence implementation that matches NumPy's
//! SeedSequence API for generating seeds for BitGenerators and creating
//! independent child streams for parallel random number generation.

use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// SeedSequence for reproducible random streams
///
/// SeedSequence provides a way to generate seeds for BitGenerators that supports
/// spawning child sequences for parallel random number generation. It ensures
/// that spawned sequences are independent and produce different random streams.
///
/// This implementation is compatible with NumPy's SeedSequence and provides
/// similar functionality for generating seeds and spawn keys.
///
/// # Examples
///
/// ```
/// use rust_numpy::random::SeedSequence;
///
/// // Create a SeedSequence with entropy
/// let seq = SeedSequence::new(Some(12345), None, None);
///
/// // Generate a 64-bit seed
/// let seed = seq.generate_seed_u64();
///
/// // Spawn a child sequence
/// let child = seq.spawn(0);
/// ```
#[derive(Clone)]
pub struct SeedSequence {
    /// The entropy (base seed) for this sequence
    entropy: u128,
    /// The spawn key that identifies this sequence in the spawn tree
    spawn_key: Vec<u64>,
    /// The number of bits in each generated integer
    n_bits: usize,
}

impl SeedSequence {
    /// Create a new SeedSequence
    ///
    /// # Arguments
    ///
    /// * `entropy` - Optional initial entropy/seed value. If None, a random value is used.
    /// * `spawn_key` - Optional spawn key for creating child sequences. If None, an empty key is used.
    /// * `n_bits` - Optional number of bits per generated integer (default: 64).
    ///
    /// # Returns
    ///
    /// A new SeedSequence instance
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_numpy::random::SeedSequence;
    ///
    /// // With entropy
    /// let seq1 = SeedSequence::new(Some(12345), None, None);
    ///
    /// // With spawn key
    /// let seq2 = SeedSequence::new(None, Some(vec![1, 2, 3]), None);
    ///
    /// // With all parameters
    /// let seq3 = SeedSequence::new(Some(42), Some(vec![0]), Some(32));
    /// ```
    pub fn new(entropy: Option<u64>, spawn_key: Option<Vec<u64>>, n_bits: Option<usize>) -> Self {
        let entropy_val = match entropy {
            Some(e) => e as u128,
            None => {
                // Use current time and thread ID to generate random entropy
                let time = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                let tid = std::thread::current().id();
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                tid.hash(&mut hasher);
                let tid_hash = hasher.finish();
                time ^ (tid_hash as u128)
            }
        };

        Self {
            entropy: entropy_val,
            spawn_key: spawn_key.unwrap_or_default(),
            n_bits: n_bits.unwrap_or(64),
        }
    }

    /// Create a SeedSequence from a 128-bit entropy value
    ///
    /// # Arguments
    ///
    /// * `entropy` - 128-bit entropy value
    /// * `spawn_key` - Optional spawn key
    ///
    /// # Returns
    ///
    /// A new SeedSequence instance
    pub fn from_entropy(entropy: u128, spawn_key: Option<Vec<u64>>) -> Self {
        Self {
            entropy,
            spawn_key: spawn_key.unwrap_or_default(),
            n_bits: 64,
        }
    }

    /// Generate a spawn key for a child sequence
    ///
    /// This generates a unique spawn key that can be used to create
    /// a child SeedSequence with an independent random stream.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the child sequence
    ///
    /// # Returns
    ///
    /// A vector of u64 values representing the spawn key
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_numpy::random::SeedSequence;
    ///
    /// let seq = SeedSequence::new(Some(12345), None, None);
    ///
    /// // Generate spawn keys for multiple child sequences
    /// let key1 = seq.spawn_key(0);
    /// let key2 = seq.spawn_key(1);
    /// let key3 = seq.spawn_key(2);
    ///
    /// assert_ne!(key1, key2);
    /// ```
    pub fn spawn_key(&self, index: u64) -> Vec<u64> {
        let mut key = self.spawn_key.clone();
        key.push(index);
        key
    }

    /// Spawn a child SeedSequence
    ///
    /// Creates a new SeedSequence that is independent of the parent
    /// but derived from it. Child sequences produce different random
    /// streams and are suitable for parallel computation.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the child sequence
    ///
    /// # Returns
    ///
    /// A new SeedSequence instance that is a child of this sequence
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_numpy::random::SeedSequence;
    ///
    /// let parent = SeedSequence::new(Some(12345), None, None);
    ///
    /// // Spawn multiple child sequences
    /// let child1 = parent.spawn(0);
    /// let child2 = parent.spawn(1);
    ///
    /// // Child sequences produce different seeds
    /// assert_ne!(child1.generate_seed_u64(), child2.generate_seed_u64());
    /// ```
    pub fn spawn(&self, index: u64) -> Self {
        let mut child_spawn_key = self.spawn_key.clone();
        child_spawn_key.push(index);

        // Mix entropy with spawn information to create unique child entropy
        let child_entropy = self.entropy.wrapping_add(mix_spawn_key(&child_spawn_key));

        Self {
            entropy: child_entropy,
            spawn_key: child_spawn_key,
            n_bits: self.n_bits,
        }
    }

    /// Generate a 64-bit seed
    ///
    /// Generates a seed value suitable for seeding a BitGenerator.
    /// The seed is derived from the entropy and spawn key.
    ///
    /// # Returns
    ///
    /// A 64-bit seed value
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_numpy::random::SeedSequence;
    ///
    /// let seq = SeedSequence::new(Some(12345), None, None);
    /// let seed = seq.generate_seed_u64();
    /// ```
    pub fn generate_seed_u64(&self) -> u64 {
        // Mix entropy with spawn key to generate a seed
        let mixed = self.entropy.wrapping_add(mix_spawn_key(&self.spawn_key));
        mixed as u64 ^ (mixed >> 64) as u64
    }

    /// Generate a 32-bit seed
    ///
    /// # Returns
    ///
    /// A 32-bit seed value
    pub fn generate_seed_u32(&self) -> u32 {
        self.generate_seed_u64() as u32
    }

    /// Generate a seed as a vector of u32 values
    ///
    /// # Arguments
    ///
    /// * `n_words` - Number of 32-bit words to generate
    ///
    /// # Returns
    ///
    /// A vector of u32 values
    pub fn generate_seed_u32_slice(&self, n_words: usize) -> Vec<u32> {
        let mut seeds = Vec::with_capacity(n_words);
        let base_seed = self.generate_seed_u64();

        for i in 0..n_words {
            // Generate unique values by mixing with index
            let mixed = base_seed.wrapping_add(i as u64 * 0x9e3779b97f4a7c15);
            seeds.push((mixed ^ (mixed >> 32)) as u32);
        }

        seeds
    }

    /// Get the entropy value
    ///
    /// # Returns
    ///
    /// The 128-bit entropy value
    pub fn entropy(&self) -> u128 {
        self.entropy
    }

    /// Get the spawn key
    ///
    /// # Returns
    ///
    /// A slice of the spawn key
    pub fn spawn_key_as_slice(&self) -> &[u64] {
        &self.spawn_key
    }

    /// Get the number of bits per generated integer
    ///
    /// # Returns
    ///
    /// The number of bits (32 or 64)
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    /// Spawn multiple child sequences at once
    ///
    /// # Arguments
    ///
    /// * `n_children` - Number of child sequences to spawn
    ///
    /// # Returns
    ///
    /// A vector of SeedSequence instances
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_numpy::random::SeedSequence;
    ///
    /// let parent = SeedSequence::new(Some(12345), None, None);
    ///
    /// // Spawn 4 child sequences for parallel computation
    /// let children = parent.spawn_n(4);
    /// assert_eq!(children.len(), 4);
    /// ```
    pub fn spawn_n(&self, n_children: usize) -> Vec<Self> {
        (0..n_children).map(|i| self.spawn(i as u64)).collect()
    }
}

impl Default for SeedSequence {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

impl Debug for SeedSequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SeedSequence")
            .field("entropy", &format!("{:#034x}", self.entropy))
            .field("spawn_key", &self.spawn_key)
            .field("n_bits", &self.n_bits)
            .finish()
    }
}

impl PartialEq for SeedSequence {
    fn eq(&self, other: &Self) -> bool {
        self.entropy == other.entropy
            && self.spawn_key == other.spawn_key
            && self.n_bits == other.n_bits
    }
}

/// Mix a spawn key into a hash value
///
/// This function mixes the spawn key values using a hash function
/// to create a unique value for the given spawn key.
fn mix_spawn_key(spawn_key: &[u64]) -> u128 {
    // Use a simple but effective mixing function
    // Based on splitmix64 and MurmurHash ideas
    let mut h: u128 = 0x6a09e667f3bcc908; // Golden ratio constant

    for (i, &k) in spawn_key.iter().enumerate() {
        // Mix in each key component
        let mut k = k as u128;
        k = k.wrapping_mul(0xbf58476d1ce4e5b9);
        k ^= k >> 27;
        k = k.wrapping_mul(0x94d049bb133111eb);

        h ^= k.wrapping_add(i as u128);
        h = h.wrapping_mul(0x9e3779b97f4a7c15); // Golden ratio
        h ^= h >> 33;
    }

    h
}

/// Create a SeedSequence for a BitGenerator
///
/// Convenience function to create a SeedSequence with the given entropy.
///
/// # Arguments
///
/// * `entropy` - Optional entropy value. If None, random entropy is used.
///
/// # Returns
///
/// A new SeedSequence instance
pub fn seed_sequence(entropy: Option<u64>) -> SeedSequence {
    SeedSequence::new(entropy, None, None)
}

/// Spawn multiple independent SeedSequences from a parent
///
/// This is useful for parallel random number generation where each
/// thread or process needs its own independent random stream.
///
/// # Arguments
///
/// * `parent_entropy` - Optional parent entropy value
/// * `n_children` - Number of child sequences to spawn
///
/// # Returns
///
/// A vector of independent SeedSequence instances
///
/// # Examples
///
/// ```
/// use rust_numpy::random::spawn_sequences;
///
/// // Create 4 independent seed sequences for parallel computation
/// let sequences = spawn_sequences(Some(12345), 4);
/// assert_eq!(sequences.len(), 4);
///
/// // Each sequence generates different seeds
/// let seeds: Vec<u64> = sequences.iter().map(|s| s.generate_seed_u64()).collect();
/// assert_ne!(seeds[0], seeds[1]);
/// ```
pub fn spawn_sequences(parent_entropy: Option<u64>, n_children: usize) -> Vec<SeedSequence> {
    let parent = SeedSequence::new(parent_entropy, None, None);
    parent.spawn_n(n_children)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_sequence_new() {
        let seq = SeedSequence::new(Some(12345), None, None);
        assert_eq!(seq.entropy(), 12345);
        assert!(seq.spawn_key_as_slice().is_empty());
        assert_eq!(seq.n_bits(), 64);
    }

    #[test]
    fn test_seed_sequence_default() {
        let seq = SeedSequence::default();
        assert_eq!(seq.n_bits(), 64);
        // Default should use random entropy
    }

    #[test]
    fn test_generate_seed_u64() {
        let seq1 = SeedSequence::new(Some(12345), None, None);
        let seq2 = SeedSequence::new(Some(12345), None, None);

        // Same entropy should produce the same seed
        assert_eq!(seq1.generate_seed_u64(), seq2.generate_seed_u64());

        // Different entropy should produce different seeds (with high probability)
        let seq3 = SeedSequence::new(Some(54321), None, None);
        assert_ne!(seq1.generate_seed_u64(), seq3.generate_seed_u64());
    }

    #[test]
    fn test_spawn_key() {
        let seq = SeedSequence::new(Some(12345), None, None);

        let key0 = seq.spawn_key(0);
        let key1 = seq.spawn_key(1);

        assert_eq!(key0, vec![0]);
        assert_eq!(key1, vec![1]);
    }

    #[test]
    fn test_spawn_with_existing_key() {
        let seq = SeedSequence::new(Some(12345), Some(vec![1, 2]), None);

        let key = seq.spawn_key(3);
        assert_eq!(key, vec![1, 2, 3]);
    }

    #[test]
    fn test_spawn() {
        let parent = SeedSequence::new(Some(12345), None, None);
        let child = parent.spawn(0);

        // Child should have different spawn key
        assert_eq!(child.spawn_key_as_slice(), &[0]);

        // Child should produce different seed than parent
        assert_ne!(parent.generate_seed_u64(), child.generate_seed_u64());
    }

    #[test]
    fn test_spawn_n() {
        let parent = SeedSequence::new(Some(12345), None, None);
        let children = parent.spawn_n(4);

        assert_eq!(children.len(), 4);

        // Each child should have a unique spawn key
        for (i, child) in children.iter().enumerate() {
            assert_eq!(child.spawn_key_as_slice(), &[i as u64]);
        }

        // All children should produce different seeds
        let seeds: Vec<u64> = children.iter().map(|c| c.generate_seed_u64()).collect();
        let unique_seeds: std::collections::HashSet<_> = seeds.iter().cloned().collect();
        assert_eq!(unique_seeds.len(), seeds.len());
    }

    #[test]
    fn test_generate_seed_u32_slice() {
        let seq = SeedSequence::new(Some(12345), None, None);

        let seeds = seq.generate_seed_u32_slice(4);
        assert_eq!(seeds.len(), 4);

        // All values should be unique (with high probability)
        let unique: std::collections::HashSet<_> = seeds.iter().cloned().collect();
        assert_eq!(unique.len(), seeds.len());
    }

    #[test]
    fn test_spawn_sequences() {
        let sequences = spawn_sequences(Some(99999), 8);
        assert_eq!(sequences.len(), 8);

        // All sequences should be independent
        let seeds: Vec<u64> = sequences.iter().map(|s| s.generate_seed_u64()).collect();
        let unique_seeds: std::collections::HashSet<_> = seeds.iter().cloned().collect();
        assert_eq!(unique_seeds.len(), seeds.len());
    }

    #[test]
    fn test_seed_sequence_equality() {
        let seq1 = SeedSequence::new(Some(12345), Some(vec![1, 2]), None);
        let seq2 = SeedSequence::new(Some(12345), Some(vec![1, 2]), None);
        let seq3 = SeedSequence::new(Some(54321), Some(vec![1, 2]), None);

        assert_eq!(seq1, seq2);
        assert_ne!(seq1, seq3);
    }

    #[test]
    fn test_debug_format() {
        let seq = SeedSequence::new(Some(12345), None, None);
        let debug_str = format!("{:?}", seq);
        assert!(debug_str.contains("SeedSequence"));
        assert!(debug_str.contains("entropy"));
        assert!(debug_str.contains("spawn_key"));
    }

    #[test]
    fn test_from_entropy() {
        let seq = SeedSequence::from_entropy(0x123456789abcdef0u128, Some(vec![1, 2]));
        assert_eq!(seq.entropy(), 0x123456789abcdef0u128);
        assert_eq!(seq.spawn_key_as_slice(), &[1, 2]);
    }

    #[test]
    fn test_n_bits_option() {
        let seq32 = SeedSequence::new(Some(12345), None, Some(32));
        assert_eq!(seq32.n_bits(), 32);

        let seq64 = SeedSequence::new(Some(12345), None, Some(64));
        assert_eq!(seq64.n_bits(), 64);
    }

    #[test]
    fn test_nested_spawn() {
        // Test spawning from a spawned sequence
        let grandparent = SeedSequence::new(Some(12345), None, None);
        let parent = grandparent.spawn(0);
        let child = parent.spawn(1);

        assert_eq!(grandparent.spawn_key_as_slice(), &[]);
        assert_eq!(parent.spawn_key_as_slice(), &[0]);
        assert_eq!(child.spawn_key_as_slice(), &[0, 1]);

        // All should have different seeds
        let seeds = vec![
            grandparent.generate_seed_u64(),
            parent.generate_seed_u64(),
            child.generate_seed_u64(),
        ];
        let unique: std::collections::HashSet<_> = seeds.iter().cloned().collect();
        assert_eq!(unique.len(), 3);
    }
}
