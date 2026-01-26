//! Examples demonstrating the BitGenerator trait
//!
//! This file shows how to use the BitGenerator trait directly for low-level
//! random number generation.

use rust_numpy::random::bit_generator::PCG64;
use rand::RngCore;

fn main() {
    println!("=== BitGenerator Examples ===\n");

    // Example 1: Basic PCG64 usage
    println!("--- Basic PCG64 Usage ---");
    let mut bit_gen = PCG64::new();

    println!("next_u32(): {}", bit_gen.next_u32());
    println!("next_u64(): {}", bit_gen.next_u64());

    let mut bytes = [0u8; 16];
    bit_gen.fill_bytes(&mut bytes);
    println!("fill_bytes(): {:?}", bytes);

    // Example 2: Seeded PCG64
    println!("\n--- Seeded PCG64 ---");
    let seed = 42;
    let mut bit_gen1 = PCG64::seed_from_u64(seed);
    let mut bit_gen2 = PCG64::seed_from_u64(seed);

    println!("Generator 1 next_u64(): {}", bit_gen1.next_u64());
    println!("Generator 2 next_u64(): {}", bit_gen2.next_u64());
    println!("Seeded generators produce identical sequences");

    // Example 3: Generate multiple values
    println!("\n--- Multiple Values ---");
    let mut bit_gen = PCG64::seed_from_u64(12345);
    print!("10 random u64 values: ");
    for _ in 0..10 {
        print!("{}, ", bit_gen.next_u64());
    }
    println!();

    // Example 4: Generate random bytes
    println!("\n--- Random Bytes ---");
    let mut bit_gen = PCG64::new();
    let mut buffer = [0u8; 32];
    bit_gen.fill_bytes(&mut buffer);
    println!("32 random bytes: {:?}", buffer);

    println!("\n=== Examples Complete ===");
}
