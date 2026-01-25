/// Base trait for bit-width type annotations.
pub trait NBitBase {}

/// 8-bit width type annotation.
pub struct nbit_8;
impl NBitBase for nbit_8 {}

/// 16-bit width type annotation.
pub struct nbit_16;
impl NBitBase for nbit_16 {}

/// 32-bit width type annotation.
pub struct nbit_32;
impl NBitBase for nbit_32 {}

/// 64-bit width type annotation.
pub struct nbit_64;
impl NBitBase for nbit_64 {}

/// 128-bit width type annotation.
pub struct nbit_128;
impl NBitBase for nbit_128 {}

/// 256-bit width type annotation.
pub struct nbit_256;
impl NBitBase for nbit_256 {}

// Bit-width type aliases
pub type Int8 = nbit_8;
pub type Int16 = nbit_16;
pub type Int32 = nbit_32;
pub type Int64 = nbit_64;

pub type UInt8 = nbit_8;
pub type UInt16 = nbit_16;
pub type UInt32 = nbit_32;
pub type UInt64 = nbit_64;

pub type Float16 = nbit_16;
pub type Float32 = nbit_32;
pub type Float64 = nbit_64;
pub type Float128 = nbit_128;

pub type Complex32 = nbit_32;
pub type Complex64 = nbit_64;
pub type Complex128 = nbit_128;
pub type Complex256 = nbit_256;
