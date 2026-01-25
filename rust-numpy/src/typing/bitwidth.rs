/// Base trait for bit-width type annotations.
/// This provides a hierarchy for bit-width types similar to NumPy's NBitBase.
pub trait NBitBase: 'static {
    /// The bit width of this type
    const BITS: usize;
    /// Whether this type is signed
    const SIGNED: bool;
    /// Whether this type is a floating point number
    const FLOAT: bool;
    /// Whether this type is complex
    const COMPLEX: bool;
}

/// Marker trait for signed integer types
pub trait SignedInt: NBitBase {}

/// Marker trait for unsigned integer types
pub trait UnsignedInt: NBitBase {}

/// Marker trait for floating point types
pub trait FloatType: NBitBase {}

/// Marker trait for complex types
pub trait ComplexType: NBitBase {}

// 8-bit types
/// 8-bit signed integer type annotation
pub struct Int8Bit;
impl NBitBase for Int8Bit {
    const BITS: usize = 8;
    const SIGNED: bool = true;
    const FLOAT: bool = false;
    const COMPLEX: bool = false;
}
impl SignedInt for Int8Bit {}

/// 8-bit unsigned integer type annotation
pub struct UInt8Bit;
impl NBitBase for UInt8Bit {
    const BITS: usize = 8;
    const SIGNED: bool = false;
    const FLOAT: bool = false;
    const COMPLEX: bool = false;
}
impl UnsignedInt for UInt8Bit {}

// 16-bit types
/// 16-bit signed integer type annotation
pub struct Int16Bit;
impl NBitBase for Int16Bit {
    const BITS: usize = 16;
    const SIGNED: bool = true;
    const FLOAT: bool = false;
    const COMPLEX: bool = false;
}
impl SignedInt for Int16Bit {}

/// 16-bit unsigned integer type annotation
pub struct UInt16Bit;
impl NBitBase for UInt16Bit {
    const BITS: usize = 16;
    const SIGNED: bool = false;
    const FLOAT: bool = false;
    const COMPLEX: bool = false;
}
impl UnsignedInt for UInt16Bit {}

/// 16-bit floating point type annotation
pub struct Float16Bit;
impl NBitBase for Float16Bit {
    const BITS: usize = 16;
    const SIGNED: bool = true;
    const FLOAT: bool = true;
    const COMPLEX: bool = false;
}
impl FloatType for Float16Bit {}

// 32-bit types
/// 32-bit signed integer type annotation
pub struct Int32Bit;
impl NBitBase for Int32Bit {
    const BITS: usize = 32;
    const SIGNED: bool = true;
    const FLOAT: bool = false;
    const COMPLEX: bool = false;
}
impl SignedInt for Int32Bit {}

/// 32-bit unsigned integer type annotation
pub struct UInt32Bit;
impl NBitBase for UInt32Bit {
    const BITS: usize = 32;
    const SIGNED: bool = false;
    const FLOAT: bool = false;
    const COMPLEX: bool = false;
}
impl UnsignedInt for UInt32Bit {}

/// 32-bit floating point type annotation
pub struct Float32Bit;
impl NBitBase for Float32Bit {
    const BITS: usize = 32;
    const SIGNED: bool = true;
    const FLOAT: bool = true;
    const COMPLEX: bool = false;
}
impl FloatType for Float32Bit {}

/// 32-bit complex type annotation (2x 16-bit float)
pub struct Complex32Bit;
impl NBitBase for Complex32Bit {
    const BITS: usize = 32;
    const SIGNED: bool = true;
    const FLOAT: bool = false;
    const COMPLEX: bool = true;
}
impl ComplexType for Complex32Bit {}

// 64-bit types
/// 64-bit signed integer type annotation
pub struct Int64Bit;
impl NBitBase for Int64Bit {
    const BITS: usize = 64;
    const SIGNED: bool = true;
    const FLOAT: bool = false;
    const COMPLEX: bool = false;
}
impl SignedInt for Int64Bit {}

/// 64-bit unsigned integer type annotation
pub struct UInt64Bit;
impl NBitBase for UInt64Bit {
    const BITS: usize = 64;
    const SIGNED: bool = false;
    const FLOAT: bool = false;
    const COMPLEX: bool = false;
}
impl UnsignedInt for UInt64Bit {}

/// 64-bit floating point type annotation
pub struct Float64Bit;
impl NBitBase for Float64Bit {
    const BITS: usize = 64;
    const SIGNED: bool = true;
    const FLOAT: bool = true;
    const COMPLEX: bool = false;
}
impl FloatType for Float64Bit {}

/// 64-bit complex type annotation (2x 32-bit float)
pub struct Complex64Bit;
impl NBitBase for Complex64Bit {
    const BITS: usize = 64;
    const SIGNED: bool = true;
    const FLOAT: bool = false;
    const COMPLEX: bool = true;
}
impl ComplexType for Complex64Bit {}

// 128-bit types
/// 128-bit complex type annotation (2x 64-bit float)
pub struct Complex128Bit;
impl NBitBase for Complex128Bit {
    const BITS: usize = 128;
    const SIGNED: bool = true;
    const FLOAT: bool = false;
    const COMPLEX: bool = true;
}
impl ComplexType for Complex128Bit {}

// 256-bit types
/// 256-bit complex type annotation (2x 128-bit float)
pub struct Complex256Bit;
impl NBitBase for Complex256Bit {
    const BITS: usize = 256;
    const SIGNED: bool = true;
    const FLOAT: bool = false;
    const COMPLEX: bool = true;
}
impl ComplexType for Complex256Bit {}

// Legacy type aliases for backward compatibility
pub type nbit_8 = Int8Bit;
pub type nbit_16 = Int16Bit;
pub type nbit_32 = Int32Bit;
pub type nbit_64 = Int64Bit;
pub type nbit_128 = Complex128Bit;
pub type nbit_256 = Complex256Bit;

// NumPy-compatible type aliases
pub type Int8 = Int8Bit;
pub type Int16 = Int16Bit;
pub type Int32 = Int32Bit;
pub type Int64 = Int64Bit;

pub type UInt8 = UInt8Bit;
pub type UInt16 = UInt16Bit;
pub type UInt32 = UInt32Bit;
pub type UInt64 = UInt64Bit;

pub type Float16 = Float16Bit;
pub type Float32 = Float32Bit;
pub type Float64 = Float64Bit;

pub type Complex32 = Complex32Bit;
pub type Complex64 = Complex64Bit;
pub type Complex128 = Complex128Bit;
pub type Complex256 = Complex256Bit;
