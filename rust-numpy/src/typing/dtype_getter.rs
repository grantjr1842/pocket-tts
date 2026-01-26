/// Dtype getter module for NumPy-compatible dtype syntax
///
/// This module provides the dtype getter functionality that enables
/// syntax like `np.dtype[np.float64]` similar to NumPy's API.
use crate::dtype::Dtype;
use crate::typing::bitwidth::*;

/// Dtype getter struct that provides NumPy-like dtype access
///
/// This enables the syntax `DtypeGetter[Float64Bit]` to get the corresponding
/// Dtype, similar to NumPy's `np.dtype[np.float64]`.
pub struct DtypeGetter;

impl DtypeGetter {
    /// Get the dtype corresponding to a bit-width type
    pub fn get<T: NBitBase>() -> Dtype {
        // Use the bit-width type information to determine the appropriate dtype
        if T::COMPLEX {
            match T::BITS {
                32 => Dtype::Complex32 { byteorder: None },
                64 => Dtype::Complex64 { byteorder: None },
                128 => Dtype::Complex128 { byteorder: None },
                256 => Dtype::Complex256 { byteorder: None },
                _ => panic!("Unsupported complex bit width: {}", T::BITS),
            }
        } else if T::FLOAT {
            match T::BITS {
                16 => Dtype::Float16 { byteorder: None },
                32 => Dtype::Float32 { byteorder: None },
                64 => Dtype::Float64 { byteorder: None },
                _ => panic!("Unsupported float bit width: {}", T::BITS),
            }
        } else if T::SIGNED {
            match T::BITS {
                8 => Dtype::Int8 { byteorder: None },
                16 => Dtype::Int16 { byteorder: None },
                32 => Dtype::Int32 { byteorder: None },
                64 => Dtype::Int64 { byteorder: None },
                _ => panic!("Unsupported signed integer bit width: {}", T::BITS),
            }
        } else {
            // Unsigned integers
            match T::BITS {
                8 => Dtype::UInt8 { byteorder: None },
                16 => Dtype::UInt16 { byteorder: None },
                32 => Dtype::UInt32 { byteorder: None },
                64 => Dtype::UInt64 { byteorder: None },
                _ => panic!("Unsupported unsigned integer bit width: {}", T::BITS),
            }
        }
    }
}

/// Trait for types that can be converted to dtypes
pub trait ToDtype {
    fn to_dtype() -> Dtype;
}

// Implement ToDtype for all bit-width types
impl ToDtype for Int8Bit {
    fn to_dtype() -> Dtype {
        Dtype::Int8 { byteorder: None }
    }
}
impl ToDtype for UInt8Bit {
    fn to_dtype() -> Dtype {
        Dtype::UInt8 { byteorder: None }
    }
}
impl ToDtype for Int16Bit {
    fn to_dtype() -> Dtype {
        Dtype::Int16 { byteorder: None }
    }
}
impl ToDtype for UInt16Bit {
    fn to_dtype() -> Dtype {
        Dtype::UInt16 { byteorder: None }
    }
}
impl ToDtype for Int32Bit {
    fn to_dtype() -> Dtype {
        Dtype::Int32 { byteorder: None }
    }
}
impl ToDtype for UInt32Bit {
    fn to_dtype() -> Dtype {
        Dtype::UInt32 { byteorder: None }
    }
}
impl ToDtype for Int64Bit {
    fn to_dtype() -> Dtype {
        Dtype::Int64 { byteorder: None }
    }
}
impl ToDtype for UInt64Bit {
    fn to_dtype() -> Dtype {
        Dtype::UInt64 { byteorder: None }
    }
}
impl ToDtype for Float16Bit {
    fn to_dtype() -> Dtype {
        Dtype::Float16 { byteorder: None }
    }
}
impl ToDtype for Float32Bit {
    fn to_dtype() -> Dtype {
        Dtype::Float32 { byteorder: None }
    }
}
impl ToDtype for Float64Bit {
    fn to_dtype() -> Dtype {
        Dtype::Float64 { byteorder: None }
    }
}
impl ToDtype for Complex32Bit {
    fn to_dtype() -> Dtype {
        Dtype::Complex32 { byteorder: None }
    }
}
impl ToDtype for Complex64Bit {
    fn to_dtype() -> Dtype {
        Dtype::Complex64 { byteorder: None }
    }
}
impl ToDtype for Complex128Bit {
    fn to_dtype() -> Dtype {
        Dtype::Complex128 { byteorder: None }
    }
}
impl ToDtype for Complex256Bit {
    fn to_dtype() -> Dtype {
        Dtype::Complex256 { byteorder: None }
    }
}

/// Convenience function for getting dtypes from bit-width types
///
/// This provides a more ergonomic API: `dtype::<Float64Bit>()`
pub fn dtype<T: ToDtype>() -> Dtype {
    T::to_dtype()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_getter_integer_types() {
        assert!(matches!(dtype::<Int8Bit>(), Dtype::Int8 { .. }));
        assert!(matches!(dtype::<UInt8Bit>(), Dtype::UInt8 { .. }));
        assert!(matches!(dtype::<Int16Bit>(), Dtype::Int16 { .. }));
        assert!(matches!(dtype::<UInt16Bit>(), Dtype::UInt16 { .. }));
        assert!(matches!(dtype::<Int32Bit>(), Dtype::Int32 { .. }));
        assert!(matches!(dtype::<UInt32Bit>(), Dtype::UInt32 { .. }));
        assert!(matches!(dtype::<Int64Bit>(), Dtype::Int64 { .. }));
        assert!(matches!(dtype::<UInt64Bit>(), Dtype::UInt64 { .. }));
    }

    #[test]
    fn test_dtype_getter_float_types() {
        assert!(matches!(dtype::<Float16Bit>(), Dtype::Float16 { .. }));
        assert!(matches!(dtype::<Float32Bit>(), Dtype::Float32 { .. }));
        assert!(matches!(dtype::<Float64Bit>(), Dtype::Float64 { .. }));
    }

    #[test]
    fn test_dtype_getter_complex_types() {
        assert!(matches!(dtype::<Complex32Bit>(), Dtype::Complex32 { .. }));
        assert!(matches!(dtype::<Complex64Bit>(), Dtype::Complex64 { .. }));
        assert!(matches!(dtype::<Complex128Bit>(), Dtype::Complex128 { .. }));
        assert!(matches!(dtype::<Complex256Bit>(), Dtype::Complex256 { .. }));
    }

    #[test]
    fn test_dtype_getter_method() {
        assert!(matches!(
            DtypeGetter::get::<Int32Bit>(),
            Dtype::Int32 { .. }
        ));
        assert!(matches!(
            DtypeGetter::get::<Float64Bit>(),
            Dtype::Float64 { .. }
        ));
        assert!(matches!(
            DtypeGetter::get::<Complex128Bit>(),
            Dtype::Complex128 { .. }
        ));
    }

    #[test]
    fn test_to_dtype_trait() {
        let int8_dtype = Int8Bit::to_dtype();
        let float64_dtype = Float64Bit::to_dtype();
        let complex128_dtype = Complex128Bit::to_dtype();

        assert!(matches!(int8_dtype, Dtype::Int8 { .. }));
        assert!(matches!(float64_dtype, Dtype::Float64 { .. }));
        assert!(matches!(complex128_dtype, Dtype::Complex128 { .. }));
    }
}
