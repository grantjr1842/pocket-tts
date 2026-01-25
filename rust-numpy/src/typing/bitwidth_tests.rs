use crate::dtype::Dtype;
use crate::typing::bitwidth::*;
use crate::typing::dtype_getter::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nbit_base_trait_properties() {
        // Test integer types
        assert_eq!(Int8Bit::BITS, 8);
        assert!(Int8Bit::SIGNED);
        assert!(!Int8Bit::FLOAT);
        assert!(!Int8Bit::COMPLEX);

        assert_eq!(UInt8Bit::BITS, 8);
        assert!(!UInt8Bit::SIGNED);
        assert!(!UInt8Bit::FLOAT);
        assert!(!UInt8Bit::COMPLEX);

        assert_eq!(Int16Bit::BITS, 16);
        assert!(Int16Bit::SIGNED);
        assert!(!Int16Bit::FLOAT);
        assert!(!Int16Bit::COMPLEX);

        assert_eq!(UInt16Bit::BITS, 16);
        assert!(!UInt16Bit::SIGNED);
        assert!(!UInt16Bit::FLOAT);
        assert!(!UInt16Bit::COMPLEX);

        // Test float types
        assert_eq!(Float16Bit::BITS, 16);
        assert!(Float16Bit::SIGNED);
        assert!(Float16Bit::FLOAT);
        assert!(!Float16Bit::COMPLEX);

        assert_eq!(Float32Bit::BITS, 32);
        assert!(Float32Bit::SIGNED);
        assert!(Float32Bit::FLOAT);
        assert!(!Float32Bit::COMPLEX);

        assert_eq!(Float64Bit::BITS, 64);
        assert!(Float64Bit::SIGNED);
        assert!(Float64Bit::FLOAT);
        assert!(!Float64Bit::COMPLEX);

        // Test complex types
        assert_eq!(Complex32Bit::BITS, 32);
        assert!(Complex32Bit::SIGNED);
        assert!(!Complex32Bit::FLOAT);
        assert!(Complex32Bit::COMPLEX);

        assert_eq!(Complex64Bit::BITS, 64);
        assert!(Complex64Bit::SIGNED);
        assert!(!Complex64Bit::FLOAT);
        assert!(Complex64Bit::COMPLEX);

        assert_eq!(Complex128Bit::BITS, 128);
        assert!(Complex128Bit::SIGNED);
        assert!(!Complex128Bit::FLOAT);
        assert!(Complex128Bit::COMPLEX);

        assert_eq!(Complex256Bit::BITS, 256);
        assert!(Complex256Bit::SIGNED);
        assert!(!Complex256Bit::FLOAT);
        assert!(Complex256Bit::COMPLEX);
    }

    #[test]
    fn test_marker_traits() {
        // Test SignedInt marker trait
        fn is_signed_int<T: SignedInt>() -> bool {
            true
        }
        fn is_unsigned_int<T: UnsignedInt>() -> bool {
            true
        }
        fn is_float<T: FloatType>() -> bool {
            true
        }
        fn is_complex<T: ComplexType>() -> bool {
            true
        }

        assert!(is_signed_int::<Int8Bit>());
        assert!(is_signed_int::<Int16Bit>());
        assert!(is_signed_int::<Int32Bit>());
        assert!(is_signed_int::<Int64Bit>());

        assert!(is_unsigned_int::<UInt8Bit>());
        assert!(is_unsigned_int::<UInt16Bit>());
        assert!(is_unsigned_int::<UInt32Bit>());
        assert!(is_unsigned_int::<UInt64Bit>());

        assert!(is_float::<Float16Bit>());
        assert!(is_float::<Float32Bit>());
        assert!(is_float::<Float64Bit>());

        assert!(is_complex::<Complex32Bit>());
        assert!(is_complex::<Complex64Bit>());
        assert!(is_complex::<Complex128Bit>());
        assert!(is_complex::<Complex256Bit>());
    }

    #[test]
    fn test_type_aliases() {
        // Test that type aliases work correctly
        fn test_int8_type(_: Int8) {}
        fn test_uint8_type(_: UInt8) {}
        fn test_float32_type(_: Float32) {}
        fn test_complex64_type(_: Complex64) {}

        test_int8_type(Int8Bit);
        test_uint8_type(UInt8Bit);
        test_float32_type(Float32Bit);
        test_complex64_type(Complex64Bit);
    }

    #[test]
    fn test_dtype_getter_function() {
        // Test dtype getter function for all types
        let int8_dtype = dtype::<Int8Bit>();
        let uint8_dtype = dtype::<UInt8Bit>();
        let int16_dtype = dtype::<Int16Bit>();
        let uint16_dtype = dtype::<UInt16Bit>();
        let int32_dtype = dtype::<Int32Bit>();
        let uint32_dtype = dtype::<UInt32Bit>();
        let int64_dtype = dtype::<Int64Bit>();
        let uint64_dtype = dtype::<UInt64Bit>();

        assert!(matches!(int8_dtype, Dtype::Int8 { .. }));
        assert!(matches!(uint8_dtype, Dtype::UInt8 { .. }));
        assert!(matches!(int16_dtype, Dtype::Int16 { .. }));
        assert!(matches!(uint16_dtype, Dtype::UInt16 { .. }));
        assert!(matches!(int32_dtype, Dtype::Int32 { .. }));
        assert!(matches!(uint32_dtype, Dtype::UInt32 { .. }));
        assert!(matches!(int64_dtype, Dtype::Int64 { .. }));
        assert!(matches!(uint64_dtype, Dtype::UInt64 { .. }));

        // Test float types
        let float16_dtype = dtype::<Float16Bit>();
        let float32_dtype = dtype::<Float32Bit>();
        let float64_dtype = dtype::<Float64Bit>();

        assert!(matches!(float16_dtype, Dtype::Float16 { .. }));
        assert!(matches!(float32_dtype, Dtype::Float32 { .. }));
        assert!(matches!(float64_dtype, Dtype::Float64 { .. }));

        // Test complex types
        let complex32_dtype = dtype::<Complex32Bit>();
        let complex64_dtype = dtype::<Complex64Bit>();
        let complex128_dtype = dtype::<Complex128Bit>();
        let complex256_dtype = dtype::<Complex256Bit>();

        assert!(matches!(complex32_dtype, Dtype::Complex32 { .. }));
        assert!(matches!(complex64_dtype, Dtype::Complex64 { .. }));
        assert!(matches!(complex128_dtype, Dtype::Complex128 { .. }));
        assert!(matches!(complex256_dtype, Dtype::Complex256 { .. }));
    }

    #[test]
    fn test_dtype_getter_method() {
        // Test DtypeGetter::get method
        let int32_dtype = DtypeGetter::get::<Int32Bit>();
        let float64_dtype = DtypeGetter::get::<Float64Bit>();
        let complex128_dtype = DtypeGetter::get::<Complex128Bit>();

        assert!(matches!(int32_dtype, Dtype::Int32 { .. }));
        assert!(matches!(float64_dtype, Dtype::Float64 { .. }));
        assert!(matches!(complex128_dtype, Dtype::Complex128 { .. }));
    }

    #[test]
    fn test_to_dtype_trait_implementation() {
        // Test ToDtype trait implementations
        let int8_dtype = Int8Bit::to_dtype();
        let uint16_dtype = UInt16Bit::to_dtype();
        let float32_dtype = Float32Bit::to_dtype();
        let complex64_dtype = Complex64Bit::to_dtype();

        assert!(matches!(int8_dtype, Dtype::Int8 { .. }));
        assert!(matches!(uint16_dtype, Dtype::UInt16 { .. }));
        assert!(matches!(float32_dtype, Dtype::Float32 { .. }));
        assert!(matches!(complex64_dtype, Dtype::Complex64 { .. }));
    }

    #[test]
    fn test_legacy_type_aliases() {
        // Test that legacy type aliases still work
        fn test_legacy_types() {
            let _: nbit_8 = Int8Bit;
            let _: nbit_16 = Int16Bit;
            let _: nbit_32 = Int32Bit;
            let _: nbit_64 = Int64Bit;
            let _: nbit_128 = Complex128Bit;
            let _: nbit_256 = Complex256Bit;
        }
        test_legacy_types();
    }

    #[test]
    fn test_bit_width_consistency() {
        // Test that bit widths are consistent across related types
        assert_eq!(Int8Bit::BITS, UInt8Bit::BITS);
        assert_eq!(Int16Bit::BITS, UInt16Bit::BITS);
        assert_eq!(Int32Bit::BITS, UInt32Bit::BITS);
        assert_eq!(Int64Bit::BITS, UInt64Bit::BITS);

        // Complex types should have twice the bit width of their component float types
        assert_eq!(Complex32Bit::BITS, Float16Bit::BITS * 2);
        assert_eq!(Complex64Bit::BITS, Float32Bit::BITS * 2);
        assert_eq!(Complex128Bit::BITS, Float64Bit::BITS * 2);
    }

    #[test]
    fn test_type_safety() {
        // Test that the type system provides proper safety
        fn require_signed_int<T: SignedInt>() {}
        fn require_unsigned_int<T: UnsignedInt>() {}
        fn require_float<T: FloatType>() {}
        fn require_complex<T: ComplexType>() {}

        // These should compile
        require_signed_int::<Int32Bit>();
        require_unsigned_int::<UInt32Bit>();
        require_float::<Float64Bit>();
        require_complex::<Complex128Bit>();

        // The following would not compile (uncomment to test):
        // require_signed_int::<UInt32Bit>(); // Error: UInt32Bit doesn't implement SignedInt
        // require_unsigned_int::<Int32Bit>(); // Error: Int32Bit doesn't implement UnsignedInt
        // require_float::<Int32Bit>(); // Error: Int32Bit doesn't implement FloatType
        // require_complex::<Float64Bit>(); // Error: Float64Bit doesn't implement ComplexType
    }

    #[test]
    fn test_all_bit_width_types_covered() {
        // Ensure we have implementations for all expected bit widths
        let types = vec![
            ("Int8", dtype::<Int8Bit>()),
            ("UInt8", dtype::<UInt8Bit>()),
            ("Int16", dtype::<Int16Bit>()),
            ("UInt16", dtype::<UInt16Bit>()),
            ("Int32", dtype::<Int32Bit>()),
            ("UInt32", dtype::<UInt32Bit>()),
            ("Int64", dtype::<Int64Bit>()),
            ("UInt64", dtype::<UInt64Bit>()),
            ("Float16", dtype::<Float16Bit>()),
            ("Float32", dtype::<Float32Bit>()),
            ("Float64", dtype::<Float64Bit>()),
            ("Complex32", dtype::<Complex32Bit>()),
            ("Complex64", dtype::<Complex64Bit>()),
            ("Complex128", dtype::<Complex128Bit>()),
            ("Complex256", dtype::<Complex256Bit>()),
        ];

        assert_eq!(types.len(), 15); // Ensure we have all 15 types

        for (name, dtype_val) in types {
            // Verify each dtype is valid (not None/null)
            match dtype_val {
                Dtype::Int8 { .. } => assert_eq!(name, "Int8"),
                Dtype::UInt8 { .. } => assert_eq!(name, "UInt8"),
                Dtype::Int16 { .. } => assert_eq!(name, "Int16"),
                Dtype::UInt16 { .. } => assert_eq!(name, "UInt16"),
                Dtype::Int32 { .. } => assert_eq!(name, "Int32"),
                Dtype::UInt32 { .. } => assert_eq!(name, "UInt32"),
                Dtype::Int64 { .. } => assert_eq!(name, "Int64"),
                Dtype::UInt64 { .. } => assert_eq!(name, "UInt64"),
                Dtype::Float16 { .. } => assert_eq!(name, "Float16"),
                Dtype::Float32 { .. } => assert_eq!(name, "Float32"),
                Dtype::Float64 { .. } => assert_eq!(name, "Float64"),
                Dtype::Complex32 { .. } => assert_eq!(name, "Complex32"),
                Dtype::Complex64 { .. } => assert_eq!(name, "Complex64"),
                Dtype::Complex128 { .. } => assert_eq!(name, "Complex128"),
                Dtype::Complex256 { .. } => assert_eq!(name, "Complex256"),
                _ => panic!("Unexpected dtype for {}", name),
            }
        }
    }
}
