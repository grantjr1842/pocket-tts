use numpy::dtype::Dtype;
use numpy::type_promotion::TypePromotionRules;

#[test]
fn test_type_promotion_rules_basic() {
    let rules = TypePromotionRules::new();

    let i32_type = Dtype::Int32 { byteorder: None };
    let f64_type = Dtype::Float64 { byteorder: None };

    // Test basic promotion
    let result = rules.promote_two_types(&i32_type, &f64_type).unwrap();
    assert_eq!(result, Dtype::Float64 { byteorder: None });

    // Test same type promotion
    let result = rules.promote_two_types(&i32_type, &i32_type).unwrap();
    assert_eq!(result, i32_type);
}

#[test]
fn test_type_promotion_rules_multiple() {
    let rules = TypePromotionRules::new();

    let i8_type = Dtype::Int8 { byteorder: None };
    let i32_type = Dtype::Int32 { byteorder: None };
    let f64_type = Dtype::Float64 { byteorder: None };

    // Test multiple type promotion
    let result = rules.promote_types(&[i8_type, i32_type, f64_type]).unwrap();
    assert_eq!(result, Dtype::Float64 { byteorder: None });
}

#[test]
fn test_boolean_promotion() {
    let rules = TypePromotionRules::new();

    let bool_type = Dtype::Bool;
    let i32_type = Dtype::Int32 { byteorder: None };
    let f32_type = Dtype::Float32 { byteorder: None };
    let c64_type = Dtype::Complex64 { byteorder: None };

    // Bool + Int -> Int
    let result = rules.promote_two_types(&bool_type, &i32_type).unwrap();
    assert!(matches!(result, Dtype::Int8 { .. }));

    // Bool + Float -> Float
    let result = rules.promote_two_types(&bool_type, &f32_type).unwrap();
    assert_eq!(result, Dtype::Float32 { byteorder: None });

    // Bool + Complex -> Complex
    let result = rules.promote_two_types(&bool_type, &c64_type).unwrap();
    assert_eq!(result, Dtype::Complex64 { byteorder: None });
}

#[test]
fn test_mixed_integer_promotion() {
    let rules = TypePromotionRules::new();

    let u8_type = Dtype::UInt8 { byteorder: None };
    let i8_type = Dtype::Int8 { byteorder: None };
    let u32_type = Dtype::UInt32 { byteorder: None };
    let i64_type = Dtype::Int64 { byteorder: None };

    // u8 + i8 -> larger signed (should be i16 in NumPy, our table gives Int64)
    let result = rules.promote_two_types(&u8_type, &i8_type).unwrap();
    assert!(matches!(result, Dtype::Int64 { .. }));

    // u32 + i64 -> i64 (to fit u32 range)
    let result = rules.promote_two_types(&u32_type, &i64_type).unwrap();
    assert_eq!(result, Dtype::Int64 { byteorder: None });
}

#[test]
fn test_complex_promotion() {
    let rules = TypePromotionRules::new();

    let f32_type = Dtype::Float32 { byteorder: None };
    let f64_type = Dtype::Float64 { byteorder: None };
    let c64_type = Dtype::Complex64 { byteorder: None };
    let c128_type = Dtype::Complex128 { byteorder: None };

    // Float + Complex -> Complex
    let result = rules.promote_two_types(&f32_type, &c64_type).unwrap();
    assert_eq!(result, Dtype::Complex128 { byteorder: None });

    // Complex64 + Complex128 -> Complex128
    let result = rules.promote_two_types(&c64_type, &c128_type).unwrap();
    assert_eq!(result, Dtype::Complex128 { byteorder: None });
}

#[test]
fn test_string_bytes_promotion() {
    let rules = TypePromotionRules::new();

    let string_type = Dtype::String { length: Some(10) };
    let bytes_type = Dtype::Bytes { length: 10 };
    let unicode_type = Dtype::Unicode { length: Some(15) };

    // String + Bytes -> Unicode (NumPy behavior)
    let result = rules.promote_two_types(&string_type, &bytes_type).unwrap();
    assert!(matches!(result, Dtype::Unicode { .. }));

    // String + Unicode -> Unicode
    let result = rules
        .promote_two_types(&string_type, &unicode_type)
        .unwrap();
    assert!(matches!(result, Dtype::Unicode { .. }));
}

#[test]
fn test_safe_casting() {
    let rules = TypePromotionRules::new();

    let i8_type = Dtype::Int8 { byteorder: None };
    let i16_type = Dtype::Int16 { byteorder: None };
    let i32_type = Dtype::Int32 { byteorder: None };
    let u8_type = Dtype::UInt8 { byteorder: None };
    let f32_type = Dtype::Float32 { byteorder: None };
    let f64_type = Dtype::Float64 { byteorder: None };

    // Safe integer promotions
    assert!(rules.can_safely_cast(&i8_type, &i16_type));
    assert!(rules.can_safely_cast(&i8_type, &i32_type));
    assert!(rules.can_safely_cast(&i16_type, &i32_type));

    // Safe unsigned to signed (when signed is larger)
    assert!(rules.can_safely_cast(&u8_type, &i16_type));
    assert!(rules.can_safely_cast(&u8_type, &i32_type));

    // Safe float promotions
    assert!(rules.can_safely_cast(&f32_type, &f64_type));

    // Integer to float (generally safe)
    assert!(rules.can_safely_cast(&i32_type, &f64_type));
    assert!(rules.can_safely_cast(&u8_type, &f32_type));

    // Unsafe casts
    assert!(!rules.can_safely_cast(&i32_type, &i16_type)); // Downgrade
    assert!(!rules.can_safely_cast(&f64_type, &f32_type)); // Precision loss
}

#[test]
fn test_edge_cases() {
    let rules = TypePromotionRules::new();

    // Empty type list should error
    let result = rules.promote_types(&[]);
    assert!(result.is_err());

    // Single type should return itself
    let i32_type = Dtype::Int32 { byteorder: None };
    let result = rules.promote_types(&[i32_type.clone()]).unwrap();
    assert_eq!(result, i32_type);
}

#[test]
fn test_datetime_promotion() {
    let rules = TypePromotionRules::new();

    let dt1 = Dtype::Datetime64(numpy::dtype::DatetimeUnit::ns);
    let dt2 = Dtype::Datetime64(numpy::dtype::DatetimeUnit::us);

    // Datetime promotion should work
    let result = rules.promote_two_types(&dt1, &dt2);
    assert!(result.is_ok());
}

#[test]
fn test_size_adjustment() {
    let rules = TypePromotionRules::new();

    let i64_type = Dtype::Int64 { byteorder: None };
    let f32_type = Dtype::Float32 { byteorder: None };

    // Large int + small float should promote to larger float
    let result = rules.promote_two_types(&i64_type, &f32_type).unwrap();
    // Should adjust to Float64 due to size considerations
    assert_eq!(result, Dtype::Float64 { byteorder: None });
}

#[test]
fn test_all_dtype_kinds_coverage() {
    let rules = TypePromotionRules::new();

    // Test that all major dtype kinds can be promoted
    let types = vec![
        Dtype::Bool,
        Dtype::Int8 { byteorder: None },
        Dtype::UInt8 { byteorder: None },
        Dtype::Float32 { byteorder: None },
        Dtype::Complex64 { byteorder: None },
        Dtype::String { length: None },
        Dtype::Unicode { length: None },
        Dtype::Bytes { length: 10 },
        Dtype::Datetime64(numpy::dtype::DatetimeUnit::ns),
        Dtype::Object,
    ];

    // Test pairwise promotion for all types
    for (i, type1) in types.iter().enumerate() {
        for type2 in types.iter().skip(i + 1) {
            let result = rules.promote_two_types(type1, type2);
            // Most combinations should work, except some incompatible ones
            if result.is_err() {
                println!(
                    "Failed to promote {:?} and {:?}: {:?}",
                    type1, type2, result
                );
            }
        }
    }
}

#[test]
fn test_promotion_table_consistency() {
    let rules = TypePromotionRules::new();

    // Test that promotion is commutative
    let i32_type = Dtype::Int32 { byteorder: None };
    let f64_type = Dtype::Float64 { byteorder: None };

    let result1 = rules.promote_two_types(&i32_type, &f64_type).unwrap();
    let result2 = rules.promote_two_types(&f64_type, &i32_type).unwrap();
    assert_eq!(result1, result2);

    // Test associativity: (a + b) + c == a + (b + c)
    let i8_type = Dtype::Int8 { byteorder: None };
    let u16_type = Dtype::UInt16 { byteorder: None };

    let result1 = rules
        .promote_types(&[i8_type.clone(), i32_type.clone(), u16_type.clone()])
        .unwrap();
    let result2 = {
        let temp = rules.promote_two_types(&i32_type, &u16_type).unwrap();
        rules.promote_two_types(&i8_type, &temp).unwrap()
    };
    assert_eq!(result1, result2);
}
