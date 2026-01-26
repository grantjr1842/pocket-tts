use rust_numpy::dtype::Dtype;
use rust_numpy::type_promotion::{promote_types, TypePromotionRules};

#[test]
fn test_new_vs_old_promotion_compatibility() {
    let rules = TypePromotionRules::new();

    // Test cases to verify compatibility
    let test_cases = vec![
        (
            Dtype::Int32 { byteorder: None },
            Dtype::Float64 { byteorder: None },
        ),
        (
            Dtype::UInt8 { byteorder: None },
            Dtype::Int16 { byteorder: None },
        ),
        (
            Dtype::Float32 { byteorder: None },
            Dtype::Complex64 { byteorder: None },
        ),
        (Dtype::Bool, Dtype::Int32 { byteorder: None }),
        (
            Dtype::Int64 { byteorder: None },
            Dtype::UInt64 { byteorder: None },
        ),
    ];

    for (t1, t2) in test_cases {
        // Test new TypePromotionRules
        let new_result = rules.promote_two_types(&t1, &t2).unwrap();

        // Test legacy promote_types function
        let old_result = promote_types(&t1, &t2).unwrap();

        // They should be compatible (though new implementation might be more precise)
        println!(
            "Promoting {:?} + {:?}: new={:?}, old={:?}",
            t1, t2, new_result, old_result
        );

        // At minimum, the kind should be the same or more precise
        assert!(
            new_result.kind() == old_result.kind()
                || new_result.itemsize() >= old_result.itemsize()
        );
    }
}

#[test]
fn test_type_promotion_rules_with_multiple_types() {
    let rules = TypePromotionRules::new();

    let types = vec![
        Dtype::Int8 { byteorder: None },
        Dtype::Int16 { byteorder: None },
        Dtype::Float32 { byteorder: None },
        Dtype::Float64 { byteorder: None },
    ];

    let result = rules.promote_types(&types).unwrap();
    assert_eq!(result, Dtype::Float64 { byteorder: None });
}

#[test]
fn test_comprehensive_promotion_coverage() {
    let rules = TypePromotionRules::new();

    // Test all major dtype categories
    let bool_type = Dtype::Bool;
    let int_type = Dtype::Int32 { byteorder: None };
    let uint_type = Dtype::UInt32 { byteorder: None };
    let float_type = Dtype::Float64 { byteorder: None };
    let complex_type = Dtype::Complex128 { byteorder: None };
    let string_type = Dtype::Unicode { length: None };

    // Test that we can promote any combination
    let type_pairs = vec![
        (&bool_type, &int_type),
        (&bool_type, &float_type),
        (&bool_type, &complex_type),
        (&int_type, &uint_type),
        (&int_type, &float_type),
        (&int_type, &complex_type),
        (&uint_type, &float_type),
        (&uint_type, &complex_type),
        (&float_type, &complex_type),
        (&string_type, &int_type),
    ];

    for (t1, t2) in type_pairs {
        let result = rules.promote_two_types(t1, t2);
        assert!(result.is_ok(), "Failed to promote {:?} and {:?}", t1, t2);
    }
}
