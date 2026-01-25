#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_new_math_functions() {
        // Test that the new NumPy-compatible math functions are exported
        let arr = array![1.0, 2.0, 3.0];

        // These should now be available
        let _result = acos(&arr);
        let _result = asin(&arr);
        let _result = atan(&arr);
        let _result = atanh(&arr);
        let _result = asinh(&arr);
        let _result = acosh(&arr);
    }

    #[test]
    fn test_new_array_functions() {
        // Test that the new array conversion functions are available
        let arr = array![1.0, 2.0, 3.0];

        // These should now be available
        let _result = asarray(vec![1.0, 2.0, 3.0], None);
        let _result = asanyarray(vec![1.0, 2.0, 3.0], None);
        let _result = ascontiguousarray(&arr);
        let _result = asfortranarray(&arr);
        let _result = asmatrix(&arr);
        let _result = array2string(&arr);
        let _result = array_repr(&arr);
        let _result = array_str(&arr);
    }

    #[test]
    fn test_new_reduction_functions() {
        // Test that the new reduction functions are available
        let arr = array![1.0, 2.0, 3.0];

        // These should now be available
        let _result = amax(&arr, None, None, false);
        let _result = amin(&arr, None, None, false);
    }

    #[test]
    fn test_new_utility_functions() {
        // Test that the new utility functions are available
        let _result = base_repr(10, 2, Some(8));
        let _result = binary_repr(10, Some(8));
        let _result = bool();
        let _result = bool_();
        let _result = byte();
        let _result = double();
        let _result = single();
        let _result = int8();
        let _result = int16();
        let _result = int32();
        let _result = int64();
        let _result = uint8();
        let _result = uint16();
        let _result = uint32();
        let _result = uint64();
        let _result = floating();
        let _result = integer();
        let _result = generic();
        let _result = flexible();
        let _result = inexact();
        let _result = signedinteger();
        let _result = unsignedinteger();
        let _result = character();
        let _result = complexfloating();
        let _result = str_();
        let _result = void();
        let _result = object_();
        let _result = version();
        let _result = show_config();
        let _result = show_runtime();
        let _result = get_include();
        let _result = test();
        let _result = info();
        let _result = typename("float64");
        let _result = get_printoptions();
        let _result = set_printoptions(Some(6));
        let _result = getbufsize();
        let _result = setbufsize(8192);
        let _result = geterr();
        let _result = seterr(None, None, None, None, None);
        let _result = geterrcall();
        let _result = seterrcall(None);
        let _result = errstate();
        let _result = finfo("float64");
        let _result = iinfo("int64");
        let _result = isscalar("123");
        let _result = iterable("[1,2,3]");
        let _result = issubdtype("int32", "integer");
        let _result = isdtype("float64");
        let _result = iscomplex("1+2j");
        let _result = isreal("1.0");
        let _result = iscomplexobj("1+2j");
        let _result = isrealobj("1.0");
        let _result = isfortran("F order");
        let _result = may_share_memory("a", "b");
        let _result = shares_memory("a", "b");
        let _result = isnat("NaT");
        let _result = min_scalar_type("array");
        let _result = mintypecode("array");
        let _result = common_type(&["float32", "float64"]);
        let _result = result_type(&["float32", "float64"]);
        let _result = utils_promote_types("float32", "float64");
        let _result = can_cast("int8", "int16");
    }
}
