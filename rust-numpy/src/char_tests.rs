use crate::char::*;
use crate::Array;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_string_array(strings: Vec<&str>) -> Array<String> {
        Array::from_vec(strings.into_iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn test_char_add() {
        let a = create_string_array(vec!["hello", "world"]);
        let b = create_string_array(vec![" ", "test"]);
        let result = add(&a, &b).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello ".to_string());
        assert_eq!(result.get(1).unwrap(), &"worldtest".to_string());
    }

    #[test]
    fn test_char_multiply() {
        let a = create_string_array(vec!["hi", "test"]);
        let result = multiply(&a, 3).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hihihi".to_string());
        assert_eq!(result.get(1).unwrap(), &"testtesttest".to_string());
    }

    #[test]
    fn test_char_multiply_zero() {
        let a = create_string_array(vec!["hi", "test"]);
        let result = multiply(&a, 0).unwrap();
        assert_eq!(result.get(0).unwrap(), &"".to_string());
        assert_eq!(result.get(1).unwrap(), &"".to_string());
    }

    #[test]
    fn test_char_multiply_negative() {
        let a = create_string_array(vec!["hi"]);
        let result = multiply(&a, -1);
        assert!(result.is_err());
    }

    #[test]
    fn test_char_capitalize() {
        let a = create_string_array(vec!["hello", "WORLD", "test"]);
        let result = capitalize(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"Hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"World".to_string());
        assert_eq!(result.get(2).unwrap(), &"Test".to_string());
    }

    #[test]
    fn test_char_capitalize_empty() {
        let a = create_string_array(vec![""]);
        let result = capitalize(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"".to_string());
    }

    #[test]
    fn test_char_lower() {
        let a = create_string_array(vec!["HELLO", "World", "TeSt"]);
        let result = lower(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"world".to_string());
        assert_eq!(result.get(2).unwrap(), &"test".to_string());
    }

    #[test]
    fn test_char_upper() {
        let a = create_string_array(vec!["hello", "World", "TeSt"]);
        let result = upper(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"HELLO".to_string());
        assert_eq!(result.get(1).unwrap(), &"WORLD".to_string());
        assert_eq!(result.get(2).unwrap(), &"TEST".to_string());
    }

    #[test]
    fn test_char_strip() {
        let a = create_string_array(vec!["  hello  ", "\tworld\n", "  test  "]);
        let result = strip(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"world".to_string());
        assert_eq!(result.get(2).unwrap(), &"test".to_string());
    }

    #[test]
    fn test_char_lstrip() {
        let a = create_string_array(vec!["  hello  ", "\tworld\t", "  test  "]);
        let result = lstrip(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello  ".to_string());
        assert_eq!(result.get(1).unwrap(), &"world\t".to_string());
        assert_eq!(result.get(2).unwrap(), &"test  ".to_string());
    }

    #[test]
    fn test_char_rstrip() {
        let a = create_string_array(vec!["  hello  ", "\tworld\n", "  test  "]);
        let result = rstrip(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"  hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"\tworld".to_string());
        assert_eq!(result.get(2).unwrap(), &"  test".to_string());
    }

    #[test]
    fn test_char_strip_chars() {
        let a = create_string_array(vec!["xxhelloxx", "yyworldyy", "zztestzz"]);
        let result = strip_chars(&a, "xyz").unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"world".to_string());
        assert_eq!(result.get(2).unwrap(), &"test".to_string());
    }

    #[test]
    fn test_char_center() {
        let a = create_string_array(vec!["a", "abc"]);
        let result = center(&a, 5, None).unwrap();
        assert_eq!(result.get(0).unwrap(), &"  a  ".to_string());
        assert_eq!(result.get(1).unwrap(), &" abc ".to_string());

        let result_char = center(&a, 5, Some('-')).unwrap();
        assert_eq!(result_char.get(0).unwrap(), &"--a--".to_string());
    }

    #[test]
    fn test_char_zfill() {
        let a = create_string_array(vec!["1", "123"]);
        let result = zfill(&a, 3).unwrap();
        assert_eq!(result.get(0).unwrap(), &"001".to_string());
        assert_eq!(result.get(1).unwrap(), &"123".to_string());
    }

    #[test]
    fn test_char_expandtabs() {
        let a = create_string_array(vec!["\t", "a\tb"]);
        let result = expandtabs(&a, Some(4)).unwrap();
        assert_eq!(result.get(0).unwrap(), &"    ".to_string());
        assert_eq!(result.get(1).unwrap(), &"a   b".to_string());
    }

    #[test]
    fn test_char_check_property() {
        let a = create_string_array(vec!["abc", "123", " "]);
        assert_eq!(isalpha(&a).unwrap().to_vec(), vec![true, false, false]);
        assert_eq!(isdigit(&a).unwrap().to_vec(), vec![false, true, false]);
        assert_eq!(isalnum(&a).unwrap().to_vec(), vec![true, true, false]);
        assert_eq!(isspace(&a).unwrap().to_vec(), vec![false, false, true]);
    }

    #[test]
    fn test_char_search() {
        let a = create_string_array(vec!["banana"]);
        assert_eq!(find(&a, "na", None, None).unwrap().to_vec(), vec![2]);
        assert_eq!(rfind(&a, "na", None, None).unwrap().to_vec(), vec![4]);
        assert_eq!(count(&a, "na", None, None).unwrap().to_vec(), vec![2]);
        assert_eq!(count(&a, "a", None, None).unwrap().to_vec(), vec![3]);
        assert_eq!(index(&a, "na", None, None).unwrap().to_vec(), vec![2]);
        assert_eq!(rindex(&a, "na", None, None).unwrap().to_vec(), vec![4]);
    }

    #[test]
    fn test_char_search_fail() {
        let a = create_string_array(vec!["abc"]);
        assert_eq!(find(&a, "z", None, None).unwrap().to_vec(), vec![-1]);
        assert!(index(&a, "z", None, None).is_err());
    }

    #[test]
    fn test_char_replace() {
        let a = create_string_array(vec!["hello world", "test case"]);
        let result = replace(&a, " ", "_").unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello_world".to_string());
        assert_eq!(result.get(1).unwrap(), &"test_case".to_string());
    }

    #[test]
    fn test_char_split() {
        let a = create_string_array(vec!["hello world", "test case"]);
        let result = split(&a, " ").unwrap();
        assert_eq!(result.size(), 4);
        assert_eq!(result.get(0).unwrap(), &"hello".to_string());
        assert_eq!(result.get(1).unwrap(), &"world".to_string());
        assert_eq!(result.get(2).unwrap(), &"test".to_string());
        assert_eq!(result.get(3).unwrap(), &"case".to_string());
    }

    #[test]
    fn test_char_join() {
        let a = create_string_array(vec!["hello", "world", "test"]);
        let result = join(",", &a).unwrap();
        assert_eq!(result.size(), 1);
        assert_eq!(result.get(0).unwrap(), &"hello,world,test".to_string());
    }

    #[test]
    fn test_char_startswith() {
        let a = create_string_array(vec!["hello", "world", "test"]);
        let result = startswith(&a, "he").unwrap();
        assert_eq!(result.get(0).unwrap(), &true);
        assert_eq!(result.get(1).unwrap(), &false);
        assert_eq!(result.get(2).unwrap(), &false);
    }

    #[test]
    fn test_char_endswith() {
        let a = create_string_array(vec!["hello", "world", "test"]);
        let result = endswith(&a, "lo").unwrap();
        assert_eq!(result.get(0).unwrap(), &true);
        assert_eq!(result.get(1).unwrap(), &false);
        assert_eq!(result.get(2).unwrap(), &false);
    }

    #[test]
    fn test_char_shape_mismatch() {
        let a = create_string_array(vec!["hello", "world"]);
        let b = create_string_array(vec!["test"]);
        let result = add(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_char_empty_arrays() {
        let a: Array<String> = Array::from_vec(vec![]);
        let b: Array<String> = Array::from_vec(vec![]);
        let result = add(&a, &b).unwrap();
        assert_eq!(result.size(), 0);
    }

    #[test]
    fn test_char_single_element() {
        let a = create_string_array(vec!["test"]);
        let result = upper(&a).unwrap();
        assert_eq!(result.size(), 1);
        assert_eq!(result.get(0).unwrap(), &"TEST".to_string());
    }

    #[test]
    fn test_char_unicode_handling() {
        let a = create_string_array(vec!["héllo", "wörld", "tëst"]);
        let result = upper(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"HÉLLO".to_string());
        assert_eq!(result.get(1).unwrap(), &"WÖRLD".to_string());
        assert_eq!(result.get(2).unwrap(), &"TËST".to_string());
    }

    #[test]
    fn test_char_complex_strings() {
        let a = create_string_array(vec!["  hello_world  ", "\t\tTest\tCase\n\n"]);
        let result = strip(&a).unwrap();
        assert_eq!(result.get(0).unwrap(), &"hello_world".to_string());
        assert_eq!(result.get(1).unwrap(), &"Test\tCase".to_string());
    }
}
