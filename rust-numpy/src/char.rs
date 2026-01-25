use crate::error::NumPyError;

/// Character string operations on arrays
///
/// This module provides vectorized string operations similar to numpy.char
pub fn add(
    x1: &crate::Array<String>,
    x2: &crate::Array<String>,
) -> Result<crate::Array<String>, NumPyError> {
    if x1.shape() != x2.shape() {
        return Err(NumPyError::shape_mismatch(
            x1.shape().to_vec(),
            x2.shape().to_vec(),
        ));
    }

    let mut result = Vec::with_capacity(x1.size());

    for i in 0..x1.size() {
        if let (Some(s1), Some(s2)) = (get_string(x1, i), get_string(x2, i)) {
            result.push(format!("{}{}", s1, s2));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn multiply(a: &crate::Array<String>, i: isize) -> Result<crate::Array<String>, NumPyError> {
    if i < 0 {
        return Err(NumPyError::invalid_value("i must be >= 0"));
    }

    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.repeat(i as usize));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn capitalize(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let mut chars: Vec<char> = s.chars().collect();
            if !chars.is_empty() {
                chars[0] = chars[0].to_ascii_uppercase();
                for c in chars.iter_mut().skip(1) {
                    *c = c.to_ascii_lowercase();
                }
            }
            result.push(chars.into_iter().collect());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn lower(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.to_lowercase());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn upper(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.to_uppercase());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn strip(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_matches(|c: char| c.is_whitespace()).to_string());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn lstrip(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(
                s.trim_start_matches(|c: char| c.is_whitespace())
                    .to_string(),
            );
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn rstrip(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_end_matches(|c: char| c.is_whitespace()).to_string());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn replace(
    a: &crate::Array<String>,
    old: &str,
    new: &str,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.replace(old, new));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn split(a: &crate::Array<String>, sep: &str) -> Result<crate::Array<String>, NumPyError> {
    let mut all_results: Vec<String> = Vec::new();

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let parts: Vec<&str> = s.split(sep).collect();
            all_results.extend(parts.iter().map(|&p| p.to_string()));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(all_results))
}

pub fn join(sep: &str, a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.to_string());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(vec![result.join(sep)]))
}

pub fn startswith(
    a: &crate::Array<String>,
    prefix: &str,
) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = s.starts_with(prefix);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn endswith(a: &crate::Array<String>, suffix: &str) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = s.ends_with(suffix);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Left-justify strings in array
pub fn ljust(
    a: &crate::Array<String>,
    width: usize,
    fillchar: Option<char>,
) -> Result<crate::Array<String>, NumPyError> {
    let fill = fillchar.unwrap_or(' ');
    if !fill.is_ascii() || fill.is_ascii_control() {
        return Err(NumPyError::invalid_value(
            "fillchar must be a non-control ASCII character",
        ));
    }

    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            if s.len() >= width {
                result.push(s.clone());
            } else {
                let padding = width - s.len();
                result.push(format!("{}{}", s, fill.to_string().repeat(padding)));
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Right-justify strings in array
pub fn rjust(
    a: &crate::Array<String>,
    width: usize,
    fillchar: Option<char>,
) -> Result<crate::Array<String>, NumPyError> {
    let fill = fillchar.unwrap_or(' ');
    if !fill.is_ascii() || fill.is_ascii_control() {
        return Err(NumPyError::invalid_value(
            "fillchar must be a non-control ASCII character",
        ));
    }

    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            if s.len() >= width {
                result.push(s.clone());
            } else {
                let padding = width - s.len();
                result.push(format!("{}{}", fill.to_string().repeat(padding), s));
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Swap case of strings (uppercase to lowercase and vice versa)
pub fn swapcase(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let swapped: String = s
                .chars()
                .map(|c| {
                    if c.is_uppercase() {
                        c.to_lowercase()
                            .collect::<String>()
                            .chars()
                            .next()
                            .unwrap_or(c)
                    } else if c.is_lowercase() {
                        c.to_uppercase()
                            .collect::<String>()
                            .chars()
                            .next()
                            .unwrap_or(c)
                    } else {
                        c
                    }
                })
                .collect();
            result.push(swapped);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Title-case strings (first character of each word uppercase)
pub fn title(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let titled: String = s
                .split_whitespace()
                .map(|word| {
                    let mut chars: Vec<char> = word.chars().collect();
                    if !chars.is_empty() {
                        chars[0] = chars[0].to_ascii_uppercase();
                        for c in chars.iter_mut().skip(1) {
                            *c = c.to_ascii_lowercase();
                        }
                    }
                    chars.into_iter().collect::<String>()
                })
                .collect::<Vec<String>>()
                .join(" ");
            result.push(titled);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Split strings from right, at most n times
pub fn rsplit(
    a: &crate::Array<String>,
    sep: &str,
    maxsplit: Option<isize>,
) -> Result<crate::Array<String>, NumPyError> {
    let n = maxsplit.unwrap_or(-1);
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let parts = if n < 0 {
                s.rsplit(sep).collect::<Vec<_>>()
            } else {
                let mut parts: Vec<&str> = s.rsplitn(n as usize + 1, sep).collect();
                parts.reverse();
                parts
            };
            result.push(parts.join(" "));
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Partition strings around separator (from left)
pub fn partition(
    a: &crate::Array<String>,
    sep: &str,
) -> Result<crate::Array<Vec<String>>, NumPyError> {
    let mut result = vec![vec![]; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            if let Some(pos) = s.find(sep) {
                result.push(vec![
                    s[..pos].to_string(),
                    sep.to_string(),
                    s[pos + sep.len()..].to_string(),
                ]);
            } else {
                result.push(vec![s.clone(), String::new(), String::new()]);
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Partition strings around separator (from right)
pub fn rpartition(
    a: &crate::Array<String>,
    sep: &str,
) -> Result<crate::Array<Vec<String>>, NumPyError> {
    let mut result = vec![vec![]; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            if let Some(pos) = s.rfind(sep) {
                result.push(vec![
                    s[..pos].to_string(),
                    sep.to_string(),
                    s[pos + sep.len()..].to_string(),
                ]);
            } else {
                result.push(vec![String::new(), String::new(), s.clone()]);
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Split strings at line boundaries
pub fn splitlines(
    a: &crate::Array<String>,
    keepends: Option<bool>,
) -> Result<crate::Array<Vec<String>>, NumPyError> {
    let keep = keepends.unwrap_or(false);
    let mut result = vec![vec![]; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let lines: Vec<String> = if keep {
                s.split('\n').map(|l| l.to_string()).collect()
            } else {
                s.lines().map(|l| l.to_string()).collect()
            };
            result.push(lines);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Get string lengths
pub fn str_len(a: &crate::Array<String>) -> Result<crate::Array<usize>, NumPyError> {
    let mut result = vec![0usize; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = s.len();
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// String comparison: equal
pub fn equal(
    a: &crate::Array<String>,
    b: &crate::Array<String>,
) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 == s2)
}

/// String comparison: not equal
pub fn not_equal(
    a: &crate::Array<String>,
    b: &crate::Array<String>,
) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 != s2)
}

/// String comparison: greater (lexicographic)
pub fn greater(
    a: &crate::Array<String>,
    b: &crate::Array<String>,
) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 > s2)
}

/// String comparison: greater or equal
pub fn greater_equal(
    a: &crate::Array<String>,
    b: &crate::Array<String>,
) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 >= s2)
}

/// String comparison: less (lexicographic)
pub fn less(
    a: &crate::Array<String>,
    b: &crate::Array<String>,
) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 < s2)
}

/// String comparison: less or equal
pub fn less_equal(
    a: &crate::Array<String>,
    b: &crate::Array<String>,
) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 <= s2)
}

fn compare_arrays<F>(
    a: &crate::Array<String>,
    b: &crate::Array<String>,
    compare: F,
) -> Result<crate::Array<bool>, NumPyError>
where
    F: Fn(&str, &str) -> bool,
{
    if a.shape() != b.shape() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let (Some(s1), Some(s2)) = (get_string(a, idx), get_string(b, idx)) {
            result[idx] = compare(&s1, &s2);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

fn get_string(a: &crate::Array<String>, idx: usize) -> Option<String> {
    a.get(idx).cloned()
}

pub fn center(
    a: &crate::Array<String>,
    width: usize,
    fillchar: Option<char>,
) -> Result<crate::Array<String>, NumPyError> {
    let fill = fillchar.unwrap_or(' ');
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let len = s.chars().count();
            if width <= len {
                result.push(s);
            } else {
                let pad = width - len;
                let left = pad / 2;
                let right = pad - left;
                let mut new_s = String::with_capacity(width);
                for _ in 0..left {
                    new_s.push(fill);
                }
                new_s.push_str(&s);
                for _ in 0..right {
                    new_s.push(fill);
                }
                result.push(new_s);
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    Ok(crate::Array::from_vec(result))
}

pub fn zfill(a: &crate::Array<String>, width: usize) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let len = s.chars().count();
            if width <= len {
                result.push(s);
            } else {
                let pad = width - len;
                let mut new_s = String::with_capacity(width);
                for _ in 0..pad {
                    new_s.push('0');
                }
                new_s.push_str(&s);
                result.push(new_s);
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn expandtabs(
    a: &crate::Array<String>,
    tabsize: Option<usize>,
) -> Result<crate::Array<String>, NumPyError> {
    let tab_width = tabsize.unwrap_or(8);
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let mut new_s = String::new();
            let mut column = 0;
            for c in s.chars() {
                if c == '\t' {
                    let spaces = tab_width - (column % tab_width);
                    for _ in 0..spaces {
                        new_s.push(' ');
                    }
                    column += spaces;
                } else {
                    new_s.push(c);
                    if c == '\n' || c == '\r' {
                        column = 0;
                    } else {
                        column += 1;
                    }
                }
            }
            result.push(new_s);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn isalnum(a: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    check_char_property(a, char::is_alphanumeric)
}

pub fn isalpha(a: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    check_char_property(a, char::is_alphabetic)
}

pub fn isdigit(a: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    check_char_property(a, char::is_numeric)
}

pub fn isnumeric(a: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    check_char_property(a, char::is_numeric)
}

pub fn isspace(a: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    check_char_property(a, char::is_whitespace)
}

fn check_char_property<F>(a: &crate::Array<String>, f: F) -> Result<crate::Array<bool>, NumPyError>
where
    F: Fn(char) -> bool,
{
    let mut result = vec![false; a.size()];
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = if s.is_empty() {
                false
            } else {
                s.chars().all(|c| f(c))
            };
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    Ok(crate::Array::from_vec(result))
}

pub fn find(
    a: &crate::Array<String>,
    sub: &str,
    start: Option<usize>,
    end: Option<usize>,
) -> Result<crate::Array<isize>, NumPyError> {
    search_string(a, sub, start, end, |s, sub| {
        s.find(sub).map(|i| i as isize).unwrap_or(-1)
    })
}

pub fn rfind(
    a: &crate::Array<String>,
    sub: &str,
    start: Option<usize>,
    end: Option<usize>,
) -> Result<crate::Array<isize>, NumPyError> {
    search_string(a, sub, start, end, |s, sub| {
        s.rfind(sub).map(|i| i as isize).unwrap_or(-1)
    })
}

pub fn index(
    a: &crate::Array<String>,
    sub: &str,
    start: Option<usize>,
    end: Option<usize>,
) -> Result<crate::Array<usize>, NumPyError> {
    let mut result = vec![0; a.size()];
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let (start, end) = adjust_indices(s.len(), start, end);
            let slice = &s[start..end];
            if let Some(i) = slice.find(sub) {
                result[idx] = start + i;
            } else {
                return Err(NumPyError::invalid_value(format!(
                    "substring '{}' not found in '{}'",
                    sub, s
                )));
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    Ok(crate::Array::from_vec(result))
}

pub fn rindex(
    a: &crate::Array<String>,
    sub: &str,
    start: Option<usize>,
    end: Option<usize>,
) -> Result<crate::Array<usize>, NumPyError> {
    let mut result = vec![0; a.size()];
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let (start, end) = adjust_indices(s.len(), start, end);
            let slice = &s[start..end];
            if let Some(i) = slice.rfind(sub) {
                result[idx] = start + i;
            } else {
                return Err(NumPyError::invalid_value(format!(
                    "substring '{}' not found in '{}'",
                    sub, s
                )));
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    Ok(crate::Array::from_vec(result))
}

pub fn count(
    a: &crate::Array<String>,
    sub: &str,
    start: Option<usize>,
    end: Option<usize>,
) -> Result<crate::Array<usize>, NumPyError> {
    let mut result = vec![0; a.size()];
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let (start, end) = adjust_indices(s.len(), start, end);
            let slice = &s[start..end];
            result[idx] = slice.matches(sub).count();
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    Ok(crate::Array::from_vec(result))
}

fn search_string<F>(
    a: &crate::Array<String>,
    sub: &str,
    start: Option<usize>,
    end: Option<usize>,
    f: F,
) -> Result<crate::Array<isize>, NumPyError>
where
    F: Fn(&str, &str) -> isize,
{
    let mut result = vec![0; a.size()];
    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let (start, end) = adjust_indices(s.len(), start, end);
            let slice = &s[start..end];
            let res = f(slice, sub);
            if res >= 0 {
                result[idx] = (start as isize) + res;
            } else {
                result[idx] = -1;
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }
    Ok(crate::Array::from_vec(result))
}

fn adjust_indices(len: usize, start: Option<usize>, end: Option<usize>) -> (usize, usize) {
    let s = start.unwrap_or(0);
    let e = end.unwrap_or(len);
    let s = s.min(len);
    let e = e.min(len);
    if s > e {
        (s, s)
    } else {
        (s, e)
    }
}

/// Check if strings contain only decimal characters
pub fn isdecimal(a: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = if s.is_empty() {
                false
            } else {
                s.chars().all(|c| c.is_ascii_digit())
            };
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Check if strings contain only lowercase characters
pub fn islower(a: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = if s.is_empty() {
                false
            } else {
                let has_cased = s.chars().any(|c| c.is_ascii_alphabetic());
                let all_lower = s.chars().all(|c| !c.is_ascii_uppercase());
                has_cased && all_lower
            };
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Check if strings contain only uppercase characters
pub fn isupper(a: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = if s.is_empty() {
                false
            } else {
                let has_cased = s.chars().any(|c| c.is_ascii_alphabetic());
                let all_upper = s.chars().all(|c| !c.is_ascii_lowercase());
                has_cased && all_upper
            };
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Check if strings are titlecased
pub fn istitle(a: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result[idx] = if s.is_empty() {
                false
            } else {
                let words: Vec<&str> = s.split_whitespace().collect();
                if words.is_empty() {
                    false
                } else {
                    words.iter().all(|word| {
                        let chars: Vec<char> = word.chars().collect();
                        if chars.is_empty() {
                            false
                        } else {
                            let first_upper = chars[0].is_uppercase() || !chars[0].is_alphabetic();
                            let rest_lower = chars.iter().skip(1).all(|c| !c.is_uppercase());
                            first_upper && rest_lower
                        }
                    })
                }
            };
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Translate strings using a translation table
pub fn translate(
    a: &crate::Array<String>,
    table: &str,
    delete_chars: Option<&str>,
) -> Result<crate::Array<String>, NumPyError> {
    let delete = delete_chars.unwrap_or("");

    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            let translated: String = s
                .chars()
                .filter(|c| !delete.contains(*c))
                .map(|c| {
                    let pos = table.chars().position(|x| x == c);
                    if let Some(p) = pos {
                        let table_chars: Vec<char> = table.chars().collect();
                        if p + 1 < table_chars.len() {
                            table_chars[p + 1]
                        } else {
                            c
                        }
                    } else {
                        c
                    }
                })
                .collect();
            result.push(translated);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// String formatting (modulo operator)
pub fn r#mod(
    a: &crate::Array<String>,
    values: &crate::Array<String>,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(format_str) = get_string(a, idx) {
            if let Some(value) = get_string(values, idx % values.size()) {
                // Simple %s substitution
                let formatted = format_str.replace("%s", &value);
                result.push(formatted);
            } else {
                return Err(NumPyError::dtype_error("Not a string array"));
            }
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Create a char array from an object (alias for identity function on String arrays)
pub fn array(obj: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    // Simply return a clone of the input array
    Ok(obj.clone())
}

/// Create a char array from an object (alias for array)
pub fn asarray(obj: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    array(obj)
}

/// Create a chararray (for compatibility, same as array in this implementation)
pub fn chararray(obj: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    array(obj)
}

/// Compare two string arrays using a comparison operator
pub fn compare_chararrays(
    a: &crate::Array<String>,
    b: &crate::Array<String>,
    cmp_op: &str,
    rstrip: Option<bool>,
) -> Result<crate::Array<bool>, NumPyError> {
    if a.shape() != b.shape() {
        return Err(NumPyError::shape_mismatch(
            a.shape().to_vec(),
            b.shape().to_vec(),
        ));
    }

    let do_rstrip = rstrip.unwrap_or(false);
    let mut result = vec![false; a.size()];

    for idx in 0..a.size() {
        if let (Some(s1), Some(s2)) = (get_string(a, idx), get_string(b, idx)) {
            let str1 = if do_rstrip { s1.trim_end() } else { &s1 };
            let str2 = if do_rstrip { s2.trim_end() } else { &s2 };

            result[idx] = match cmp_op {
                "<" => str1 < str2,
                "<=" => str1 <= str2,
                "==" => str1 == str2,
                "!=" => str1 != str2,
                ">=" => str1 >= str2,
                ">" => str1 > str2,
                _ => {
                    return Err(NumPyError::invalid_value(format!(
                        "Unknown comparison operator: {}",
                        cmp_op
                    )))
                }
            };
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

/// Decode strings (for compatibility, strings are already decoded)
pub fn decode(
    a: &crate::Array<String>,
    _encoding: Option<&str>,
) -> Result<crate::Array<String>, NumPyError> {
    // In Rust, strings are already Unicode/decoded, so we just return the array as-is
    Ok(a.clone())
}

/// Encode strings to bytes (returned as hex string for compatibility)
pub fn encode(
    a: &crate::Array<String>,
    _encoding: Option<&str>,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = Vec::with_capacity(a.size());

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            // Encode as hex string representation of UTF-8 bytes
            let hex_string: String = s.bytes().map(|b| format!("{:02x}", b)).collect();
            result.push(hex_string);
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub mod exports {
    pub use super::{
        add, array, asarray, capitalize, center, chararray, compare_chararrays, count, decode,
        encode, endswith, equal, expandtabs, find, greater, greater_equal, index, isalnum, isalpha,
        isdecimal, isdigit, islower, isnumeric, isspace, istitle, isupper, join, less, less_equal,
        ljust, lower, lstrip, multiply, not_equal, partition, r#mod, replace, rfind, rindex, rjust,
        rpartition, rsplit, rstrip, split, splitlines, startswith, str_len, strip, swapcase, title,
        translate, upper, zfill,
    };
}
