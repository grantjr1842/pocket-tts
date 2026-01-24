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

    let mut result = vec![String::new(); x1.size()];

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

    let mut result = vec![String::new(); a.size()];

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
    let mut result = vec![String::new(); a.size()];

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
    let mut result = vec![String::new(); a.size()];

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
    let mut result = vec![String::new(); a.size()];

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
    strip_chars(a, " \t\n\r")
}

pub fn lstrip(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    lstrip_chars(a, " \t\n\r")
}

pub fn rstrip(a: &crate::Array<String>) -> Result<crate::Array<String>, NumPyError> {
    rstrip_chars(a, " \t\n\r")
}

pub fn strip_chars(
    a: &crate::Array<String>,
    chars: &str,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_matches(|c| chars.contains(c)).to_string());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn lstrip_chars(
    a: &crate::Array<String>,
    chars: &str,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_start_matches(|c| chars.contains(c)).to_string());
        } else {
            return Err(NumPyError::dtype_error("Not a string array"));
        }
    }

    Ok(crate::Array::from_vec(result))
}

pub fn rstrip_chars(
    a: &crate::Array<String>,
    chars: &str,
) -> Result<crate::Array<String>, NumPyError> {
    let mut result = vec![String::new(); a.size()];

    for idx in 0..a.size() {
        if let Some(s) = get_string(a, idx) {
            result.push(s.trim_end_matches(|c| chars.contains(c)).to_string());
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
    let mut result = vec![String::new(); a.size()];

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
    let mut result = vec![String::new(); a.size()];

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
pub fn ljust(a: &crate::Array<String>, width: usize, fillchar: Option<char>) -> Result<crate::Array<String>, NumPyError> {
    let fill = fillchar.unwrap_or(' ');
    if !fill.is_ascii() || fill.is_ascii_control() {
        return Err(NumPyError::invalid_value("fillchar must be a non-control ASCII character"));
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
pub fn rjust(a: &crate::Array<String>, width: usize, fillchar: Option<char>) -> Result<crate::Array<String>, NumPyError> {
    let fill = fillchar.unwrap_or(' ');
    if !fill.is_ascii() || fill.is_ascii_control() {
        return Err(NumPyError::invalid_value("fillchar must be a non-control ASCII character"));
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
            let swapped: String = s.chars()
                .map(|c| {
                    if c.is_uppercase() {
                        c.to_lowercase().collect::<String>().chars().next().unwrap_or(c)
                    } else if c.is_lowercase() {
                        c.to_uppercase().collect::<String>().chars().next().unwrap_or(c)
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
            let titled: String = s.split_whitespace()
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
pub fn rsplit(a: &crate::Array<String>, sep: &str, maxsplit: Option<isize>) -> Result<crate::Array<String>, NumPyError> {
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
pub fn partition(a: &crate::Array<String>, sep: &str) -> Result<crate::Array<Vec<String>>, NumPyError> {
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
pub fn rpartition(a: &crate::Array<String>, sep: &str) -> Result<crate::Array<Vec<String>>, NumPyError> {
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
pub fn splitlines(a: &crate::Array<String>, keepends: Option<bool>) -> Result<crate::Array<Vec<String>>, NumPyError> {
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
pub fn equal(a: &crate::Array<String>, b: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 == s2)
}

/// String comparison: not equal
pub fn not_equal(a: &crate::Array<String>, b: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 != s2)
}

/// String comparison: greater (lexicographic)
pub fn greater(a: &crate::Array<String>, b: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 > s2)
}

/// String comparison: greater or equal
pub fn greater_equal(a: &crate::Array<String>, b: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 >= s2)
}

/// String comparison: less (lexicographic)
pub fn less(a: &crate::Array<String>, b: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
    compare_arrays(a, b, |s1, s2| s1 < s2)
}

/// String comparison: less or equal
pub fn less_equal(a: &crate::Array<String>, b: &crate::Array<String>) -> Result<crate::Array<bool>, NumPyError> {
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

pub mod exports {
    pub use super::{
        add, capitalize, endswith, equal, greater, greater_equal, join, less, less_equal,
        ljust, lower, lstrip, lstrip_chars, multiply, not_equal, partition, replace, rjust,
        rpartition, rsplit, rstrip, rstrip_chars, split, splitlines, startswith, str_len, strip,
        strip_chars, swapcase, title, upper,
    };
}
