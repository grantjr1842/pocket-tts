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
    let mut result = Vec::with_capacity(a.size());

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
    let mut result = Vec::with_capacity(a.size());

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
    let mut result = Vec::with_capacity(a.size());

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

pub mod exports {
    pub use super::{
        add, capitalize, center, count, endswith, expandtabs, find, index, isalnum, isalpha,
        isdigit, isnumeric, isspace, join, lower, lstrip, lstrip_chars, multiply, replace, rfind,
        rindex, rstrip, rstrip_chars, split, startswith, strip, strip_chars, upper, zfill,
    };
}
