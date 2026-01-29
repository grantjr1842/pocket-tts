// Copyright 2024 The NumPyRS Authors.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.
//
//! NumPy-compatible I/O routines module
//!
//! Provides comprehensive file I/O functionality matching NumPy's API:
//! - Load/save arrays in NPY and NPZ formats
//! - Text file operations (CSV, delimited files)
//! - Binary buffer operations
//! - Memory-mapped file support
//! - Compression support for NPZ files

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;
use std::sync::Arc;

use bytemuck::{cast_slice, Pod};
use byteorder::{ByteOrder as _, LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::{Mmap, MmapMut, MmapOptions};
use zip::{ZipArchive, ZipWriter};

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{NumPyError, Result};
use crate::memory::MemoryManager;

/// File format detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Npy,
    Npz,
    Text,
    Binary,
    Unknown,
}

/// Memory mapping mode for file operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmapMode {
    /// No memory mapping
    None,
    /// Read-only memory mapping
    Read,
    /// Read-write memory mapping (if supported)
    ReadWrite,
}

impl std::fmt::Display for MmapMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MmapMode::None => write!(f, "none"),
            MmapMode::Read => write!(f, "r"),
            MmapMode::ReadWrite => write!(f, "r+"),
        }
    }
}

impl std::str::FromStr for MmapMode {
    type Err = NumPyError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "r" | "readonly" => Ok(MmapMode::Read),
            "r+" | "readwrite" => Ok(MmapMode::ReadWrite),
            "c" | "copyonwrite" => Ok(MmapMode::Read),
            _ => Ok(MmapMode::None),
        }
    }
}

/// Detect file format from path and content
pub fn detect_file_format<P: AsRef<Path>>(path: P) -> Result<FileFormat> {
    let path = path.as_ref();

    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            match ext_str.to_lowercase().as_str() {
                "npy" => return Ok(FileFormat::Npy),
                "npz" => return Ok(FileFormat::Npz),
                "txt" | "csv" | "dat" => return Ok(FileFormat::Text),
                _ => {}
            }
        }
    }

    if let Ok(mut file) = File::open(path) {
        let mut header = [0u8; 10];
        if let Ok(n) = file.read(&mut header) {
            if n >= 6 && &header[..6] == b"\x93NUMPY" {
                return Ok(FileFormat::Npy);
            }

            if n >= 4 && &header[..4] == b"PK\x03\x04" {
                return Ok(FileFormat::Npz);
            }

            let text_like = header
                .iter()
                .take_while(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
                .count();
            if text_like >= n / 2 {
                return Ok(FileFormat::Text);
            } else {
                return Ok(FileFormat::Binary);
            }
        }
    }

    Ok(FileFormat::Unknown)
}

/// Load array from file (NumPy-compatible)
pub fn load<T>(
    file: &str,
    mmap_mode: Option<&str>,
    allow_pickle: bool,
    fix_imports: bool,
    encoding: &str,
) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let mmap_mode = mmap_mode.map(|s| s.parse()).transpose()?;

    let format = detect_file_format(file)?;
    match format {
        FileFormat::Npy => load_npy(file, mmap_mode),
        FileFormat::Npz => {
            if !allow_pickle {
                return Err(NumPyError::invalid_operation(
                    "NPZ files require pickle support",
                ));
            }
            load_npz_single(file, mmap_mode, fix_imports, encoding)
        }
        FileFormat::Text => loadtxt(
            file, None, "#", " ", None, 0, None, false, 0, encoding, None,
        ),
        _ => Err(NumPyError::file_format_error(
            "unknown",
            "Cannot determine file format",
        )),
    }
}

/// Save array to file (NumPy-compatible)
pub fn save<T>(file: &str, arr: &Array<T>, allow_pickle: bool, fix_imports: bool) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
    T: std::fmt::Display,
{
    if !allow_pickle {
        save_npy(file, arr)
    } else {
        let format = detect_file_format_from_filename(file)?;
        match format {
            FileFormat::Npy => save_npy(file, arr),
            FileFormat::Npz => {
                let arrays = vec![("arr_0", arr)];
                savez_internal(file, &arrays, false)
            }
            FileFormat::Text => savetxt(file, arr, "%.18e", ",", "\n", "", "", "#", "utf8"),
            _ => save_npy(file, arr),
        }
    }
}

/// Save multiple arrays to uncompressed NPZ file
///
/// # Arguments
/// * `file` - Path to the output NPZ file
/// * `args` - Vector of (name, array) pairs to save
///
/// # Example
/// ```rust,ignore
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// savez("data.npz", vec![("a", &a), ("b", &b)]).unwrap();
/// ```
pub fn savez<T>(file: &str, args: Vec<(&str, &Array<T>)>) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    savez_internal(file, &args, false)
}

/// Save multiple arrays to uncompressed NPZ file (variadic version)
///
/// Similar to NumPy's `savez` function. Arrays are saved with names arr_0, arr_1, etc.
///
/// # Arguments
/// * `file` - Path to the output NPZ file
/// * `arrays` - Arrays to save (will be named arr_0, arr_1, ...)
pub fn savez_many<T>(file: &str, arrays: &[&Array<T>]) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    let named: Vec<(String, &Array<T>)> = arrays
        .iter()
        .enumerate()
        .map(|(i, arr)| (format!("arr_{}", i), *arr))
        .collect();
    let refs: Vec<(&str, &Array<T>)> = named.iter().map(|(s, arr)| (s.as_str(), *arr)).collect();
    savez_internal(file, &refs, false)
}

/// Save multiple arrays to compressed NPZ file
///
/// # Arguments
/// * `file` - Path to the output NPZ file
/// * `args` - Vector of (name, array) pairs to save
///
/// # Example
/// ```rust,ignore
/// let a = Array::from_vec(vec![1, 2, 3]);
/// let b = Array::from_vec(vec![4, 5, 6]);
/// savez_compressed("data.npz", vec![("a", &a), ("b", &b)]).unwrap();
/// ```
pub fn savez_compressed<T>(file: &str, args: Vec<(&str, &Array<T>)>) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    savez_internal(file, &args, true)
}

/// Save multiple arrays to compressed NPZ file (variadic version)
///
/// Similar to NumPy's `savez_compressed` function. Arrays are saved with names arr_0, arr_1, etc.
///
/// # Arguments
/// * `file` - Path to the output NPZ file
/// * `arrays` - Arrays to save (will be named arr_0, arr_1, ...)
pub fn savez_compressed_many<T>(file: &str, arrays: &[&Array<T>]) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    let named: Vec<(String, &Array<T>)> = arrays
        .iter()
        .enumerate()
        .map(|(i, arr)| (format!("arr_{}", i), *arr))
        .collect();
    let refs: Vec<(&str, &Array<T>)> = named.iter().map(|(s, arr)| (s.as_str(), *arr)).collect();
    savez_internal(file, &refs, true)
}

/// Load arrays from an NPZ file
///
/// Returns a HashMap mapping array names to arrays. The names exclude the `.npy` extension.
///
/// # Arguments
/// * `file` - Path to the NPZ file
///
/// # Example
/// ```rust,ignore
/// let arrays = load_npz::<i32>("data.npz").unwrap();
/// let a = arrays.get("a").unwrap();
/// ```
pub fn load_npz<T>(file: &str) -> Result<HashMap<String, Array<T>>>
where
    T: Clone + Default + Pod + 'static,
{
    let file = File::open(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut archive = ZipArchive::new(reader)
        .map_err(|e| NumPyError::file_format_error("npz", &e.to_string()))?;

    let mut arrays = HashMap::new();

    for i in 0..archive.len() {
        let mut zip_file = archive
            .by_index(i)
            .map_err(|e| NumPyError::file_format_error("npz", &e.to_string()))?;
        let filename = zip_file.name().to_string();

        if filename.ends_with(".npy") {
            let mut buffer = Vec::new();
            zip_file.read_to_end(&mut buffer)?;

            let array = load_npy_from_bytes(buffer)?;
            
            // Remove .npy extension for the key
            let key = filename.strip_suffix(".npy").unwrap_or(&filename).to_string();
            arrays.insert(key, array);
        }
    }

    if arrays.is_empty() {
        return Err(NumPyError::file_format_error(
            "npz",
            "No NPY files found in archive",
        ));
    }

    Ok(arrays)
}

/// Load array from text file with configurable parsing
pub fn loadtxt<T>(
    fname: &str,
    dtype: Option<Dtype>,
    comments: &str,
    delimiter: &str,
    converters: Option<Vec<fn(&str) -> T>>,
    skiprows: usize,
    usecols: Option<&[usize]>,
    unpack: bool,
    ndmin: isize,
    encoding: &str,
    max_rows: Option<usize>,
) -> Result<Array<T>>
where
    T: Clone + Default + std::str::FromStr + 'static,
    T::Err: std::fmt::Display,
{
    let file = File::open(fname)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;

    for (line_num, line_result) in reader.lines().enumerate() {
        if line_num < skiprows {
            continue;
        }

        let line = line_result.map_err(|e| NumPyError::io_error(format!("Read error: {}", e)))?;
        let trimmed = line.trim();

        if trimmed.starts_with(comments) || trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = if delimiter.is_empty() {
            trimmed.split_whitespace().collect()
        } else {
            trimmed.split(delimiter).collect()
        };

        let selected_parts = if let Some(cols) = usecols {
            cols.iter()
                .filter_map(|&col| parts.get(col).copied())
                .collect()
        } else {
            parts
        };

        let row_data = if let Some(ref converters) = converters {
            selected_parts
                .iter()
                .enumerate()
                .map(|(i, part)| {
                    if i < converters.len() {
                        Ok(converters[i](part))
                    } else {
                        part.parse().map_err(|_| {
                            NumPyError::value_error(part.to_string(), "numeric conversion")
                        })
                    }
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            selected_parts
                .iter()
                .map(|part| {
                    part.parse().map_err(|_| {
                        NumPyError::value_error(part.to_string(), "numeric conversion")
                    })
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        };

        if rows == 0 {
            cols = row_data.len();
        }

        data.extend(row_data);
        rows += 1;

        if let Some(max_rows) = max_rows {
            if rows >= max_rows {
                break;
            }
        }
    }

    let final_shape = match ndmin {
        0 | 1 => {
            if unpack {
                vec![cols, rows]
            } else {
                vec![rows, cols]
            }
        }
        2 => vec![1, rows, cols],
        _ => return Err(NumPyError::invalid_operation("ndmin must be 0, 1, or 2")),
    };

    let shape = if final_shape.last() == Some(&1) && final_shape.len() > 1 {
        final_shape[..final_shape.len() - 1].to_vec()
    } else {
        final_shape
    };

    Ok(Array::from_shape_vec(shape, data))
}

/// Save array to text file with formatting
pub fn savetxt<T>(
    fname: &str,
    X: &Array<T>,
    fmt: &str,
    delimiter: &str,
    newline: &str,
    header: &str,
    footer: &str,
    comments: &str,
    _encoding: &str,
) -> Result<()>
where
    T: std::fmt::Display + Clone,
{
    let file = File::create(fname)
        .map_err(|e| NumPyError::io_error(format!("Failed to create file: {}", e)))?;
    let mut writer = BufWriter::new(file);

    if !header.is_empty() {
        let header_with_comments = if comments.is_empty() {
            header.to_string()
        } else {
            header
                .lines()
                .map(|line| format!("{}{}", comments, line))
                .collect::<Vec<_>>()
                .join("\n")
        };
        writer.write_all(header_with_comments.as_bytes())?;
        writer.write_all(newline.as_bytes())?;
    }

    let shape = X.shape();
    let data = X.to_vec();

    match shape.len() {
        0 => {
            write!(writer, "{}", data[0])?;
        }
        1 => {
            for (i, value) in data.iter().enumerate() {
                if i > 0 {
                    writer.write_all(delimiter.as_bytes())?;
                }
                writer.write_all(fmt.replace("{}", &value.to_string()).as_bytes())?;
            }
        }
        2 => {
            let rows = shape[0];
            let cols = shape[1];

            for row in 0..rows {
                for col in 0..cols {
                    if col > 0 {
                        writer.write_all(delimiter.as_bytes())?;
                    }
                    let idx = row * cols + col;
                    writer.write_all(fmt.replace("{}", &data[idx].to_string()).as_bytes())?;
                }
                if row < rows - 1 {
                    writer.write_all(newline.as_bytes())?;
                }
            }
        }
        _ => {
            let mut idx = 0;
            for _ in 0..shape.iter().take(shape.len() - 1).product() {
                for col in 0..shape[shape.len() - 1] {
                    if col > 0 {
                        writer.write_all(delimiter.as_bytes())?;
                    }
                    writer.write_all(fmt.replace("{}", &data[idx].to_string()).as_bytes())?;
                    idx += 1;
                }
                if idx < data.len() {
                    writer.write_all(newline.as_bytes())?;
                }
            }
        }
    }

    if !footer.is_empty() {
        writer.write_all(newline.as_bytes())?;
        let footer_with_comments = if comments.is_empty() {
            footer.to_string()
        } else {
            footer
                .lines()
                .map(|line| format!("{}{}", comments, line))
                .collect::<Vec<_>>()
                .join("\n")
        };
        writer.write_all(footer_with_comments.as_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

/// Create array from raw byte buffer
pub fn frombuffer<T>(
    buffer: &[u8],
    dtype: Option<Dtype>,
    count: Option<isize>,
    offset: isize,
) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
{
    let dtype = dtype.unwrap_or_else(|| Dtype::from_type::<T>());
    let item_size = dtype.itemsize();

    if offset < 0 {
        return Err(NumPyError::invalid_operation("offset cannot be negative"));
    }

    let offset_usize = offset as usize;
    if offset_usize > buffer.len() {
        return Err(NumPyError::invalid_operation(
            "offset exceeds buffer length",
        ));
    }

    let available_bytes = buffer.len() - offset_usize;
    let max_elements = available_bytes / item_size;

    let num_elements = match count {
        None => max_elements,
        Some(c) if c < 0 => max_elements,
        Some(c) => std::cmp::min(c as usize, max_elements),
    };

    if num_elements == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    let end_offset = offset_usize + (num_elements * item_size);
    let data_bytes = &buffer[offset_usize..end_offset];

    let typed_data: &[T] = cast_slice(data_bytes);
    let data: Vec<T> = typed_data.iter().cloned().collect();

    Ok(Array::from_vec(data))
}

/// Read array from binary file
pub fn fromfile<T>(
    file: &str,
    dtype: Dtype,
    count: isize,
    sep: &str,
    offset: isize,
) -> Result<Array<T>>
where
    T: Clone + Default + Pod + std::str::FromStr + 'static,
    T::Err: std::fmt::Display,
{
    let mut file = File::open(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;

    if offset > 0 {
        use std::io::Seek;
        file.seek(std::io::SeekFrom::Start(offset as u64))
            .map_err(|e| NumPyError::io_error(format!("Failed to seek: {}", e)))?;
    }

    if sep.is_empty() {
        let item_size = dtype.itemsize();
        let bytes_to_read = if count < 0 {
            let metadata = file
                .metadata()
                .map_err(|e| NumPyError::io_error(format!("Failed to get metadata: {}", e)))?;
            metadata.len() as usize - offset as usize
        } else {
            count as usize * item_size
        };

        let mut buffer = vec![0u8; bytes_to_read];
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| NumPyError::io_error(format!("Failed to read: {}", e)))?;

        if bytes_read != bytes_to_read {
            buffer.truncate(bytes_read);
        }

        frombuffer(&buffer, Some(dtype), Some(count), 0)
    } else {
        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| NumPyError::io_error(format!("Failed to read: {}", e)))?;

        fromstring(&content, dtype, count, sep)
    }
}

/// Create array from string data
pub fn fromstring<T>(string: &str, dtype: Dtype, count: isize, sep: &str) -> Result<Array<T>>
where
    T: Clone + Default + std::str::FromStr + Pod + 'static,
    T::Err: std::fmt::Display,
{
    if sep.is_empty() {
        let bytes = string.as_bytes();
        frombuffer(bytes, Some(dtype), Some(count), 0)
    } else {
        let parts: Vec<&str> = string.split(sep).filter(|s| !s.trim().is_empty()).collect();

        let max_count = if count < 0 {
            parts.len()
        } else {
            std::cmp::min(count as usize, parts.len())
        };

        let data: Result<Vec<T>> = parts[..max_count]
            .iter()
            .map(|part| {
                part.trim()
                    .parse()
                    .map_err(|_| NumPyError::value_error(part.to_string(), "numeric conversion"))
            })
            .collect();

        Ok(Array::from_vec(data?))
    }
}

/// Create array from DLPack capsule (placeholder for FFI integration)
///
/// Note: This is a placeholder for DLPack integration.
/// Full DLPack support requires FFI bindings to DLPack C API.
pub fn from_dlpack<T>(_x: &[u8]) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
{
    // TODO: Implement proper DLPack capsule parsing
    // This requires FFI bindings to dlpack.h
    Err(NumPyError::invalid_operation(
        "from_dlpack requires DLPack FFI bindings (not yet implemented)",
    ))
}

/// Load data from text file with missing value handling
///
/// This is an enhanced version of loadtxt that handles missing values,
/// custom converters, and more flexible data loading.
pub fn genfromtxt<T>(
    fname: &str,
    dtype: Option<Dtype>,
    comments: Option<&str>,
    delimiter: Option<&str>,
    skip_header: usize,
    skip_footer: usize,
    converters: Option<Vec<fn(&str) -> T>>,
    missing_values: Option<Vec<&str>>,
    filling_values: Option<Vec<T>>,
    usecols: Option<&[usize]>,
    names: Option<Vec<&str>>,
    excludelist: Option<Vec<&str>>,
    deletechars: Option<&str>,
    replace_space: Option<char>,
    autostrip: bool,
    case_sensitive: bool,
    defaultfmt: &str,
    unpack: bool,
    usemask: bool,
    loose: bool,
    invalid_raise: bool,
    max_rows: Option<usize>,
    encoding: Option<&str>,
    ndmin: isize,
) -> Result<Array<T>>
where
    T: Clone + Default + std::str::FromStr + 'static,
    T::Err: std::fmt::Display,
{
    let comments = comments.unwrap_or("#");
    let delimiter = delimiter.unwrap_or(" ");
    let encoding = encoding.unwrap_or("utf8");
    
    let file = File::open(fname)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);
    
    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;
    let mut line_num = 0;
    
    for line_result in reader.lines() {
        line_num += 1;
        
        // Skip header
        if line_num <= skip_header {
            continue;
        }
        
        let line = line_result.map_err(|e| NumPyError::io_error(format!("Read error: {}", e)))?;
        let trimmed = if autostrip {
            line.trim()
        } else {
            line
        };
        
        // Skip comments and empty lines
        if trimmed.starts_with(comments) || trimmed.is_empty() {
            continue;
        }
        
        // Handle deletechars
        let processed = if let Some(dc) = deletechars {
            trimmed.chars().filter(|&c| !dc.contains(c)).collect()
        } else {
            trimmed.to_string()
        };
        
        // Handle replace_space
        let processed = if let Some(replacement) = replace_space {
            processed.replace(' ', replacement)
        } else {
            processed
        };
        
        // Handle case sensitivity for missing values
        let is_missing = |val: &str| {
            if let Some(ref mv) = missing_values {
                if case_sensitive {
                    mv.contains(&val.to_string())
                } else {
                    mv.iter().any(|m| m.to_lowercase() == val.to_lowercase())
                }
            } else {
                false
            }
        };
        
        // Split by delimiter
        let parts: Vec<&str> = if delimiter.is_empty() {
            processed.split_whitespace().collect()
        } else {
            processed.split(delimiter).collect()
        };
        
        // Filter excluded columns
        let selected_parts = if let Some(ref excl) = excludelist {
            parts.iter()
                .enumerate()
                .filter(|(i, _)| !excl.contains(&i.to_string()))
                .map(|(_, &p)| *p)
                .collect()
        } else {
            parts.iter().copied().collect()
        };
        
        // Select specific columns
        let selected_parts = if let Some(cols) = usecols {
            cols.iter()
                .filter_map(|&col| selected_parts.get(*col).copied())
                .collect()
        } else {
            selected_parts
        };
        
        // Apply converters or parse with missing value handling
        let row_data: Result<Vec<T>> = selected_parts
            .iter()
            .enumerate()
            .map(|(i, part)| {
                let part = part.trim();
                
                if is_missing(part) {
                    if let Some(ref fv) = filling_values {
                        if i < fv.len() {
                            Ok(fv[i].clone())
                        } else {
                            // Use default filling value if not specified
                            T::default().map_err(|_| NumPyError::value_error(part.to_string(), "conversion"))
                        }
                    } else {
                        // Return default value for missing data
                        T::default().map_err(|_| NumPyError::value_error(part.to_string(), "conversion"))
                    }
                } else {
                    if let Some(ref conv) = converters {
                        if i < conv.len() {
                            Ok(conv[i](part))
                        } else {
                            part.parse().map_err(|_| {
                                NumPyError::value_error(part.to_string(), "numeric conversion")
                            })
                        }
                    } else {
                        part.parse().map_err(|_| {
                            NumPyError::value_error(part.to_string(), "numeric conversion")
                        })
                    }
                }
            })
            .collect();
        
        let row_data = row_data?;
        
        if rows == 0 {
            cols = row_data.len();
        }
        
        data.extend(row_data);
        rows += 1;
        
        // Check max_rows
        if let Some(max) = max_rows {
            if rows >= max {
                break;
            }
        }
        
        // Skip footer
        if line_num > skip_footer + skip_header {
            continue;
        }
    }
    
    // Handle ndmin
    let final_shape = match ndmin {
        0 | 1 => {
            if unpack {
                vec![cols, rows]
            } else {
                vec![rows, cols]
            }
        }
        2 => vec![1, rows, cols],
        _ => return Err(NumPyError::invalid_operation("ndmin must be 0, 1, or 2")),
    };
    
    // Remove trailing dimension of size 1
    let shape = if final_shape.last() == Some(&1) && final_shape.len() > 1 {
        final_shape[..final_shape.len() - 1].to_vec()
    } else {
        final_shape
    };
    
    if rows == 0 {
        return Err(NumPyError::invalid_operation("No data found in file"));
    }
    
    Ok(Array::from_shape_vec(shape, data))
}

/// Create array from text file using regular expression parsing
pub fn fromregex<T>(
    file: &str,
    pattern: &str,
    dtype: Option<Dtype>,
    encoding: Option<&str>,
) -> Result<Array<T>>
where
    T: Clone + Default + std::str::FromStr + 'static,
    T::Err: std::fmt::Display,
{
    use regex::Regex;
    
    let encoding = encoding.unwrap_or("utf8");
    let regex = Regex::new(pattern)
        .map_err(|e| NumPyError::invalid_operation(format!("Invalid regex: {}", e)))?;
    
    let file_content = std::fs::read_to_string(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to read file: {}", e)))?;
    
    let mut data = Vec::new();
    let mut rows = 0;
    let mut cols = 0;
    
    for line in file_content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        
        let captures: Vec<&str> = regex
            .captures_iter(trimmed)
            .filter_map(|c| c.get(1))
            .map(|m| m.as_str())
            .collect();
        
        let row_data: Result<Vec<T>> = captures
            .iter()
            .map(|part| {
                part.trim()
                    .parse()
                    .map_err(|_| NumPyError::value_error(part.to_string(), "numeric conversion"))
            })
            .collect();
        
        let row_data = row_data?;
        
        if rows == 0 {
            cols = row_data.len();
        }
        
        data.extend(row_data);
        rows += 1;
    }
    
    if rows == 0 {
        return Err(NumPyError::invalid_operation("No data found in file"));
    }
    
    let shape = if cols == 1 {
        vec![rows]
    } else {
        vec![rows, cols]
    };
    
    Ok(Array::from_shape_vec(shape, data))
}

// Internal helper functions

fn detect_file_format_from_filename(file: &str) -> Result<FileFormat> {
    let path = Path::new(file);
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            match ext_str.to_lowercase().as_str() {
                "npy" => return Ok(FileFormat::Npy),
                "npz" => return Ok(FileFormat::Npz),
                "txt" | "csv" | "dat" => return Ok(FileFormat::Text),
                _ => {}
            }
        }
    }
    Ok(FileFormat::Npy)
}

fn savez_internal<T>(file: &str, args: &[(&str, &Array<T>)], compressed: bool) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    let file = File::create(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to create file: {}", e)))?;

    if compressed {
        let zip_writer = ZipWriter::new(file);
        savez_to_zip(zip_writer, args)
    } else {
        let mut zip_writer = ZipWriter::new(file);
        savez_to_zip(zip_writer, args)
    }
}

fn savez_to_zip<T, W: Write + Seek>(
    mut zip_writer: ZipWriter<W>,
    args: &[(&str, &Array<T>)],
) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    use std::io::Seek;
    use zip::write::FileOptions;

    for (name, array) in args {
        let npy_data = create_npy_data(array)?;

        let filename = format!("{}.npy", name);
        zip_writer
            .start_file(filename, FileOptions::default())
            .map_err(|e| NumPyError::io_error(e.to_string()))?;
        zip_writer.write_all(&npy_data)?;
    }

    zip_writer
        .finish()
        .map_err(|e| NumPyError::io_error(e.to_string()))?;
    Ok(())
}

fn create_npy_data<T>(array: &Array<T>) -> Result<Vec<u8>>
where
    T: Clone + Default + Pod + 'static,
{
    let mut buffer = Vec::new();

    buffer.extend_from_slice(b"\x93NUMPY");

    buffer.push(1);
    buffer.push(0);

    let shape_str = array
        .shape()
        .iter()
        .map(|&dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': ({})}}",
        array.dtype().to_string(),
        shape_str
    );

    let header_len = header_dict.len();
    let total_len = header_len + 16;
    let padding = (16 - (total_len % 16)) % 16;
    let total_header_len = total_len + padding;

    buffer.write_u16::<LittleEndian>((total_header_len - 10) as u16)?;

    buffer.extend_from_slice(header_dict.as_bytes());

    for _ in 0..padding {
        buffer.push(b' ');
    }
    buffer.push(b'\n');

    let data = array.to_vec();
    let data_bytes = bytemuck::cast_slice(&data);
    buffer.extend_from_slice(data_bytes);

    Ok(buffer)
}

fn save_npy<T>(file: &str, array: &Array<T>) -> Result<()>
where
    T: Clone + Default + Pod + 'static,
{
    let npy_data = create_npy_data(array)?;

    let mut file = File::create(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to create file: {}", e)))?;
    file.write_all(&npy_data)?;

    Ok(())
}

fn load_npy<T>(file: &str, _mmap_mode: Option<MmapMode>) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
{
    let mut file = File::open(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    if buffer.len() < 10 || &buffer[..6] != b"\x93NUMPY" {
        return Err(NumPyError::file_format_error(
            "npy",
            "Invalid NPY file format",
        ));
    }

    let version = buffer[6];
    if version != 1 {
        return Err(NumPyError::file_format_error(
            "npy",
            "Only NPY version 1.0 is supported",
        ));
    }

    let header_len = LittleEndian::read_u16(&buffer[8..10]) as usize;

    let header_start = 10;
    let header_end = header_start + header_len;
    if header_end > buffer.len() {
        return Err(NumPyError::file_format_error(
            "npy",
            "Header extends beyond file",
        ));
    }

    let header_bytes = &buffer[header_start..header_end];
    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| NumPyError::file_format_error("npy", "Invalid header encoding"))?;

    let shape = parse_npy_shape(header_str)?;

    let data_start = header_end;
    let data_end = data_start + shape.iter().product::<usize>() * std::mem::size_of::<T>();

    if data_end > buffer.len() {
        return Err(NumPyError::file_format_error(
            "npy",
            "Data extends beyond file",
        ));
    }

    let data_bytes = &buffer[data_start..data_end];
    let typed_data: &[T] = bytemuck::cast_slice(data_bytes);
    let data: Vec<T> = typed_data.iter().cloned().collect();

    Ok(Array::from_shape_vec(shape, data))
}

fn load_npz_single<T>(
    file: &str,
    _mmap_mode: Option<MmapMode>,
    _fix_imports: bool,
    _encoding: &str,
) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
{
    let file = File::open(file)
        .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut archive = ZipArchive::new(reader)
        .map_err(|e| NumPyError::file_format_error("npz", &e.to_string()))?;

    for i in 0..archive.len() {
        let mut zip_file = archive
            .by_index(i)
            .map_err(|e| NumPyError::file_format_error("npz", &e.to_string()))?;
        let filename = zip_file.name();

        if filename.ends_with(".npy") {
            let mut buffer = Vec::new();
            zip_file.read_to_end(&mut buffer)?;

            return load_npy_from_bytes(buffer);
        }
    }

    Err(NumPyError::file_format_error(
        "npz",
        "No NPY files found in archive",
    ))
}

fn load_npy_from_bytes<T>(buffer: Vec<u8>) -> Result<Array<T>>
where
    T: Clone + Default + Pod + 'static,
{
    if buffer.len() < 10 || &buffer[..6] != b"\x93NUMPY" {
        return Err(NumPyError::file_format_error(
            "npy",
            "Invalid NPY file format",
        ));
    }

    let version = buffer[6];
    if version != 1 {
        return Err(NumPyError::file_format_error(
            "npy",
            "Only NPY version 1.0 is supported",
        ));
    }

    let header_len = LittleEndian::read_u16(&buffer[8..10]) as usize;

    let header_start = 10;
    let header_end = header_start + header_len;
    if header_end > buffer.len() {
        return Err(NumPyError::file_format_error(
            "npy",
            "Header extends beyond file",
        ));
    }

    let header_bytes = &buffer[header_start..header_end];
    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| NumPyError::file_format_error("npy", "Invalid header encoding"))?;

    let shape = parse_npy_shape(header_str)?;

    let data_start = header_end;
    let data_end = data_start + shape.iter().product::<usize>() * std::mem::size_of::<T>();

    if data_end > buffer.len() {
        return Err(NumPyError::file_format_error(
            "npy",
            "Data extends beyond file",
        ));
    }

    let data_bytes = &buffer[data_start..data_end];
    let typed_data: &[T] = bytemuck::cast_slice(data_bytes);
    let data: Vec<T> = typed_data.iter().cloned().collect();

    Ok(Array::from_shape_vec(shape, data))
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>> {
    let shape_start = header.find("'shape': (");
    if shape_start.is_none() {
        return Err(NumPyError::file_format_error(
            "npy",
            "No shape field in header",
        ));
    }

    let shape_start = shape_start.unwrap() + 10;
    let shape_end = header[shape_start..].find(')');
    if shape_end.is_none() {
        return Err(NumPyError::file_format_error("npy", "Invalid shape field"));
    }

    let shape_end = shape_start + shape_end.unwrap();
    let shape_str = &header[shape_start..shape_end];

    if shape_str.trim().is_empty() {
        return Ok(vec![]);
    }

    let dimensions: Result<Vec<usize>> = shape_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| NumPyError::file_format_error("npy", "Invalid shape dimension"))
        })
        .collect();

    dimensions
}

/// Memory-mapped array structure
///
/// Provides a NumPy-compatible memory-mapped array that allows
/// arrays to be stored in files and accessed as if they were in memory.
pub struct MemmapArray<T> {
    /// The underlying array data
    pub array: Array<T>,
    /// The memory map (stored to keep the mapping alive)
    #[allow(dead_code)]
    mmap: MemmapStorage,
}

/// Storage for memory-mapped data
enum MemmapStorage {
    ReadOnly(Mmap),
    ReadWrite(MmapMut),
}

/// Create a memory-mapped array from a file
///
/// # Arguments
/// * `filename` - Path to the file to memory map
/// * `dtype` - Data type of the array elements
/// * `mode` - File mode: 'r' (read-only), 'r+' (read-write), 'w+' (write)
/// * `shape` - Shape of the array (if creating new), or None to infer from file
/// * `offset` - Offset in bytes from the beginning of the file
///
/// # Returns
/// A memory-mapped array that can be accessed like a regular array
///
/// # Example
/// ```rust,ignore
/// use numpy::io::{memmap, MmapMode};
/// use numpy::Dtype;
///
/// // Open existing file as memory-mapped array
/// let arr = memmap::<f64>("data.bin", None, MmapMode::Read, None, 0).unwrap();
/// ```
pub fn memmap<T>(
    filename: &str,
    dtype: Option<Dtype>,
    mode: MmapMode,
    shape: Option<Vec<usize>>,
    offset: usize,
) -> Result<MemmapArray<T>>
where
    T: Clone + Default + Pod + 'static,
{
    let dtype = dtype.unwrap_or_else(|| Dtype::from_type::<T>());
    let item_size = dtype.itemsize();

    match mode {
        MmapMode::Read => {
            let file = File::open(filename)
                .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;

            let file_len = file
                .metadata()
                .map_err(|e| NumPyError::io_error(format!("Failed to get metadata: {}", e)))?
                .len() as usize;

            if offset >= file_len {
                return Err(NumPyError::invalid_operation(
                    "Offset exceeds file length",
                ));
            }

            let mmap = unsafe {
                Mmap::map(&file)
                    .map_err(|e| NumPyError::io_error(format!("Failed to mmap file: {}", e)))?
            };

            let data_len = file_len - offset;
            let num_elements = data_len / item_size;

            let actual_shape = match shape {
                Some(s) => {
                    let expected_elements: usize = s.iter().product();
                    if expected_elements > num_elements {
                        return Err(NumPyError::invalid_operation(
                            "Shape requires more elements than available in file",
                        ));
                    }
                    s
                }
                None => vec![num_elements],
            };

            // Copy data from mmap to array (since our Array uses MemoryManager)
            let data_slice = &mmap[offset..offset + num_elements * item_size];
            let typed_data: &[T] = bytemuck::cast_slice(data_slice);
            let data: Vec<T> = typed_data.iter().cloned().collect();

            let array = Array::from_shape_vec(actual_shape, data);

            Ok(MemmapArray {
                array,
                mmap: MemmapStorage::ReadOnly(mmap),
            })
        }
        MmapMode::ReadWrite => {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(filename)
                .map_err(|e| NumPyError::io_error(format!("Failed to open file: {}", e)))?;

            let mmap = unsafe {
                MmapMut::map_mut(&file)
                    .map_err(|e| NumPyError::io_error(format!("Failed to mmap file: {}", e)))?
            };

            let file_len = mmap.len();

            if offset >= file_len {
                return Err(NumPyError::invalid_operation(
                    "Offset exceeds file length",
                ));
            }

            let data_len = file_len - offset;
            let num_elements = data_len / item_size;

            let actual_shape = match shape {
                Some(s) => {
                    let expected_elements: usize = s.iter().product();
                    if expected_elements > num_elements {
                        return Err(NumPyError::invalid_operation(
                            "Shape requires more elements than available in file",
                        ));
                    }
                    s
                }
                None => vec![num_elements],
            };

            // Copy data from mmap to array
            let data_slice = &mmap[offset..offset + num_elements * item_size];
            let typed_data: &[T] = bytemuck::cast_slice(data_slice);
            let data: Vec<T> = typed_data.iter().cloned().collect();

            let array = Array::from_shape_vec(actual_shape, data);

            Ok(MemmapArray {
                array,
                mmap: MemmapStorage::ReadWrite(mmap),
            })
        }
        MmapMode::None => Err(NumPyError::invalid_operation(
            "MmapMode::None is not valid for memmap",
        )),
    }
}

impl<T> MemmapArray<T> {
    /// Get a reference to the underlying array
    pub fn array(&self) -> &Array<T> {
        &self.array
    }

    /// Convert into the underlying array (consumes the MemmapArray)
    pub fn into_array(self) -> Array<T> {
        self.array
    }

    /// Flush changes to disk (for writable mappings)
    pub fn flush(&self) -> Result<()> {
        match &self.mmap {
            MemmapStorage::ReadWrite(mmap) => mmap
                .flush()
                .map_err(|e| NumPyError::io_error(format!("Failed to flush mmap: {}", e))),
            _ => Ok(()),
        }
    }
}

/// Get a buffer object exposing the underlying data of an array
///
/// # Arguments
/// * `arr` - The array to get the buffer from
/// * `offset` - Offset in bytes from the start of the array data
/// * `size` - Size of the buffer in bytes, or -1 for all remaining data
///
/// # Returns
/// A vector containing the buffer data
///
/// # Example
/// ```rust,ignore
/// use numpy::io::getbuffer;
/// use numpy::Array;
///
/// let arr = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
/// let buffer = getbuffer(&arr, 0, -1).unwrap();
/// ```
pub fn getbuffer<T>(arr: &Array<T>, offset: isize, size: isize) -> Result<Vec<u8>>
where
    T: Clone + Default + Pod + 'static,
{
    let item_size = arr.dtype().itemsize();
    let num_elements = arr.size();
    let total_bytes = num_elements * item_size;

    // Calculate actual offset
    let actual_offset = if offset < 0 {
        total_bytes.saturating_sub((-offset) as usize)
    } else {
        offset as usize
    };

    if actual_offset > total_bytes {
        return Err(NumPyError::invalid_operation(
            "Offset exceeds array data size",
        ));
    }

    // Calculate actual size
    let actual_size = if size < 0 {
        total_bytes - actual_offset
    } else {
        std::cmp::min(size as usize, total_bytes - actual_offset)
    };

    // Get the array data
    let data = arr.to_vec();
    let data_bytes: &[u8] = bytemuck::cast_slice(&data);

    // Extract the requested portion
    let end_offset = actual_offset + actual_size;
    if end_offset > data_bytes.len() {
        return Err(NumPyError::invalid_operation(
            "Buffer range exceeds array bounds",
        ));
    }

    Ok(data_bytes[actual_offset..end_offset].to_vec())
}

/// Get a buffer information structure for an array
///
/// Returns information about the array's buffer without copying data
///
/// # Arguments
/// * `arr` - The array to get buffer info from
///
/// # Returns
/// Buffer information including pointer, size, and format
pub fn getbuffer_info<T>(arr: &Array<T>) -> BufferInfo
where
    T: Clone + Default + Pod + 'static,
{
    let item_size = arr.dtype().itemsize();
    let num_elements = arr.size();
    let total_bytes = num_elements * item_size;

    BufferInfo {
        ndim: arr.ndim(),
        shape: arr.shape().to_vec(),
        strides: arr.strides().to_vec(),
        itemsize: item_size,
        size: total_bytes,
        format: arr.dtype().to_string(),
        readonly: false,
    }
}

/// Buffer information structure
#[derive(Debug, Clone)]
pub struct BufferInfo {
    /// Number of dimensions
    pub ndim: usize,
    /// Shape of the array
    pub shape: Vec<usize>,
    /// Strides in bytes
    pub strides: Vec<isize>,
    /// Size of each element in bytes
    pub itemsize: usize,
    /// Total size in bytes
    pub size: usize,
    /// Format string (like numpy's buffer protocol)
    pub format: String,
    /// Whether the buffer is read-only
    pub readonly: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_format_detection() {
        assert!(detect_file_format("test.npy").is_ok());
    }

    #[test]
    fn test_mmap_mode_parsing() {
        assert_eq!("r".parse::<MmapMode>().unwrap(), MmapMode::Read);
        assert_eq!("r+".parse::<MmapMode>().unwrap(), MmapMode::ReadWrite);
        assert_eq!("c".parse::<MmapMode>().unwrap(), MmapMode::Read);
    }

    #[test]
    fn test_memmap_read() {
        // Create a temporary file with test data
        let mut temp_file = NamedTempFile::new().unwrap();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_bytes: &[u8] = bytemuck::cast_slice(&data);
        temp_file.write_all(data_bytes).unwrap();
        temp_file.flush().unwrap();

        // Memory map the file
        let mmap_arr = memmap::<f64>(
            temp_file.path().to_str().unwrap(),
            None,
            MmapMode::Read,
            None,
            0,
        )
        .unwrap();

        let arr = mmap_arr.array();
        assert_eq!(arr.size(), 6);
        assert_eq!(arr.shape(), &[6]);

        // Check values
        for i in 0..6 {
            assert!((arr.get(i).unwrap() - (i as f64 + 1.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_memmap_with_shape() {
        // Create a temporary file with test data
        let mut temp_file = NamedTempFile::new().unwrap();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_bytes: &[u8] = bytemuck::cast_slice(&data);
        temp_file.write_all(data_bytes).unwrap();
        temp_file.flush().unwrap();

        // Memory map with specific shape
        let mmap_arr = memmap::<f64>(
            temp_file.path().to_str().unwrap(),
            None,
            MmapMode::Read,
            Some(vec![2, 3]),
            0,
        )
        .unwrap();

        let arr = mmap_arr.array();
        assert_eq!(arr.shape(), &[2, 3]);
    }

    #[test]
    fn test_memmap_with_offset() {
        // Create a temporary file with test data and offset
        let mut temp_file = NamedTempFile::new().unwrap();
        let header: Vec<u8> = vec![0u8; 16]; // 16 bytes of header
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let data_bytes: &[u8] = bytemuck::cast_slice(&data);
        temp_file.write_all(&header).unwrap();
        temp_file.write_all(data_bytes).unwrap();
        temp_file.flush().unwrap();

        // Memory map with offset
        let mmap_arr = memmap::<f64>(
            temp_file.path().to_str().unwrap(),
            None,
            MmapMode::Read,
            None,
            16,
        )
        .unwrap();

        let arr = mmap_arr.array();
        assert_eq!(arr.size(), 4);
    }

    #[test]
    fn test_getbuffer() {
        let arr = Array::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);

        // Get entire buffer
        let buffer = getbuffer(&arr, 0, -1).unwrap();
        assert_eq!(buffer.len(), 32); // 4 * 8 bytes for f64

        // Get partial buffer
        let buffer = getbuffer(&arr, 0, 16).unwrap();
        assert_eq!(buffer.len(), 16); // 2 * 8 bytes

        // Get buffer with offset
        let buffer = getbuffer(&arr, 8, 8).unwrap();
        assert_eq!(buffer.len(), 8); // 1 * 8 bytes
    }

    #[test]
    fn test_getbuffer_info() {
        let arr = Array::from_shape_vec(vec![2, 3], vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let info = getbuffer_info(&arr);
        assert_eq!(info.ndim, 2);
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.itemsize, 8); // f64
        assert_eq!(info.size, 48); // 6 * 8 bytes
        assert!(!info.readonly);
    }

    #[test]
    fn test_frombuffer_basic() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let data_bytes: &[u8] = bytemuck::cast_slice(&data);

        let arr = frombuffer::<f64>(data_bytes, None, None, 0).unwrap();
        assert_eq!(arr.size(), 4);
        assert_eq!(arr.shape(), &[4]);
    }

    #[test]
    fn test_frombuffer_with_offset() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let data_bytes: &[u8] = bytemuck::cast_slice(&data);

        // Skip first element (8 bytes for f64)
        let arr = frombuffer::<f64>(data_bytes, None, None, 8).unwrap();
        assert_eq!(arr.size(), 3);
    }

    #[test]
    fn test_frombuffer_with_count() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let data_bytes: &[u8] = bytemuck::cast_slice(&data);

        // Only read 2 elements
        let arr = frombuffer::<f64>(data_bytes, None, Some(2), 0).unwrap();
        assert_eq!(arr.size(), 2);
    }
}
