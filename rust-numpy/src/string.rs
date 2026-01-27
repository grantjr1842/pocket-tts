// String type definitions for NumPy compatibility

/// Fixed-length byte string type
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct String(pub Vec<u8>);

/// Fixed-length Unicode string type (UTF-32 storage)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unicode(pub Vec<u32>);
