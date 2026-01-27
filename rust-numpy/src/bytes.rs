// Bytes type definitions for NumPy compatibility

/// Fixed-length bytes type
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bytes(pub Vec<u8>);
