#![allow(non_camel_case_types)]
use std::fmt;

/// Comprehensive dtype system matching NumPy's type system
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dtype {
    // Signed integer types
    Int8,
    Int16,
    Int32,
    Int64,

    // Unsigned integer types
    UInt8,
    UInt16,
    UInt32,
    UInt64,

    // Floating point types
    Float16,
    Float32,
    Float64,

    // Complex types
    Complex32,
    Complex64,
    Complex128,

    // Boolean type
    Bool,

    // String types
    String,
    Unicode,

    // Datetime types
    Datetime64(DatetimeUnit),
    Timedelta64(TimedeltaUnit),

    // Object type
    Object,

    // Structured type
    Struct(Vec<StructField>),
}

/// Units for datetime64
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatetimeUnit {
    Y,
    M,
    W,
    D,
    h,
    m,
    s,
    ms,
    us,
    ns,
    ps,
    fs,
    As, // Fixed: 'as' -> 'As'
}

/// Units for timedelta64
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimedeltaUnit {
    Y,
    M,
    W,
    D,
    h,
    m,
    s,
    ms,
    us,
    ns,
    ps,
    fs,
    As,
}

/// Field definition for structured dtypes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub dtype: Dtype,
    pub offset: Option<usize>,
}

/// Kind of dtype for type checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DtypeKind {
    Integer,
    Unsigned,
    Float,
    Complex,
    Bool,
    String,
    Datetime,
    Object,
    Struct,
}

impl Dtype {
    /// Get the kind of this dtype
    pub fn kind(&self) -> DtypeKind {
        match self {
            Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 => DtypeKind::Integer,
            Dtype::UInt8 | Dtype::UInt16 | Dtype::UInt32 | Dtype::UInt64 => DtypeKind::Unsigned,
            Dtype::Float16 | Dtype::Float32 | Dtype::Float64 => DtypeKind::Float,
            Dtype::Complex32 | Dtype::Complex64 | Dtype::Complex128 => DtypeKind::Complex,
            Dtype::Bool => DtypeKind::Bool,
            Dtype::String | Dtype::Unicode => DtypeKind::String,
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => DtypeKind::Datetime,
            Dtype::Object => DtypeKind::Object,
            Dtype::Struct(_) => DtypeKind::Struct,
        }
    }

    /// Get size in bytes
    pub fn itemsize(&self) -> usize {
        match self {
            Dtype::Int8 | Dtype::UInt8 | Dtype::Bool => 1,
            Dtype::Int16 | Dtype::UInt16 | Dtype::Float16 => 2,
            Dtype::Int32 | Dtype::UInt32 | Dtype::Float32 | Dtype::Complex32 => 4,
            Dtype::Int64 | Dtype::UInt64 | Dtype::Float64 | Dtype::Complex64 => 8,
            Dtype::Complex128 => 16,
            Dtype::String | Dtype::Unicode => 8, // Pointer size
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => 8,
            Dtype::Object => 8, // Pointer size
            Dtype::Struct(fields) => fields.iter().map(|f| f.dtype.itemsize()).sum(),
        }
    }

    /// Get alignment requirement in bytes
    pub fn alignment(&self) -> usize {
        match self {
            Dtype::Int8 | Dtype::UInt8 | Dtype::Bool => 1,
            Dtype::Int16 | Dtype::UInt16 | Dtype::Float16 => 2,
            Dtype::Int32 | Dtype::UInt32 | Dtype::Float32 | Dtype::Complex32 => 4,
            Dtype::Int64 | Dtype::UInt64 | Dtype::Float64 | Dtype::Complex64 => 8,
            Dtype::Complex128 => 16,
            Dtype::String | Dtype::Unicode => 8,
            Dtype::Datetime64(_) | Dtype::Timedelta64(_) => 8,
            Dtype::Object => 8,
            Dtype::Struct(fields) => fields
                .iter()
                .map(|f| f.dtype.alignment())
                .max()
                .unwrap_or(1),
        }
    }

    /// Create dtype from Rust type
    pub fn from_type<T: 'static>() -> Self {
        use std::any::TypeId;

        let type_id = TypeId::of::<T>();

        if type_id == TypeId::of::<i8>() {
            Dtype::Int8
        } else if type_id == TypeId::of::<i16>() {
            Dtype::Int16
        } else if type_id == TypeId::of::<i32>() {
            Dtype::Int32
        } else if type_id == TypeId::of::<i64>() {
            Dtype::Int64
        } else if type_id == TypeId::of::<u8>() {
            Dtype::UInt8
        } else if type_id == TypeId::of::<u16>() {
            Dtype::UInt16
        } else if type_id == TypeId::of::<u32>() {
            Dtype::UInt32
        } else if type_id == TypeId::of::<u64>() {
            Dtype::UInt64
        } else if type_id == TypeId::of::<f16>() {
            Dtype::Float16
        } else if type_id == TypeId::of::<f32>() {
            Dtype::Float32
        } else if type_id == TypeId::of::<f64>() {
            Dtype::Float64
        } else if type_id == TypeId::of::<bool>() {
            Dtype::Bool
        } else {
            Dtype::Object
        }
    }

    /// Parse dtype from string (NumPy compatible)
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "int8" | "i1" => Ok(Dtype::Int8),
            "int16" | "i2" => Ok(Dtype::Int16),
            "int32" | "i4" => Ok(Dtype::Int32),
            "int64" | "i8" => Ok(Dtype::Int64),
            "uint8" | "u1" => Ok(Dtype::UInt8),
            "uint16" | "u2" => Ok(Dtype::UInt16),
            "uint32" | "u4" => Ok(Dtype::UInt32),
            "uint64" | "u8" => Ok(Dtype::UInt64),
            "float16" | "f2" => Ok(Dtype::Float16),
            "float32" | "f4" => Ok(Dtype::Float32),
            "float64" | "f8" => Ok(Dtype::Float64),
            "complex64" | "c8" => Ok(Dtype::Complex64),
            "complex128" | "c16" => Ok(Dtype::Complex128),
            "bool" => Ok(Dtype::Bool),
            "str" => Ok(Dtype::String),
            "unicode" => Ok(Dtype::Unicode),
            "object" => Ok(Dtype::Object),
            _ => {
                // Try to parse datetime types
                if s.starts_with("datetime64") {
                    let unit = s
                        .strip_prefix("datetime64[")
                        .and_then(|s| s.strip_suffix("]"));
                    if let Some(unit_str) = unit {
                        match unit_str {
                            "Y" => Ok(Dtype::Datetime64(DatetimeUnit::Y)),
                            "M" => Ok(Dtype::Datetime64(DatetimeUnit::M)),
                            "W" => Ok(Dtype::Datetime64(DatetimeUnit::W)),
                            "D" => Ok(Dtype::Datetime64(DatetimeUnit::D)),
                            "h" => Ok(Dtype::Datetime64(DatetimeUnit::h)),
                            "m" => Ok(Dtype::Datetime64(DatetimeUnit::m)),
                            "s" => Ok(Dtype::Datetime64(DatetimeUnit::s)),
                            "ms" => Ok(Dtype::Datetime64(DatetimeUnit::ms)),
                            "us" => Ok(Dtype::Datetime64(DatetimeUnit::us)),
                            "ns" => Ok(Dtype::Datetime64(DatetimeUnit::ns)),
                            _ => Err(format!("Unknown datetime unit: {}", unit_str)),
                        }
                    } else {
                        Ok(Dtype::Datetime64(DatetimeUnit::ns)) // Default
                    }
                } else {
                    Err(format!("Unknown dtype: {}", s))
                }
            }
        }
    }

    /// Convert to string (NumPy compatible)
    pub fn to_string(&self) -> String {
        match self {
            Dtype::Int8 => "int8".to_string(),
            Dtype::Int16 => "int16".to_string(),
            Dtype::Int32 => "int32".to_string(),
            Dtype::Int64 => "int64".to_string(),
            Dtype::UInt8 => "uint8".to_string(),
            Dtype::UInt16 => "uint16".to_string(),
            Dtype::UInt32 => "uint32".to_string(),
            Dtype::UInt64 => "uint64".to_string(),
            Dtype::Float16 => "float16".to_string(),
            Dtype::Float32 => "float32".to_string(),
            Dtype::Float64 => "float64".to_string(),
            Dtype::Complex32 => "complex64".to_string(),
            Dtype::Complex64 => "complex64".to_string(),
            Dtype::Complex128 => "complex128".to_string(),
            Dtype::Bool => "bool".to_string(),
            Dtype::String => "str".to_string(),
            Dtype::Unicode => "unicode".to_string(),
            Dtype::Datetime64(unit) => format!(
                "datetime64[{}]",
                match unit {
                    DatetimeUnit::Y => "Y",
                    DatetimeUnit::M => "M",
                    DatetimeUnit::W => "W",
                    DatetimeUnit::D => "D",
                    DatetimeUnit::h => "h",
                    DatetimeUnit::m => "m",
                    DatetimeUnit::s => "s",
                    DatetimeUnit::ms => "ms",
                    DatetimeUnit::us => "us",
                    DatetimeUnit::ns => "ns",
                    DatetimeUnit::ps => "ps",
                    DatetimeUnit::fs => "fs",
                    DatetimeUnit::As => "as",
                }
            ),
            Dtype::Timedelta64(unit) => format!(
                "timedelta64[{}]",
                match unit {
                    TimedeltaUnit::Y => "Y",
                    TimedeltaUnit::M => "M",
                    TimedeltaUnit::W => "W",
                    TimedeltaUnit::D => "D",
                    TimedeltaUnit::h => "h",
                    TimedeltaUnit::m => "m",
                    TimedeltaUnit::s => "s",
                    TimedeltaUnit::ms => "ms",
                    TimedeltaUnit::us => "us",
                    TimedeltaUnit::ns => "ns",
                    TimedeltaUnit::ps => "ps",
                    TimedeltaUnit::fs => "fs",
                    TimedeltaUnit::As => "as",
                }
            ),
            Dtype::Object => "object".to_string(),
            Dtype::Struct(_) => "struct".to_string(),
        }
    }

    /// Check if dtype can be safely cast to another dtype
    pub fn can_cast_to(&self, other: &Dtype) -> bool {
        use DtypeKind::*;

        let self_kind = self.kind();
        let other_kind = other.kind();

        match (self_kind, other_kind) {
            (Integer, Integer) | (Unsigned, Integer) | (Unsigned, Unsigned) => {
                self.itemsize() <= other.itemsize()
            }
            (Integer, Unsigned) => false,
            (Float, Float) => self.itemsize() <= other.itemsize(),
            (Complex, Complex) => self.itemsize() <= other.itemsize(),
            (Integer | Unsigned | Float, Complex) => true,
            (Complex, Float) => false,
            (Bool, _) => true,
            (_, Bool) => false,
            (String, String) | (Datetime, Datetime) | (Object, _) | (_, Object) => true,
            _ => false,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Half-precision float type
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct f16(u16);

impl f16 {
    pub fn new(value: f32) -> Self {
        // Simplified conversion - real implementation would be more complex
        Self(((value.to_bits()) >> 16) as u16)
    }

    pub fn to_f32(self) -> f32 {
        // Simplified conversion - real implementation would be more complex
        f32::from_bits((self.0 as u32) << 16)
    }
}

impl From<f32> for f16 {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<f16> for f32 {
    fn from(value: f16) -> Self {
        value.to_f32()
    }
}
