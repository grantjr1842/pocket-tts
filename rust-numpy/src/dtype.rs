#![allow(non_camel_case_types)]
use std::fmt;

/// Byte order for endianness support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    Little, // '<' - little-endian
    Big,    // '>' - big-endian
    Native, // '=' - native byte order
    Ignore, // '|' - ignore byte order
}

/// Casting rules for type conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Casting {
    /// Only allow safe casts that preserve values
    Safe,
    /// Allow safe casts and casts within the same kind (e.g. float64 -> float32)
    SameKind,
    /// Allow any cast
    Unsafe,
    /// Allow any cast (legacy name for Unsafe)
    Equiv,
    /// Identical types only
    No,
}

/// Comprehensive dtype system matching NumPy's type system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dtype {
    // Signed integer types
    Int8 { byteorder: Option<ByteOrder> },
    Int16 { byteorder: Option<ByteOrder> },
    Int32 { byteorder: Option<ByteOrder> },
    Int64 { byteorder: Option<ByteOrder> },
    Intp { byteorder: Option<ByteOrder> }, // Platform-dependent (i32 on 32-bit, i64 on 64-bit)

    // Unsigned integer types
    UInt8 { byteorder: Option<ByteOrder> },
    UInt16 { byteorder: Option<ByteOrder> },
    UInt32 { byteorder: Option<ByteOrder> },
    UInt64 { byteorder: Option<ByteOrder> },
    Uintp { byteorder: Option<ByteOrder> }, // Platform-dependent (u32 on 32-bit, u64 on 64-bit)

    // Floating point types
    Float16 { byteorder: Option<ByteOrder> },
    Float32 { byteorder: Option<ByteOrder> },
    Float64 { byteorder: Option<ByteOrder> },
    Float128 { byteorder: Option<ByteOrder> }, // Extended precision (platform-dependent)

    // Complex types
    Complex32 { byteorder: Option<ByteOrder> },  // 2x f16
    Complex64 { byteorder: Option<ByteOrder> },  // 2x f32
    Complex128 { byteorder: Option<ByteOrder> }, // 2x f64
    Complex256 { byteorder: Option<ByteOrder> }, // 2x f128 (platform-dependent)

    // Boolean type
    Bool,

    // String types with optional length specification
    String { length: Option<usize> },  // Byte strings (S)
    Unicode { length: Option<usize> }, // Unicode strings (U)

    // Binary data type
    Bytes { length: usize }, // Fixed-width bytes (b)

    // Datetime types
    Datetime64(DatetimeUnit),
    Timedelta64(TimedeltaUnit),

    // Object type
    Object,

    // Void type for padding/unstructured data
    Void { size: usize },

    // Structured type with enhanced field support
    Struct(Vec<StructField>),
}

/// Units for datetime64
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    As,
}

impl DatetimeUnit {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Y => "Y",
            Self::M => "M",
            Self::W => "W",
            Self::D => "D",
            Self::h => "h",
            Self::m => "m",
            Self::s => "s",
            Self::ms => "ms",
            Self::us => "us",
            Self::ns => "ns",
            Self::ps => "ps",
            Self::fs => "fs",
            Self::As => "as",
        }
    }
}

/// Units for timedelta64
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

/// Character codes for NumPy dtype string representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DTypeChar {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    Bytes,
    Unicode,
    Object,
    Datetime64,
    Timedelta64,
    Void,
}

/// Field definition for structured dtypes with enhanced NumPy compatibility
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructField {
    pub name: String,
    pub dtype: Dtype,
    pub offset: Option<usize>,
    pub title: Option<String>,     // Field title for display
    pub shape: Option<Vec<usize>>, // Subarray shape, None for scalar
}

/// Kind of dtype for type checking and promotion
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DtypeKind {
    Integer,
    Unsigned,
    Float,
    Complex,
    Bool,
    String,
    Bytes,
    Datetime,
    Object,
    Struct,
    Void,
}

impl Dtype {
    /// Get the kind of this dtype
    pub fn kind(&self) -> DtypeKind {
        match self {
            Self::Int8 { .. }
            | Self::Int16 { .. }
            | Self::Int32 { .. }
            | Self::Int64 { .. }
            | Self::Intp { .. } => DtypeKind::Integer,
            Self::UInt8 { .. }
            | Self::UInt16 { .. }
            | Self::UInt32 { .. }
            | Self::UInt64 { .. }
            | Self::Uintp { .. } => DtypeKind::Unsigned,
            Self::Float16 { .. }
            | Self::Float32 { .. }
            | Self::Float64 { .. }
            | Self::Float128 { .. } => DtypeKind::Float,
            Self::Complex32 { .. }
            | Self::Complex64 { .. }
            | Self::Complex128 { .. }
            | Self::Complex256 { .. } => DtypeKind::Complex,
            Self::Bool => DtypeKind::Bool,
            Self::String { .. } | Self::Unicode { .. } => DtypeKind::String,
            Self::Bytes { .. } => DtypeKind::Bytes,
            Self::Datetime64(_) | Self::Timedelta64(_) => DtypeKind::Datetime,
            Self::Object => DtypeKind::Object,
            Self::Struct(_) => DtypeKind::Struct,
            Self::Void { .. } => DtypeKind::Void,
        }
    }

    /// Get size in bytes
    pub fn itemsize(&self) -> usize {
        match self {
            Self::Int8 { .. } | Self::UInt8 { .. } | Self::Bool => 1,
            Self::Int16 { .. } | Self::UInt16 { .. } | Self::Float16 { .. } => 2,
            Self::Int32 { .. }
            | Self::UInt32 { .. }
            | Self::Float32 { .. }
            | Self::Complex32 { .. } => 4,
            Self::Int64 { .. }
            | Self::UInt64 { .. }
            | Self::Float64 { .. }
            | Self::Complex64 { .. } => 8,
            Self::Complex128 { .. } => 16,
            Self::Complex256 { .. } => 32,
            Self::Float128 { .. } => 16,
            Self::Intp { .. } => std::mem::size_of::<isize>(),
            Self::Uintp { .. } => std::mem::size_of::<usize>(),
            Self::String { length } => length.unwrap_or_else(|| 8),
            Self::Unicode { length } => length.unwrap_or_else(|| 8) * 4,
            Self::Bytes { length } => *length,
            Self::Datetime64(_) | Self::Timedelta64(_) => 8,
            Self::Object => 8,
            Self::Void { size } => *size,
            Self::Struct(fields) => fields.iter().map(|f| f.dtype.itemsize()).sum(),
        }
    }

    /// Get alignment requirement in bytes
    pub fn alignment(&self) -> usize {
        match self {
            Self::Int8 { .. } | Self::UInt8 { .. } | Self::Bool => 1,
            Self::Int16 { .. } | Self::UInt16 { .. } | Self::Float16 { .. } => 2,
            Self::Int32 { .. }
            | Self::UInt32 { .. }
            | Self::Float32 { .. }
            | Self::Complex32 { .. } => 4,
            Self::Int64 { .. }
            | Self::UInt64 { .. }
            | Self::Float64 { .. }
            | Self::Complex64 { .. } => 8,
            Self::Complex128 { .. } => 16,
            Self::Complex256 { .. } => 32,
            Self::Float128 { .. } => 16,
            Self::Intp { .. } => std::mem::align_of::<isize>(),
            Self::Uintp { .. } => std::mem::align_of::<usize>(),
            Self::String { .. } => 8,
            Self::Unicode { .. } => 8,
            Self::Bytes { .. } => 1,
            Self::Datetime64(_) | Self::Timedelta64(_) => 8,
            Self::Object => 8,
            Self::Void { .. } => 1,
            Self::Struct(fields) => fields
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
            Self::Int8 { byteorder: None }
        } else if type_id == TypeId::of::<i16>() {
            Self::Int16 { byteorder: None }
        } else if type_id == TypeId::of::<i32>() {
            Self::Int32 { byteorder: None }
        } else if type_id == TypeId::of::<i64>() {
            Self::Int64 { byteorder: None }
        } else if type_id == TypeId::of::<isize>() {
            Self::Intp { byteorder: None }
        } else if type_id == TypeId::of::<u8>() {
            Self::UInt8 { byteorder: None }
        } else if type_id == TypeId::of::<u16>() {
            Self::UInt16 { byteorder: None }
        } else if type_id == TypeId::of::<u32>() {
            Self::UInt32 { byteorder: None }
        } else if type_id == TypeId::of::<u64>() {
            Self::UInt64 { byteorder: None }
        } else if type_id == TypeId::of::<usize>() {
            Self::Uintp { byteorder: None }
        } else if type_id == TypeId::of::<f32>() {
            Self::Float32 { byteorder: None }
        } else if type_id == TypeId::of::<f64>() {
            Self::Float64 { byteorder: None }
        } else if type_id == TypeId::of::<bool>() {
            Self::Bool
        } else if type_id == TypeId::of::<String>() {
            Self::Unicode { length: None }
        } else if type_id == TypeId::of::<&str>() {
            Self::Unicode { length: None }
        } else {
            Self::Object
        }
    }

    /// Parse dtype from string (NumPy compatible)
    pub fn from_str(s: &str) -> Result<Self, String> {
        let s = s.strip_prefix("np.").unwrap_or(s);
        match s {
            "int8" | "i1" => Ok(Self::Int8 { byteorder: None }),
            "int16" | "i2" => Ok(Self::Int16 { byteorder: None }),
            "int32" | "i4" => Ok(Self::Int32 { byteorder: None }),
            "int64" | "i8" => Ok(Self::Int64 { byteorder: None }),
            "intp" | "ip" => Ok(Self::Intp { byteorder: None }),
            "uint8" | "u1" => Ok(Self::UInt8 { byteorder: None }),
            "uint16" | "u2" => Ok(Self::UInt16 { byteorder: None }),
            "uint32" | "u4" => Ok(Self::UInt32 { byteorder: None }),
            "uint64" | "u8" => Ok(Self::UInt64 { byteorder: None }),
            "uintp" | "up" => Ok(Self::Uintp { byteorder: None }),
            "float16" | "f2" => Ok(Self::Float16 { byteorder: None }),
            "float32" | "f4" => Ok(Self::Float32 { byteorder: None }),
            "float64" | "f8" => Ok(Self::Float64 { byteorder: None }),
            "complex32" | "c4" => Ok(Self::Complex32 { byteorder: None }),
            "complex64" | "c8" => Ok(Self::Complex64 { byteorder: None }),
            "complex128" | "c16" => Ok(Self::Complex128 { byteorder: None }),
            "bool" => Ok(Self::Bool),
            "str" => Ok(Self::String { length: None }),
            "unicode" => Ok(Self::Unicode { length: None }),
            "object" => Ok(Self::Object),
            _ => {
                if s.starts_with("dtype[") && s.ends_with(']') {
                    let inner = &s[6..s.len() - 1];
                    return Self::from_str(inner);
                }
                if s.starts_with("datetime64") {
                    let unit = s
                        .strip_prefix("datetime64[")
                        .and_then(|s| s.strip_suffix("]"));
                    if let Some(unit_str) = unit {
                        match unit_str {
                            "Y" => Ok(Self::Datetime64(DatetimeUnit::Y)),
                            "M" => Ok(Self::Datetime64(DatetimeUnit::M)),
                            "W" => Ok(Self::Datetime64(DatetimeUnit::W)),
                            "D" => Ok(Self::Datetime64(DatetimeUnit::D)),
                            "h" => Ok(Self::Datetime64(DatetimeUnit::h)),
                            "m" => Ok(Self::Datetime64(DatetimeUnit::m)),
                            "s" => Ok(Self::Datetime64(DatetimeUnit::s)),
                            "ms" => Ok(Self::Datetime64(DatetimeUnit::ms)),
                            "us" => Ok(Self::Datetime64(DatetimeUnit::us)),
                            "ns" => Ok(Self::Datetime64(DatetimeUnit::ns)),
                            _ => Err(format!("Unknown datetime unit: {}", unit_str)),
                        }
                    } else {
                        Ok(Self::Datetime64(DatetimeUnit::ns))
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
            Self::Int8 { .. } => "int8".to_string(),
            Self::Int16 { .. } => "int16".to_string(),
            Self::Int32 { .. } => "int32".to_string(),
            Self::Int64 { .. } => "int64".to_string(),
            Self::UInt8 { .. } => "uint8".to_string(),
            Self::UInt16 { .. } => "uint16".to_string(),
            Self::UInt32 { .. } => "uint32".to_string(),
            Self::UInt64 { .. } => "uint64".to_string(),
            Self::Float16 { .. } => "float16".to_string(),
            Self::Float32 { .. } => "float32".to_string(),
            Self::Float64 { .. } => "float64".to_string(),
            Self::Complex32 { .. } => "complex32".to_string(),
            Self::Complex64 { .. } => "complex64".to_string(),
            Self::Complex128 { .. } => "complex128".to_string(),
            Self::Bool => "bool".to_string(),
            Self::String { .. } => "str".to_string(),
            Self::Unicode { .. } => "unicode".to_string(),
            Self::Object => "object".to_string(),
            Self::Struct(_) => "struct".to_string(),
            Self::Datetime64(unit) => format!(
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
            _ => "unknown".to_string(),
        }
    }

    /// Check if dtype can be cast to another dtype according to the given casting policy
    pub fn can_cast(&self, to: &Dtype, policy: Casting) -> bool {
        if self == to {
            return true;
        }

        match policy {
            Casting::Unsafe | Casting::Equiv => true,
            Casting::SameKind => self.can_cast_same_kind(to),
            Casting::Safe => self.can_cast_safe(to),
            Casting::No => false,
        }
    }

    /// Check if dtype can be reliably cast to another dtype (equivalent to casting='safe')
    pub fn can_cast_to(&self, other: &Dtype) -> bool {
        self.can_cast(other, Casting::Safe)
    }

    /// Helper for 'same_kind' casting
    fn can_cast_same_kind(&self, other: &Dtype) -> bool {
        if self.kind() == other.kind() {
            true
        } else {
            self.can_cast_safe(other)
        }
    }

    /// Helper for 'safe' casting
    fn can_cast_safe(&self, other: &Dtype) -> bool {
        use DtypeKind::*;
        let self_kind = self.kind();
        let other_kind = other.kind();

        match (self_kind, other_kind) {
            // Bool casts
            (Bool, Bool) => true,
            (Bool, Integer) | (Bool, Unsigned) | (Bool, Float) | (Bool, Complex) => true,
            (_, Bool) => false,

            // Integer casts
            (Integer, Integer) => self.itemsize() <= other.itemsize(),
            (Unsigned, Unsigned) => self.itemsize() <= other.itemsize(),
            (Unsigned, Integer) => self.itemsize() < other.itemsize(), // Needs strictly larger to fit all unsigned values
            (Integer, Unsigned) => false, // Negative values cannot be represented suitable

            // Float casts
            (Float, Float) => self.itemsize() <= other.itemsize(),
            (Integer, Float) => true, // Conventionally considered safe in NumPy
            (Unsigned, Float) => true,

            // Complex casts
            (Complex, Complex) => self.itemsize() <= other.itemsize(),
            (Float, Complex) => self.itemsize() * 2 <= other.itemsize(),
            (Integer | Unsigned, Complex) => true,

            // Others
            (String, String) => true,
            (Datetime, Datetime) => true,
            (Object, _) | (_, Object) => true,

            // Default fail
            _ => false,
        }
    }

    /// Get byte order of this dtype
    pub fn byteorder(&self) -> Option<ByteOrder> {
        match self {
            Self::Int8 { byteorder } => *byteorder,
            Self::Int16 { byteorder } => *byteorder,
            Self::Int32 { byteorder } => *byteorder,
            Self::Int64 { byteorder } => *byteorder,
            Self::Intp { byteorder } => *byteorder,
            Self::UInt8 { byteorder } => *byteorder,
            Self::UInt16 { byteorder } => *byteorder,
            Self::UInt32 { byteorder } => *byteorder,
            Self::UInt64 { byteorder } => *byteorder,
            Self::Uintp { byteorder } => *byteorder,
            Self::Float16 { byteorder } => *byteorder,
            Self::Float32 { byteorder } => *byteorder,
            Self::Float64 { byteorder } => *byteorder,
            Self::Float128 { byteorder } => *byteorder,
            Self::Complex32 { byteorder } => *byteorder,
            Self::Complex64 { byteorder } => *byteorder,
            Self::Complex128 { byteorder } => *byteorder,
            Self::Complex256 { byteorder } => *byteorder,
            Self::Bool => None,
            Self::String { .. } => None,
            Self::Unicode { .. } => None,
            Self::Bytes { .. } => None,
            Self::Datetime64(_) => None,
            Self::Timedelta64(_) => None,
            Self::Object => None,
            Self::Void { .. } => None,
            Self::Struct(_) => None,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// IEEE 754 half-precision float type (re-exported from half crate)
pub use half::f16;
