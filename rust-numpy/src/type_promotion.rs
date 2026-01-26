use crate::dtype::{Dtype, DtypeKind};
use std::collections::HashMap;
use std::sync::Arc;

/// Comprehensive type promotion rules system matching NumPy's behavior
///
/// This struct provides NumPy-compatible type promotion with:
/// - Full promotion tables for all dtype combinations
/// - Safe casting with overflow detection
/// - Complex type promotion rules
/// - Boolean and datetime promotion support
#[derive(Debug, Clone)]
pub struct TypePromotionRules {
    promotion_table: Arc<HashMap<(DtypeKind, DtypeKind), Dtype>>,
    safe_casting_table: Arc<HashMap<(Dtype, Dtype), bool>>,
}

impl TypePromotionRules {
    /// Create a new TypePromotionRules with NumPy-compatible promotion tables
    pub fn new() -> Self {
        let mut promotion_table = HashMap::new();
        let mut safe_casting_table = HashMap::new();

        // Initialize promotion tables
        Self::build_promotion_table(&mut promotion_table);
        Self::build_safe_casting_table(&mut safe_casting_table);

        Self {
            promotion_table: Arc::new(promotion_table),
            safe_casting_table: Arc::new(safe_casting_table),
        }
    }

    /// Promote multiple dtypes to a common result type
    ///
    /// Implements NumPy's result_type logic:
    /// 1. Find common type for all inputs
    /// 2. Apply safe casting rules
    /// 3. Handle special cases (bool, complex, datetime)
    pub fn promote_types(&self, dtypes: &[Dtype]) -> Result<Dtype, String> {
        if dtypes.is_empty() {
            return Err("Cannot promote empty type list".to_string());
        }

        if dtypes.len() == 1 {
            return Ok(dtypes[0].clone());
        }

        let mut result = dtypes[0].clone();

        for i in 1..dtypes.len() {
            result = self.promote_two_types(&result, &dtypes[i])?;
        }

        Ok(result)
    }

    /// Promote two dtypes to their common type
    pub fn promote_two_types(&self, t1: &Dtype, t2: &Dtype) -> Result<Dtype, String> {
        if t1 == t2 {
            return Ok(t1.clone());
        }

        let k1 = t1.kind();
        let k2 = t2.kind();

        // Check promotion table first
        if let Some(result) = self.promotion_table.get(&(k1.clone(), k2.clone())) {
            return Ok(self.adjust_result_size(result, t1, t2));
        }

        // Try reverse order
        if let Some(result) = self.promotion_table.get(&(k2.clone(), k1.clone())) {
            return Ok(self.adjust_result_size(result, t1, t2));
        }

        // Fallback to existing logic for edge cases
        promote_types(t1, t2).ok_or_else(|| format!("Cannot promote types {:?} and {:?}", t1, t2))
    }

    /// Check if casting is safe without loss of precision
    pub fn can_safely_cast(&self, from: &Dtype, to: &Dtype) -> bool {
        if from == to {
            return true;
        }

        // Check safe casting table
        if let Some(&safe) = self.safe_casting_table.get(&(from.clone(), to.clone())) {
            return safe;
        }

        // Fallback to dtype's built-in logic
        from.can_cast_to(to)
    }

    /// Adjust result dtype based on input sizes for better precision
    fn adjust_result_size(&self, result: &Dtype, t1: &Dtype, t2: &Dtype) -> Dtype {
        match result.kind() {
            DtypeKind::Float => {
                // For floats, prefer larger size if inputs are large
                let max_size = t1.itemsize().max(t2.itemsize());
                if max_size > result.itemsize() {
                    match max_size {
                        8 => Dtype::Float64 { byteorder: None },
                        16 => Dtype::Float128 { byteorder: None },
                        _ => result.clone(),
                    }
                } else {
                    result.clone()
                }
            }
            DtypeKind::Complex => {
                // For complex, prefer larger size if inputs are large
                let max_size = t1.itemsize().max(t2.itemsize());
                if max_size > result.itemsize() {
                    match max_size {
                        8 => Dtype::Complex64 { byteorder: None },
                        16 => Dtype::Complex128 { byteorder: None },
                        _ => result.clone(),
                    }
                } else {
                    result.clone()
                }
            }
            _ => result.clone(),
        }
    }

    /// Build the main promotion table following NumPy's rules
    fn build_promotion_table(table: &mut HashMap<(DtypeKind, DtypeKind), Dtype>) {
        use DtypeKind::*;

        // Boolean promotion rules
        table.insert((Bool, Bool), Dtype::Bool);
        table.insert((Bool, Integer), Dtype::Int8 { byteorder: None });
        table.insert((Bool, Unsigned), Dtype::UInt8 { byteorder: None });
        table.insert((Bool, Float), Dtype::Float32 { byteorder: None });
        table.insert((Bool, Complex), Dtype::Complex64 { byteorder: None });

        // Integer promotion rules
        table.insert((Integer, Integer), Dtype::Int64 { byteorder: None });
        table.insert((Integer, Unsigned), Dtype::Int64 { byteorder: None });
        table.insert((Integer, Float), Dtype::Float64 { byteorder: None });
        table.insert((Integer, Complex), Dtype::Complex128 { byteorder: None });

        // Unsigned promotion rules
        table.insert((Unsigned, Unsigned), Dtype::UInt64 { byteorder: None });
        table.insert((Unsigned, Float), Dtype::Float64 { byteorder: None });
        table.insert((Unsigned, Complex), Dtype::Complex128 { byteorder: None });

        // Float promotion rules
        table.insert((Float, Float), Dtype::Float64 { byteorder: None });
        table.insert((Float, Complex), Dtype::Complex128 { byteorder: None });

        // Complex promotion rules
        table.insert((Complex, Complex), Dtype::Complex128 { byteorder: None });

        // String/Bytes promotion
        table.insert((String, String), Dtype::Unicode { length: None });
        table.insert((String, Bytes), Dtype::Unicode { length: None });
        table.insert((Bytes, Bytes), Dtype::Bytes { length: 0 });

        // Datetime promotion
        table.insert(
            (Datetime, Datetime),
            Dtype::Datetime64(crate::dtype::DatetimeUnit::ns),
        );

        // Object fallback
        table.insert((Object, Object), Dtype::Object);
    }

    /// Build safe casting table for overflow detection
    fn build_safe_casting_table(table: &mut HashMap<(Dtype, Dtype), bool>) {
        // Integer to integer (safe if target is larger or same size)
        table.insert(
            (
                Dtype::Int8 { byteorder: None },
                Dtype::Int16 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Int8 { byteorder: None },
                Dtype::Int32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Int8 { byteorder: None },
                Dtype::Int64 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Int16 { byteorder: None },
                Dtype::Int32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Int16 { byteorder: None },
                Dtype::Int64 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Int32 { byteorder: None },
                Dtype::Int64 { byteorder: None },
            ),
            true,
        );

        // Unsigned to unsigned (safe if target is larger or same size)
        table.insert(
            (
                Dtype::UInt8 { byteorder: None },
                Dtype::UInt16 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt8 { byteorder: None },
                Dtype::UInt32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt8 { byteorder: None },
                Dtype::UInt64 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt16 { byteorder: None },
                Dtype::UInt32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt16 { byteorder: None },
                Dtype::UInt64 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt32 { byteorder: None },
                Dtype::UInt64 { byteorder: None },
            ),
            true,
        );

        // Unsigned to signed (only safe if signed is larger)
        table.insert(
            (
                Dtype::UInt8 { byteorder: None },
                Dtype::Int16 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt8 { byteorder: None },
                Dtype::Int32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt8 { byteorder: None },
                Dtype::Int64 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt16 { byteorder: None },
                Dtype::Int32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt16 { byteorder: None },
                Dtype::Int64 { byteorder: None },
            ),
            true,
        );
        // UInt32 to Int64 is safe, UInt32 to Int32 is not
        table.insert(
            (
                Dtype::UInt32 { byteorder: None },
                Dtype::Int64 { byteorder: None },
            ),
            true,
        );
        // UInt64 cannot safely cast to any standard integer type

        // Float promotion (generally safe to go to larger precision)
        table.insert(
            (
                Dtype::Float16 { byteorder: None },
                Dtype::Float32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Float16 { byteorder: None },
                Dtype::Float64 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Float32 { byteorder: None },
                Dtype::Float64 { byteorder: None },
            ),
            true,
        );

        // Complex promotion
        table.insert(
            (
                Dtype::Complex32 { byteorder: None },
                Dtype::Complex64 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Complex32 { byteorder: None },
                Dtype::Complex128 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Complex64 { byteorder: None },
                Dtype::Complex128 { byteorder: None },
            ),
            true,
        );

        // Integer to float (generally safe)
        table.insert(
            (
                Dtype::Int8 { byteorder: None },
                Dtype::Float32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Int16 { byteorder: None },
                Dtype::Float32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Int32 { byteorder: None },
                Dtype::Float64 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::Int64 { byteorder: None },
                Dtype::Float64 { byteorder: None },
            ),
            true,
        );

        // Unsigned to float (generally safe)
        table.insert(
            (
                Dtype::UInt8 { byteorder: None },
                Dtype::Float32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt16 { byteorder: None },
                Dtype::Float32 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt32 { byteorder: None },
                Dtype::Float64 { byteorder: None },
            ),
            true,
        );
        table.insert(
            (
                Dtype::UInt64 { byteorder: None },
                Dtype::Float64 { byteorder: None },
            ),
            true,
        );

        // Mark some unsafe casts
        table.insert(
            (
                Dtype::Int64 { byteorder: None },
                Dtype::Int32 { byteorder: None },
            ),
            false,
        );
        table.insert(
            (
                Dtype::UInt64 { byteorder: None },
                Dtype::Int64 { byteorder: None },
            ),
            false,
        );
        table.insert(
            (
                Dtype::Float64 { byteorder: None },
                Dtype::Float32 { byteorder: None },
            ),
            false,
        );
    }
}

impl Default for TypePromotionRules {
    fn default() -> Self {
        Self::new()
    }
}

/// Promote two dtypes to a common dtype that can safely hold values of both.
///
/// This implements logic similar to NumPy's `result_type`.
/// - Bool -> Integer -> Float -> Complex
/// - Size increases to max of both (e.g. i8 + i32 -> i32)
/// - Mixed Signed/Unsigned:
///   - Same size: Signed wins (u8 + i8 -> i16 to be safe? NumPy does i16)
///   - Different size: Largest wins, but if unsigned is larger, might need next size up signed.
///     - u8 + i16 -> i16
///     - u32 + i16 -> i64 (to hold u32 range)
///     - u64 + i64 -> float64 (cannot comfortably fit in i64)
pub fn promote_types(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    if t1 == t2 {
        return Some(t1.clone());
    }

    let k1 = t1.kind();
    let k2 = t2.kind();

    use DtypeKind::*;

    // Hierarchy of kinds
    let score = |k: &DtypeKind| match k {
        Bool => 0,
        Integer => 1,
        Unsigned => 1, // Same level, handled specifically
        Float => 2,
        Complex => 3,
        Datetime => 4,
        String => 7, // String > Bytes for promotion (e.g. U + S -> U)
        Bytes => 6,
        Object => 10,
        _ => 20,
    };

    let s1 = score(&k1);
    let s2 = score(&k2);

    // If kinds differ significantly (e.g. Int vs Float), pick higher kind
    if s1 != s2 {
        let (lower, higher_type) = if s1 < s2 { (t1, t2) } else { (t2, t1) };
        let higher_kind = higher_type.kind();

        // If higher is float/complex, we usually just take the higher type's size,
        // unless lower type is actually larger?
        // e.g. i64 + f32 -> f64 (NumPy)
        // Check special case: if integer meets float/complex
        if matches!(lower.kind(), Integer | Unsigned | Bool)
            && matches!(higher_kind, Float | Complex)
        {
            return Some(promote_int_float_complex(lower, higher_type));
        }

        // Default: use the higher kind
        return Some(higher_type.clone());
    }

    // Same kind group
    match (k1, k2) {
        (Bool, Bool) => Some(Dtype::Bool),
        (Integer, Integer) => promote_signed(t1, t2),
        (Unsigned, Unsigned) => promote_unsigned(t1, t2),
        (Integer, Unsigned) => promote_mixed_int(t1, t2),
        (Unsigned, Integer) => promote_mixed_int(t2, t1), // swap
        (Float, Float) => promote_float(t1, t2),
        (Complex, Complex) => promote_complex(t1, t2),
        (Datetime, Datetime) => promote_datetime(t1, t2),
        (String, String) => {
            // Check if either is Unicode
            let is_unicode =
                matches!(t1, Dtype::Unicode { .. }) || matches!(t2, Dtype::Unicode { .. });

            let l1 = match t1 {
                Dtype::String { length } => length.unwrap_or_else(|| 0),
                Dtype::Unicode { length } => length.unwrap_or_else(|| 0),
                _ => 0,
            };
            let l2 = match t2 {
                Dtype::String { length } => length.unwrap_or_else(|| 0),
                Dtype::Unicode { length } => length.unwrap_or_else(|| 0),
                _ => 0,
            };

            let max_len = l1.max(l2);

            if is_unicode {
                Some(Dtype::Unicode {
                    length: Some(max_len),
                })
            } else {
                Some(Dtype::String {
                    length: Some(max_len),
                })
            }
        }
        (Bytes, Bytes) => {
            let l1 = match t1 {
                Dtype::Bytes { length } => *length,
                _ => 0,
            };
            let l2 = match t2 {
                Dtype::Bytes { length } => *length,
                _ => 0,
            };
            Some(Dtype::Bytes { length: l1.max(l2) })
        }
        _ => {
            if t1 == t2 {
                Some(t1.clone())
            } else {
                Some(Dtype::Object) // Fallback to Object if mixed and no rule
            }
        }
    }
}

/// Promote multiple types to a common result type
pub fn result_type(types: &[&Dtype]) -> Option<Dtype> {
    if types.is_empty() {
        return None;
    }
    let mut res = types[0].clone();
    for i in 1..types.len() {
        res = promote_types(&res, types[i])?;
    }
    Some(res)
}

fn promote_int_float_complex(int_dtype: &Dtype, float_complex_dtype: &Dtype) -> Dtype {
    let f_size = float_complex_dtype.itemsize();
    let i_size = int_dtype.itemsize();

    // NumPy logic:
    // i8/i16/u8/u16 + f32 -> f32
    // i32/u32 + f32 -> f64 (f32 only has 24 bits of precision)
    // i64/u64 + f32/f64 -> f64

    if float_complex_dtype.kind() == DtypeKind::Float {
        if f_size < 8 && i_size >= 4 {
            return Dtype::Float64 { byteorder: None };
        }
        float_complex_dtype.clone()
    } else {
        // Complex
        // c8 (2x f32) + i32 -> c16 (2x f64)
        if f_size < 16 && i_size >= 4 {
            return Dtype::Complex128 { byteorder: None };
        }
        float_complex_dtype.clone()
    }
}

fn promote_signed(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let s1 = t1.itemsize();
    let s2 = t2.itemsize();
    if s1 >= s2 {
        Some(t1.clone())
    } else {
        Some(t2.clone())
    }
}

fn promote_unsigned(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let s1 = t1.itemsize();
    let s2 = t2.itemsize();
    if s1 >= s2 {
        Some(t1.clone())
    } else {
        Some(t2.clone())
    }
}

fn promote_mixed_int(signed: &Dtype, unsigned: &Dtype) -> Option<Dtype> {
    let s_signed = signed.itemsize();
    let s_unsigned = unsigned.itemsize();

    // If signed type is strictly larger than unsigned, it can hold it (e.g. i16 can hold u8)
    if s_signed > s_unsigned {
        return Some(signed.clone());
    }

    // If unsigned is same or larger, we need a larger signed type
    // e.g. i32 + u32 -> i64
    // i64 + u64 -> float64 (NumPy fallback when it cannot fit in signed integer)
    if s_unsigned >= 8 {
        return Some(Dtype::Float64 { byteorder: None });
    }

    size_to_signed(s_unsigned * 2)
}

fn promote_float(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let s1 = t1.itemsize();
    let s2 = t2.itemsize();
    if s1 >= s2 {
        Some(t1.clone())
    } else {
        Some(t2.clone())
    }
}

fn promote_complex(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let s1 = t1.itemsize();
    let s2 = t2.itemsize();
    if s1 >= s2 {
        Some(t1.clone())
    } else {
        Some(t2.clone())
    }
}

/// Promote types specifically for division operations.
///
/// NumPy's `true_divide` (/) always promotes to at least float64 (or float32).
pub fn promote_types_division(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    let base = promote_types(t1, t2)?;
    match base.kind() {
        DtypeKind::Integer | DtypeKind::Unsigned | DtypeKind::Bool => {
            Some(Dtype::Float64 { byteorder: None })
        }
        _ => Some(base),
    }
}

/// Promote types for bitwise operations.
///
/// Only valid for integer and boolean types.
pub fn promote_types_bitwise(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    use DtypeKind::*;
    match (t1.kind(), t2.kind()) {
        (Integer | Unsigned | Bool, Integer | Unsigned | Bool) => promote_types(t1, t2),
        _ => None,
    }
}

fn promote_datetime(t1: &Dtype, t2: &Dtype) -> Option<Dtype> {
    // For datetime/timedelta, NumPy usually takes the finer unit or errors if incompatible
    // For simplicity, we'll return the first one if they match, or error for now
    if t1 == t2 {
        Some(t1.clone())
    } else {
        None
    }
}

fn size_to_signed(size: usize) -> Option<Dtype> {
    match size {
        1 => Some(Dtype::Int8 { byteorder: None }),
        2 => Some(Dtype::Int16 { byteorder: None }),
        4 => Some(Dtype::Int32 { byteorder: None }),
        8 => Some(Dtype::Int64 { byteorder: None }),
        _ => Some(Dtype::Float64 { byteorder: None }), // Fallback
    }
}

fn size_to_unsigned(size: usize) -> Option<Dtype> {
    match size {
        1 => Some(Dtype::UInt8 { byteorder: None }),
        2 => Some(Dtype::UInt16 { byteorder: None }),
        4 => Some(Dtype::UInt32 { byteorder: None }),
        8 => Some(Dtype::UInt64 { byteorder: None }),
        _ => None,
    }
}
