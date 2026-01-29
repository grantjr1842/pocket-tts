# [Sub-Issue] Create Protocol Traits (SupportsArray, SupportsDType) (#572)

**Issue #597**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/597  
**Created:** 2026-01-29T03:10:37Z  
**Updated:** 2026-01-29T03:24:05Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:typing, ralph/numpy:impl  

---

## Parent Issue
Part of #572 - typing module incomplete vs NumPy.typing

## Scope
Create Protocol Traits for ArrayLike (SupportsArray, SupportsDType)

## Background
NumPy typing uses protocols for duck typing. rust-numpy needs equivalent traits.

## Tasks

### 1. Create SupportsArray Protocol
```rust
// src/typing/protocols.rs
pub trait SupportsArray {
    type Element: Clone + Default + 'static;
    
    /// Return array representation
    fn __array__(&self) -> Result<Array<Self::Element>, NumPyError>;
    
    /// Optional: Return shape without copying
    fn __array_shape__(&self) -> Option<&[usize]>;
    
    /// Optional: Return dtype
    fn __array_dtype__(&self) -> Dtype;
}

// Implement for common types
impl<T: Clone + Default + 'static> SupportsArray for Vec<T> {
    type Element = T;
    fn __array__(&self) -> Result<Array<T>, NumPyError> {
        Ok(Array::from_vec(self.clone()))
    }
    // ...
}

impl<T: Clone + Default + 'static, const N: usize> SupportsArray for [T; N] {
    type Element = T;
    fn __array__(&self) -> Result<Array<T>, NumPyError> {
        Ok(Array::from_vec(self.to_vec()))
    }
}

impl<T: Clone + Default + 'static> SupportsArray for Array<T> {
    type Element = T;
    fn __array__(&self) -> Result<Array<T>, NumPyError> {
        Ok(self.clone())
    }
}
```

### 2. Create SupportsDType Protocol
```rust
pub trait SupportsDType {
    fn __dtype__(&self) -> Dtype;
}

// Implement for types that have dtype info
impl SupportsDType for Dtype {
    fn __dtype__(&self) -> Dtype {
        self.clone()
    }
}

impl<T: HasDtype> SupportsDType for Array<T> {
    fn __dtype__(&self) -> Dtype {
        T::dtype()
    }
}
```

### 3. Create NestedSequence Protocol
```rust
pub trait NestedSequence<T> {
    /// Get element or nested sequence at index
    fn get_nested(&self, index: usize) -> Option<Either<&T, &dyn NestedSequence<T>>>;
    
    /// Length of this sequence level
    fn len(&self) -> usize;
    
    /// Check if this is the innermost level (contains T, not NestedSequence)
    fn is_leaf(&self) -> bool;
    
    /// Flatten to vector
    fn flatten(&self) -> Vec<T>;
}

// Implement for Vec<Vec<...>> types
impl<T> NestedSequence<T> for Vec<T> {
    fn get_nested(&self, index: usize) -> Option<Either<&T, &dyn NestedSequence<T>>> {
        self.get(index).map(|x| Either::Left(x))
    }
    
    fn len(&self) -> usize { self.len() }
    fn is_leaf(&self) -> bool { true }
    fn flatten(&self) -> Vec<T> { self.clone() }
}

impl<T: Clone> NestedSequence<T> for Vec<Vec<T>> {
    fn get_nested(&self, index: usize) -> Option<Either<&T, &dyn NestedSequence<T>>> {
        self.get(index).map(|x| Either::Right(x as &dyn NestedSequence<T>))
    }
    
    fn len(&self) -> usize { self.len() }
    fn is_leaf(&self) -> bool { false }
    fn flatten(&self) -> Vec<T> {
        self.iter().flat_map(|v| v.clone()).collect()
    }
}
```

### 4. Update ArrayLike Type
```rust
// More comprehensive ArrayLike
pub enum ArrayLike<'a, T: Clone + Default + 'static> {
    Array(Array<T>),
    Vec(Vec<T>),
    Slice(&'a [T]),
    ArrayRef(&'a Array<T>),
    SupportsArray(Box<dyn SupportsArray<Element = T> + 'a>),
    Nested(Box<dyn NestedSequence<T> + 'a>),
    Scalar(T),
}

impl<'a, T: Clone + Default + 'static> ArrayLike<'a, T> {
    pub fn to_array(&self) -> Result<Array<T>, NumPyError> {
        match self {
            ArrayLike::Array(arr) => Ok(arr.clone()),
            ArrayLike::Vec(v) => Ok(Array::from_vec(v.clone())),
            ArrayLike::Slice(s) => Ok(Array::from_slice(s)),
            ArrayLike::SupportsArray(sa) => sa.__array__(),
            ArrayLike::Nested(n) => Ok(Array::from_vec(n.flatten())),
            ArrayLike::Scalar(s) => Ok(Array::from_vec(vec![s.clone()])),
            // ...
        }
    }
}
```

## Files to Create/Modify
- NEW: `src/typing/protocols.rs` - Protocol traits
- `src/typing/mod.rs` - Export protocols and update ArrayLike

## Acceptance Criteria
- [ ] SupportsArray trait defined
- [ ] SupportsDType trait defined
- [ ] NestedSequence trait defined
- [ ] ArrayLike updated to use protocols
- [ ] Tests for protocol implementations

## Conflict Avoidance
⚠️ **COORDINATION:**
- New file - minimal conflict
- May need Either type from crates.io or custom enum

