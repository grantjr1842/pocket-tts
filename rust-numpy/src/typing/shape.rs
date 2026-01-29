/// Shape parameter for type-level shape checking
///
/// This trait provides compile-time and runtime shape information for arrays,
/// similar to NumPy's shape parameter in typing.NDArray.
pub trait Shape: 'static {
    /// Number of dimensions (0 for dynamic)
    const NDIM: usize;
    
    /// Total size (None for dynamic or unknown)
    const SIZE: Option<usize>;
    
    /// Get shape as slice (for runtime operations)
    fn as_slice(&self) -> &[usize];
}

/// Dynamic shape (unknown at compile time)
///
/// Represents arrays with runtime-determined shape.
#[derive(Clone, Debug, Default)]
pub struct Dynamic;

impl Shape for Dynamic {
    const NDIM: usize = 0;  // Unknown at compile time
    const SIZE: Option<usize> = None;
    
    fn as_slice(&self) -> &[usize] {
        &[]
    }
}

/// Fixed 1D shape with const generic
#[derive(Clone, Debug, Default)]
pub struct Shape1<const N: usize>;

impl<const N: usize> Shape for Shape1<N> {
    const NDIM: usize = 1;
    const SIZE: Option<usize> = Some(N);
    
    fn as_slice(&self) -> &[usize] {
        &[N]
    }
}

/// Fixed 2D shape with const generics
#[derive(Clone, Debug, Default)]
pub struct Shape2<const M: usize, const N: usize>;

impl<const M: usize, const N: usize> Shape for Shape2<M, N> {
    const NDIM: usize = 2;
    const SIZE: Option<usize> = Some(M * N);
    
    fn as_slice(&self) -> &[usize] {
        &[M, N]
    }
}

/// Fixed 3D shape with const generics
#[derive(Clone, Debug, Default)]
pub struct Shape3<const D0: usize, const D1: usize, const D2: usize>;

impl<const D0: usize, const D1: usize, const D2: usize> Shape for Shape3<D0, D1, D2> {
    const NDIM: usize = 3;
    const SIZE: Option<usize> = Some(D0 * D1 * D2);
    
    fn as_slice(&self) -> &[usize] {
        &[D0, D1, D2]
    }
}

/// Fixed 4D shape with const generics
#[derive(Clone, Debug, Default)]
pub struct Shape4<const D0: usize, const D1: usize, const D2: usize, const D3: usize>;

impl<const D0: usize, const D1: usize, const D2: usize, const D3: usize> Shape for Shape4<D0, D1, D2, D3> {
    const NDIM: usize = 4;
    const SIZE: Option<usize> = Some(D0 * D1 * D2 * D3);
    
    fn as_slice(&self) -> &[usize] {
        &[D0, D1, D2, D3]
    }
}

/// Runtime shape (determined at runtime)
///
/// Represents arrays with shape known only at runtime.
#[derive(Clone, Debug)]
pub struct RuntimeShape(pub Vec<usize>);

impl Default for RuntimeShape {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl Shape for RuntimeShape {
    const NDIM: usize = 0;  // Dynamic
    const SIZE: Option<usize> = None;
    
    fn as_slice(&self) -> &[usize] {
        &self.0
    }
}

impl RuntimeShape {
    /// Create a new RuntimeShape from a vector
    pub fn new(shape: Vec<usize>) -> Self {
        Self(shape)
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.0.len()
    }
    
    /// Get the total size
    pub fn size(&self) -> usize {
        self.0.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape1() {
        let shape = Shape1::<10>;
        assert_eq!(shape.NDIM, 1);
        assert_eq!(shape.SIZE, Some(10));
        assert_eq!(shape.as_slice(), &[10]);
    }

    #[test]
    fn test_shape2() {
        let shape = Shape2::<3, 4>;
        assert_eq!(shape.NDIM, 2);
        assert_eq!(shape.SIZE, Some(12));
        assert_eq!(shape.as_slice(), &[3, 4]);
    }

    #[test]
    fn test_shape3() {
        let shape = Shape3::<2, 3, 4>;
        assert_eq!(shape.NDIM, 3);
        assert_eq!(shape.SIZE, Some(24));
        assert_eq!(shape.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_shape4() {
        let shape = Shape4::<2, 3, 4, 5>;
        assert_eq!(shape.NDIM, 4);
        assert_eq!(shape.SIZE, Some(120));
        assert_eq!(shape.as_slice(), &[2, 3, 4, 5]);
    }

    #[test]
    fn test_dynamic() {
        let shape = Dynamic;
        assert_eq!(shape.NDIM, 0);
        assert_eq!(shape.SIZE, None);
        assert_eq!(shape.as_slice(), &[]);
    }

    #[test]
    fn test_runtime_shape() {
        let shape = RuntimeShape::new(vec![3, 4, 5]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.size(), 60);
        assert_eq!(shape.as_slice(), &[3, 4, 5]);
    }
}
