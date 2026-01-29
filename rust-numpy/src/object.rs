// Object type definitions for NumPy compatibility

/// Object type for storing arbitrary Python objects
#[derive(Debug)]
pub struct Object(pub Box<dyn std::any::Any>);
