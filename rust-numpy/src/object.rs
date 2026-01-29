// Object type definitions for NumPy compatibility

/// Object type for storing arbitrary Python objects
#[derive(Debug, Clone)]
pub struct Object(pub std::sync::Arc<dyn std::any::Any + Send + Sync>);
