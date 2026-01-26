//! NumPy linalg module exports
//!
//! This module re-exports all NumPy linalg functions to match Python's numpy.linalg namespace.
//! Functions are organized by category for maintainability.

pub use crate::decompositions::{cholesky, qr, svd};

pub use crate::solvers::{lstsq, solve, tensorinv, tensorsolve};

pub use crate::eigen::{eig, eigh, eigvals, eigvalsh};

pub use crate::norms::{matrix_norm, norm, vector_norm};

pub use crate::products::{det, slogdet, trace};

pub use super::{Complex, Float, LinalgScalar, One, Zero};
