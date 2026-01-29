/// Modern random number generation API
///
/// This sub-module provides modern Generator/BitGenerator API
/// that matches NumPy's current random module structure.
pub use super::bit_generator::{BitGenerator, PCG64};
pub use super::generator::Generator;
pub use super::random_state::RandomState;
pub use super::{default_rng, default_rng_with_seed};
