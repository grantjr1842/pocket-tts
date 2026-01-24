# NumPy 2.4 Correctness-First Parity Checklist

This checklist is a **behavioral** audit companion to `PARITY_BASELINE_NUMPY_2_4.csv`.
Complete items before marking a callable as parity-complete.

## Global Correctness Gates (applies to all modules)
- [ ] **Signature parity**: positional/keyword arguments, defaults, and return shapes match NumPy 2.4.
- [ ] **Broadcasting parity**: shape rules, stride-0 behavior, and mixed-dim broadcasting are correct.
- [ ] **Dtype parity**: promotions, casting rules, and dtype-specific outputs match NumPy 2.4.
- [ ] **Error parity**: error type and message semantics are aligned (e.g., ValueError, LinAlgError).
- [ ] **View semantics**: non-contiguous arrays, slicing views, and offset handling behave correctly.
- [ ] **`where`/mask parity**: masked execution and `where=` semantics match NumPy.
- [ ] **NaN/Inf parity**: NaN propagation and edge behavior match NumPy.

## numpy (top-level)
- [ ] Array creation (`array`, `arange`, `linspace`, `logspace`, `geomspace`) match NumPy’s dtype/shape defaults.
- [ ] Top-level `dot`, `inner`, `outer`, `matmul`, `kron`, `vdot`, `trace`, `cross` match NumPy semantics.
- [ ] `where` broadcasting and scalar/array combinations match NumPy.
- [ ] `arg*`/`sort*` outputs (dtype, axis defaults) match NumPy.
- [ ] Top-level wrappers call correct ufunc kernels (no aliasing bugs).

## numpy.ndarray Methods
- [ ] Implement missing methods per baseline (see CSV), matching NumPy signatures and return types.
- [ ] Validate in-place vs out-of-place behavior (`sort`, `resize`, `fill`, etc.).
- [ ] Ensure method/ufunc results are identical to top-level numpy.* equivalents.

## numpy.ndarray Properties
- [ ] `T`, `real`, `imag`, `itemsize`, `nbytes`, `flags`, `base`, `flat` parity validated.
- [ ] Views vs copies reflected in `base` and `flags`.

## numpy.linalg
- [ ] Matrix shape rules (1D/2D/batched) and broadcasting match NumPy.
- [ ] Error parity: singular matrices, non-square, invalid shapes, convergence failures.
- [ ] Numerical accuracy vs NumPy on representative cases (including ill-conditioned matrices).
- [ ] `matrix_norm`, `vector_norm`, `svdvals`, `tensordot`, `tensorinv`, `tensorsolve` added/verified.
- [ ] Output dtype conventions (e.g., float for real, complex for complex) match NumPy.

## numpy.fft
- [ ] `n`, `s`, `axes`, `norm` parameters behave like NumPy.
- [ ] Real/complex output types and shapes match NumPy (`rfft*`, `irfft*`).
- [ ] `fftshift`/`ifftshift` behavior matches NumPy for odd/even sizes.
- [ ] Multidimensional FFTs validated against NumPy for axis order and shape.

## numpy.random
- [ ] Generator API parity (`Generator`, `BitGenerator`, `default_rng`, `SeedSequence`).
- [ ] Reproducibility and seeding match NumPy reference behavior.
- [ ] Distribution parameter validation matches NumPy (domain checks, errors).
- [ ] Output shapes and dtype behavior match NumPy (float vs int distributions).

## numpy.ma (Masked Arrays)
- [ ] Mask propagation rules match NumPy (binary ops, reductions, comparisons).
- [ ] `filled`, `fill_value`, and `mask` semantics match NumPy.
- [ ] Masked reductions match NumPy defaults (axis, keepdims).
- [ ] `choose`, `compress`, `where` parity verified with masks.

## numpy.char
- [ ] Unicode case behavior (`upper`, `lower`, `title`, `swapcase`) matches NumPy.
- [ ] Encoding/decoding behavior (`encode`, `decode`) matches NumPy output types.
- [ ] Vectorized split/join semantics match NumPy for arrays of strings.

## numpy.testing
- [ ] Assertion helper parity (exception types, error messages, tolerance behavior).
- [ ] `assert_allclose`/`assert_array_equal` semantics match NumPy.

## numpy.typing
- [ ] `NDArray` and `NBitBase` definitions implemented and documented.
- [ ] Type aliases reflect NumPy 2.4 typing semantics.

## Conformance & Regression
- [ ] Add golden tests comparing against NumPy (saved outputs) for each module.
- [ ] Coverage exceeds 80% for core modules (array, ufunc, linalg).
- [ ] Tests include NaN/Inf/empty/broadcast edge cases.

