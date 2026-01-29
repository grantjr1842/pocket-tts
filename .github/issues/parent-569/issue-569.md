# rust-numpy: Top-level lib.rs exports don't match NumPy public API

**Issue #569**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/569  
**Created:** 2026-01-29T02:46:44Z  
**Updated:** 2026-01-29T03:23:37Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports, ralph/numpy:audit  

---

## Summary
The `rust-numpy/src/lib.rs` top-level exports do NOT match NumPy's public API structure. This issue tracks all missing, misplaced, or incorrectly exported items at the crate root level.

## Reference: NumPy's Top-Level Public API
NumPy exposes these at the top level (`numpy.*`):

### Array Creation (MUST BE AT ROOT)
- `np.array`, `np.zeros`, `np.ones`, `np.empty`, `np.full`
- `np.arange`, `np.linspace`, `np.logspace`, `np.geomspace`
- `np.eye`, `np.identity`, `np.diag`, `np.tri`, `np.tril`, `np.triu`
- `np.fromfunction`, `np.fromiter`, `np.fromfile`, `np.frombuffer`, `np.fromstring`, `np.loadtxt`, `np.genfromtxt`
- `np.copy`, `np.asarray`, `np.asanyarray`, `np.asmatrix`, `np.asfarray`, `np.ascontiguousarray`, `np.asfortranarray`, `np.require`

### Array Manipulation (MOSTLY MISSING FROM ROOT)
- `np.reshape`, `np.ravel`, `np.ndarray.flatten`
- `np.transpose`, `np.swapaxes`, `np.rollaxis`, `np.moveaxis`
- `np.concatenate`, `np.stack`, `np.vstack`, `np.hstack`, `np.dstack`, `np.column_stack`, `np.row_stack`
- `np.split`, `np.hsplit`, `np.vsplit`, `np.dsplit`, `np.array_split`
- `np.tile`, `np.repeat`, `np.delete`, `np.insert`, `np.append`, `np.resize`, `np.trim_zeros`, `np.unique`
- `np.flip`, `np.fliplr`, `np.flipud`, `np.roll`, `np.rot90`

### Mathematical Operations (PARTIALLY AT ROOT)
- `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.power`, `np.mod`, `np.remainder`, `np.divmod`
- `np.abs`, `np.absolute`, `np.fabs`, `np.sign`, `np.negative`, `np.positive`, `np.reciprocal`
- `np.floor`, `np.ceil`, `np.trunc`, `np.round`, `np.rint`, `np.fix`, `np.around`
- `np.exp`, `np.expm1`, `np.exp2`, `np.log`, `np.log10`, `np.log2`, `np.log1p`
- `np.sqrt`, `np.cbrt`, `np.square`
- `np.sin`, `np.cos`, `np.tan`, `np.arcsin`, `np.arccos`, `np.arctan`, `np.arctan2`
- `np.hypot`, `np.degrees`, `np.radians`, `np.deg2rad`, `np.rad2deg`, `np.unwrap`

### Logic/Comparison (PARTIALLY MISSING)
- `np.greater`, `np.greater_equal`, `np.less`, `np.less_equal`, `np.equal`, `np.not_equal`, `np.logical_and`, `np.logical_or`, `np.logical_not`, `np.logical_xor`
- `np.all`, `np.any`, `np.isfinite`, `np.isinf`, `np.isnan`, `np.isnat`, `np.signbit`

### Statistical Functions (IN statistics MODULE, NOT ROOT)
In NumPy these are at ROOT: `np.mean`, `np.std`, `np.var`, `np.min`, `np.max`, `np.sum`, `np.prod`, `np.cumsum`, `np.cumprod`, `np.percentile`, `np.quantile`, `np.median`, `np.average`

### Constants (PRESENT - KEEP)
- `np.pi`, `np.e`, `np.euler_gamma`, `np.inf`, `np.nan`, `np.newaxis`, `np.Infinity`, `np.NaN`

### Data Types (PARTIALLY MISSING)
- `np.bool_`, `np.byte`, `np.ubyte`, `np.short`, `np.ushort`, `np.intc`, `np.uintc`, `np.int_`, `np.uint`
- `np.longlong`, `np.ulonglong`, `np.single`, `np.double`, `np.longdouble`
- `np.csingle`, `np.cdouble`, `np.clongdouble`
- `np.int8`, `np.int16`, `np.int32`, `np.int64`, `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64`
- `np.float16`, `np.float32`, `np.float64`, `np.float128`
- `np.complex64`, `np.complex128`, `np.complex256`
- `np.dtype`, `np.generic`, `np.number`, `np.integer`, `np.inexact`, `np.floating`, `np.complexfloating`

## Current State vs Expected

### ✅ Correctly Exported at Root
```rust
pub use array::Array;
pub use array_creation::{array, array2string, array_repr, array_str, asanyarray, asarray, ...};
pub use dtype::{Casting, Dtype, DtypeKind};
pub use error::{NumPyError, Result};
pub use reductions::{all, all_bool, any, any_bool, argmax, argmin, ...};
pub use constants::*;
```

### ❌ MISSING from Root (but should be there)
1. **Array manipulation functions** - ONLY `expand_dims` is exported; missing:
   - `reshape`, `ravel`, `flatten` (methods on Array, but not free functions)
   - `transpose`, `swapaxes`, `rollaxis`, `moveaxis`
   - `concatenate`, `stack`, `vstack`, `hstack`, `dstack`, `column_stack`, `row_stack`
   - `split`, `hsplit`, `vsplit`, `dsplit`, `array_split`
   - `tile`, `repeat`, `delete`, `insert`, `append`, `resize`, `trim_zeros`
   - `flip`, `fliplr`, `flipud`, `roll`, `rot90`

2. **Mathematical ufuncs** - NOT exported at root (only in `math_ufuncs` module):
   - `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan`, `arctan2`
   - `exp`, `expm1`, `exp2`, `log`, `log10`, `log2`, `log1p`
   - `sqrt`, `cbrt`, `square`
   - `floor`, `ceil`, `trunc`, `round`, `rint`

3. **Logical/comparison ufuncs** - NOT exported at root:
   - `greater`, `greater_equal`, `less`, `less_equal`, `equal`, `not_equal`
   - `logical_and`, `logical_or`, `logical_not`, `logical_xor`

4. **Type constructors** - Many missing:
   - `bool_`, `byte`, `ubyte`, `short`, `ushort`, `intc`, `uintc`, `int_`, `uint`
   - `longlong`, `ulonglong`, `longdouble`, `clongdouble`
   - `complex64`, `complex128`, `complex256`

5. **Statistical functions** - Should be at root but only in `statistics` module:
   - `mean`, `std`, `var`, `min`, `max`, `sum`, `prod`, `cumsum`, `cumprod`
   - `median`, `average`, `percentile`, `quantile`

### ❌ INCORRECTLY at Root (should be module-only)
- `utils_promote_types` - should be `np.promote_types`, not `utils_promote_types`
- Various utility placeholders (`get_include`, `show_config`, `test`, `info`)

## Acceptance Criteria
- [ ] All NumPy top-level array creation functions exported at crate root
- [ ] All NumPy top-level array manipulation functions exported at crate root
- [ ] All NumPy mathematical ufuncs exported at crate root
- [ ] All NumPy logical/comparison functions exported at crate root
- [ ] All NumPy statistical functions exported at crate root
- [ ] All NumPy data type constructors exported at crate root
- [ ] Remove internal/non-public items from root exports
- [ ] Add `#[doc(hidden)]` to internal re-exports that shouldn't be public
- [ ] Verify with a script that compares `numpy.*` in Python to `rust_numpy::*` in Rust

## Verification Plan
```python
# Python reference
import numpy as np
public_api = set(dir(np)) - set(['__doc__', '__file__', '__name__', '__package__', '__version__'])

# Rust should match:
// use rust_numpy::*;
// Check that all items in public_api have corresponding exports
```

## Affected Files
- `rust-numpy/src/lib.rs` - Main export declarations
- `rust-numpy/src/array_manipulation.rs` - Add free functions
- `rust-numpy/src/math_ufuncs.rs` - Add root re-exports
- `rust-numpy/src/comparison_ufuncs.rs` - Add root re-exports
- `rust-numpy/src/statistics.rs` - Add root re-exports
- `rust-numpy/src/utils.rs` - Clean up type placeholders


---

## Sub-Issues (for parallel development)

| Issue | Title | Scope |
|-------|-------|-------|
| #576 | Array Manipulation Part 1: Join/Stack/Split | concatenate, stack, vstack, hstack, split, etc. |
| #577 | Array Manipulation Part 2: Reshape/Rearrange | reshape, transpose, flip, roll, rot90, etc. |
| #578 | Array Manipulation Part 3: Add/Remove Elements | append, insert, delete, unique, trim_zeros |
| #579 | Math Ufuncs: Trigonometric/Hyperbolic at Root | sin, cos, tan, arcsin, sinh, degrees, radians |
| #580 | Math Ufuncs: Exp/Log/Power at Root | exp, log, sqrt, power, expm1, log1p |
| #581 | Comparison and Logical Ops at Root | greater, less, equal, logical_and, isfinite |
| #582 | Statistical Functions at Root | mean, std, var, min, max, sum, percentile, median |

**Coordination Notes:**
- Each sub-issue can be worked on independently
- Minimize conflicts by focusing ONLY on assigned functions
- Do not modify other sub-issues' code

