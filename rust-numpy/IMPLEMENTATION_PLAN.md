# Rust NumPy Implementation Plan

**Purpose:** Complete implementation checklist for achieving full NumPy 2.4 API parity  
**Reference:** Based on comparison in [`RUST_NUMPY_EXPORT_COMPARISON.md`](RUST_NUMPY_EXPORT_COMPARISON.md)

---

## Priority 1: Core Array Creation & Manipulation Functions (~85% coverage)

### Array Creation Functions

- [ ] `empty(shape, dtype=None, order='C', *, like=None)` - Create a new array of given shape and type, without initializing entries
- [ ] `empty_like(prototype, dtype=None, order='K', subok=True, shape=None)` - Return a new array with the same shape and type as a given array
- [ ] `full(shape, fill_value, dtype=None, order='C')` - Return a new array of given shape and type, filled with fill_value
- [ ] `full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None)` - Return a full array with the same shape and type as a given array
- [ ] `ones(shape, dtype=None, order='C', *, like=None)` - Return a new array of given shape and type, filled with ones
- [ ] `ones_like(a, dtype=None, order='K', subok=True, shape=None)` - Return an array of ones with the same shape and type as a given array
- [ ] `zeros(shape, dtype=None, order='C', *, like=None)` - Return a new array of given shape and type, filled with zeros
- [ ] `zeros_like(a, dtype=None, order='K', subok=True, shape=None)` - Return an array of zeros with the same shape and type as a given array
- [ ] `identity(n, dtype=None, *, like=None)` - Return the identity array
- [ ] `eye(N, M=None, k=0, dtype=<class 'float'>, order='C', *, like=None)` - Return a 2-D array with ones on the diagonal and zeros elsewhere
- [ ] `diag(v, k=0)` - Extract a diagonal or construct a diagonal array
- [ ] `diagflat(v, k=0)` - Create a 2-D array with the flattened input as a diagonal
- [ ] `diag_indices(n, ndim, k=0)` - Return the indices to access the main diagonal of an array
- [ ] `diag_indices_from(n, ndim, k=0)` - Return the indices to access the main diagonal of an array with offset
- [ ] `tri(N, k=0, m=None)` - An array with ones at and below the given diagonal and zeros elsewhere
- [ ] `tril(m, k=0)` - Lower triangle of an array
- [ ] `triu(m, k=0)` - Upper triangle of an array
- [ ] `tril_indices(n, k=0, m=None)` - Return the indices for the lower-triangle of an array
- [ ] `triu_indices(n, k=0, m=None)` - Return the indices for the upper-triangle of an array
- [ ] `vander(x, N=None, increasing=False)` - Generate a Vandermonde matrix
- [ ] `matrices(m, k=None)` - Stack a sequence of matrices along a new leading axis
- [ ] `block(arrays)` - Assemble an nd-array from nested list of blocks
- [ ] `bmat(objects, *, dtype=None, ldict=None)` - Build a matrix object from a string, nested sequence, or array
- [ ] `indices(dimensions, dtype=int, sparse=False)` - Return an array representing the indices of a grid
- [ ] `mgrid(*xi, copy=True, sparse=False, indexing='ij')` - Return a dense multi-dimensional "meshgrid"
- [ ] `ogrid(*xi, shape=None, sparse=False, indexing='ij')` - Return an open multi-dimensional "meshgrid"
- [ ] `atleast_1d(*arys)` - Ensure inputs are at least 1-D
- [ ] `atleast_2d(*arys)` - Ensure inputs are at least 2-D
- [ ] `atleast_3d(*arys)` - Ensure inputs are at least 3-D
- [ ] `atleast_nd(*arys, ndim)` - Ensure inputs are at least ndim-D
- [ ] `broadcast_arrays(*args)` - Broadcast any number of arrays against each other
- [ ] `broadcast_shapes(*args)` - Broadcast the input shapes against each other
- [ ] `broadcast_to(array, shape)` - Broadcast an array to a given shape
- [ ] `broadcast(shape)` - Produce an object that mimics broadcasting
- [ ] `ravel_multi_index(multi_index, dims, mode='raise')` - Convert a tuple of index arrays into an array of flat indices
- [ ] `unravel_index(indices, dims, mode='raise')` - Convert a flat index array into a tuple of index arrays

### Array Manipulation Functions

- [ ] `append(arr, values, axis=None)` - Append values to the end of an array
- [ ] `insert(arr, obj, axis, values)` - Insert values along the given axis before the given indices
- [ ] `delete(arr, obj, axis)` - Delete sub-arrays along an axis
- [ ] `concatenate(arrays, axis=0, out=None, dtype=None, casting="same_kind")` - Join a sequence of arrays along an existing axis
- [ ] `stack(arrays, axis=0, out=None)` - Join a sequence of arrays along a new axis
- [ ] `hstack(tup)` - Stack arrays in sequence horizontally (column wise)
- [ ] `vstack(tup)` - Stack arrays in sequence vertically (row wise)
- [ ] `dstack(tup)` - Stack arrays in sequence depth wise (along third axis)
- [ ] `column_stack(tup)` - Stack 1-D arrays as columns into a 2-D array
- [ ] `row_stack(tup)` - Stack 1-D arrays as rows into a 2-D array
- [ ] `split(ary, indices_or_sections, axis=0)` - Split an array into multiple sub-arrays
- [ ] `array_split(ary, indices_or_sections, axis=0)` - Split an array into multiple sub-arrays
- [ ] `dsplit(ary, indices_or_sections)` - Split array into multiple sub-arrays along the 3rd axis (depth)
- [ ] `hsplit(ary, indices_or_sections)` - Split array into multiple sub-arrays along the 2nd axis (columns)
- [ ] `vsplit(ary, indices_or_sections)` - Split array into multiple sub-arrays along the 1st axis (rows)
- [ ] `tile(A, reps)` - Construct an array by repeating A the number of times given by reps
- [ ] `repeat(a, repeats, axis=None)` - Repeat elements of an array
- [ ] `flip(m, axis=None)` - Reverse the order of elements along the given axis
- [ ] `fliplr(m)` - Flip array in the left/right direction
- [ ] `flipud(m)` - Flip array in the up/down direction
- [ ] `roll(a, shift, axis=None)` - Roll array elements along a given axis
- [ ] `rot90(m, k=1, axes=(0,1))` - Rotate an array by 90 degrees in the plane specified by axes
- [ ] `ediff1d(ary, to_end=None, to_begin=None)` - The differences between consecutive elements of an array
- [ ] `gradient(f, *varargs, axis=None, edge_order=1)` - The gradient of an N-dimensional array
- [ ] `trim_zeros(f)` - Trim the leading and trailing zeros from a 1-D array or sequence
- [ ] `select(condlist, choicelist, default=0)` - Return an array drawn from elements in choicelist, depending on conditions
- [ ] `place(arr, mask, vals)` - Change elements of an array based on conditional and input values
- [ ] `put(a, ind, v)` - Replaces specified elements of an array with given values
- [ ] `put_along_axis(arr, indices, axis, values)` - Put values into the destination array along given axis
- [ ] `putmask(a, mask, values)` - Put values into the destination array using a mask
- [ ] `fill_diagonal(a, val, wrap=False)` - Fill the main diagonal of the given array
- [ ] `resize(a, new_shape)` - Return a new array with the specified shape
- [ ] `moveaxis(a, source, destination)` - Move axes of an array to new positions
- [ ] `rollaxis(a, axis, start=0)` - Roll the specified axis backwards, until it lies in a given position
- [ ] `swapaxes(a, axis1, axis2)` - Interchange two axes of an array
- [ ] `ndenumerate(arr)` - Multidimensional index iterator
- [ ] `ndindex(shape)` - Multidimensional index iterator over a given shape
- [ ] `nditer(op, flags=None, op_flags=None, op_dtypes=None, order='C', casting='safe', op_axes=None, itershape=None)` - Efficient multi-dimensional iterator object to iterate over arrays
- [ ] `nested_iters(op)` - Create nditer that uses nested iteration
- [ ] `flatiter(a)` - Flat iterator object to iterate over arrays
- [ ] `flatnonzero(a)` - Return indices that are non-zero in the flattened version of a
- [ ] `nonzero(a)` - Return the indices of the elements that are non-zero
- [ ] `count_nonzero(a)` - Count the number of non-zero elements in the array
- [ ] `argpartition(a, kth, axis=-1, kind='introselect', order=None)` - Perform an indirect partition along the given axis
- [ ] `partition(a, kth, axis=-1, kind='introselect', order=None)` - Partition of an array
- [ ] `choose(a, choices, out=None, mode='raise')` - Construct an array from an index array and a list of arrays
- [ ] `compress(condition, a, axis=None, out=None)` - Return selected slices of an array along given axis
- [ ] `extract(condition, arr)` - Return the elements of an array that satisfy some condition
- [ ] `take(a, indices, axis=None, out=None, mode='raise')` - Return an array formed from elements of a at the given indices
- [ ] `place(arr, mask, vals)` - Change elements of an array based on conditional and input values (duplicate of above)
- [ ] `repeat(a, repeats, axis=None)` - Repeat elements of an array (duplicate of above)
- [ ] `searchsorted(a, v, side='left', sorter=None)` - Find indices where elements of v should be inserted to maintain order
- [ ] `digitize(x, bins, right=False)` - Return the indices of the bins to which each value in x belongs
- [ ] `setfield(a, val, dtype, offset=0)` - Assign a value to a field-position, in a record array
- [ ] `getfield(a, dtype, offset=0)` - Fetch a field from a record array
- [ ] `packbits(myarray, axis)` - Packs the elements of a binary-valued array into bits in a uint8 array
- [ ] `unpackbits(myarray, axis)` - Unpacks elements of a uint8 array into a binary-valued output array
- [ ] `pad(array, pad_width, mode='constant', constant_values=0)` - Pads an array
- [ ] `piecewise(x, condlist, funclist, *args, **kwds)` - Evaluate a function piecewise on an array
- [ ] `select(condlist, choicelist, default=0)` - Return an array drawn from elements in choicelist, depending on conditions (duplicate of above)
- [ ] `shares_memory(a, b)` - Determine if two arrays share memory
- [ ] `may_share_memory(a, b)` - Determine if two arrays might share memory
- [ ] `set_printoptions(*args, **kwargs)` - Set printing options
- [ ] `get_printoptions()` - Return the current print options
- [ ] `setbufsize(size)` - Set size of buffer used in ufuncs
- [ ] `getbufsize()` - Return the size of buffer used in ufuncs
- [ ] `seterr(all, divide='warn', over='warn', under='warn', invalid='warn')` - Set how floating-point errors are handled
- [ ] `geterr()` - Get the current way of handling floating-point errors
- [ ] `seterrcall(func)` - Set the floating-point error callback function
- [ ] `geterrcall()` - Return the current callback for floating-point errors
- [ ] `errstate(**kwargs)` - Context manager for floating-point error handling
- [ ] `is_busday(dates, weekmask, busdaycal, holidays=None)` - Calculates which of the given dates are valid days
- [ ] `busday_offset(dates, offsets, roll, busdaycal, holidays=None)` - First adjusts the date to fall on a valid day
- [ ] `busdaycalendar(weekmask, holidays=None)` - Return a boolean array indicating valid days
- [ ] `show_config()` - Show libraries NumPy depends on and compile flags
- [ ] `show_runtime()` - Show runtime information
- [ ] `printoptions(*args, **kwargs)` - Print options and return the current options (duplicate of set_printoptions)

### Reduction Functions

- [ ] `amax(a, axis=None, out=None, keepdims=False, initial=<no value>, where=True)` - Return the maximum of an array or maximum along an axis
- [ ] `amin(a, axis=None, out=None, keepdims=False, initial=<no value>, where=True)` - Return the minimum of an array or minimum along an axis
- [ ] `nanargmax(a, axis=None, out=None, keepdims=False)` - Return the indices of the maximum values in the specified axis ignoring NaN
- [ ] `nanargmin(a, axis=None, out=None, keepdims=False)` - Return the indices of the minimum values in the specified axis ignoring NaN
- [ ] `nansum(a, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)` - Return the sum of array elements over a given axis treating Not a Numbers as zeros
- [ ] `nanprod(a, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)` - Return the product of array elements over a given axis treating Not a Numbers as ones
- [ ] `trace(a, offset=0, axis1=None, axis2=None, dtype=None, out=None)` - Return the sum along diagonals of the array
- [ ] `allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)` - Returns True if two arrays are element-wise equal within tolerance
- [ ] `isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)` - Returns True if two arrays are element-wise equal within tolerance (alias)
- [ ] `array_equal(a1, a2, equal_nan=False)` - True if two arrays have the same shape and elements
- [ ] `array_equiv(a1, a2)` - Returns True if input arrays are shape equivalent
- [ ] `result_type(*arrays_and_dtypes)` - Returns the type that results from applying the NumPy type promotion rules

### Mathematical Functions (Ufuncs)

- [ ] `exp2(x, /, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Compute 2**p element-wise
- [ ] `cbrt(x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Cube root of each element
- [ ] `ldexp(x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Returns x1 * 2**exp2
- [ ] `frexp(x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Decompose the elements of x into mantissa and twos exponent
- [ ] `modf(x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Return the fractional and integral parts of a floating-point number
- [ ] `spacing(x, out=None, dtype=None)` - Return the distance between x and the nearest adjacent floating-point value
- [ ] `nextafter(x1, x2, out=None, dtype=None)` - Return the next floating-point value after x1 in the direction of x2
- [ ] `float_power(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - First array elements raised to powers from second array, element-wise
- [ ] `true_divide(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Returns a true division of the inputs element-wise
- [ ] `floor_divide(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Returns the floor division of the inputs element-wise
- [ ] `reciprocal(x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Return the reciprocal of the argument element-wise

### I/O Functions

- [ ] `fromfile(file, dtype=float, count=-1, sep='', offset=0, *, like=None)` - Construct an array from data in a text or binary file
- [ ] `fromstring(string, dtype=float, count=-1, sep='', *, like=None)` - A new 1-D array initialized from text data in a string
- [ ] `loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=None, max_rows=None, encoding='bytes')` - Load data from a text file
- [ ] `genfromtxt()` - Generator to read data from a text file
- [ ] `save(file, arr, allow_pickle=True, fix_imports=True)` - Save an array to a binary file in NumPy .npy format
- [ ] `from_dlpack()` - Return an array from a binary file created by DLpack
- [ ] `fromregex()` - Construct an array by parsing a text file using regular expression

---

## Priority 2: Type System Improvements

### Top-level Dtype Constructors

- [ ] `int8(x, /, *, like=None)` - Return a signed 8-bit integer
- [ ] `int16(x, /, *, like=None)` - Return a signed 16-bit integer
- [ ] `int32(x, /, *, like=None)` - Return a signed 32-bit integer
- [ ] `int64(x, /, *, like=None)` - Return a signed 64-bit integer
- [ ] `uint8(x, /, *, like=None)` - Return an unsigned 8-bit integer
- [ ] `uint16(x, /, *, like=None)` - Return an unsigned 16-bit integer
- [ ] `uint32(x, /, *, like=None)` - Return an unsigned 32-bit integer
- [ ] `uint64(x, /, *, like=None)` - Return an unsigned 64-bit integer
- [ ] `float16(x, /, *, like=None)` - Return a half-precision floating-point number
- [ ] `float32(x, /, *, like=None)` - Return a single-precision floating-point number
- [ ] `float64(x, /, *, like=None)` - Return a double-precision floating-point number
- [ ] `complex64(real, imag, *, like=None)` - Return a complex number composed of 64-bit floats
- [ ] `complex128(real, imag, *, like=None)` - Return a complex number composed of 128-bit floats
- [ ] `bool_(x, *, like=None)` - Boolean type
- [ ] `object_()` - Object type
- [ ] `str_()` - String type
- [ ] `bytes_()` - Bytes type
- [ ] `void()` - Void type
- [ ] `datetime64()` - Datetime64 type
- [ ] `timedelta64()` - Timedelta64 type
- [ ] `character()` - Character type
- [ ] `complexfloating()` - Complex floating-point type
- [ ] `signedinteger()` - Signed integer type
- [ ] `unsignedinteger()` - Unsigned integer type
- [ ] `integer()` - Integer type
- [ ] `floating()` - Floating-point type
- [ ] `generic()` - Generic type
- [ ] `inexact()` - Inexact type
- [ ] `number()` - Number type
- [ ] `bool_()` - Boolean type (duplicate)
- [ ] `int_()` - Default integer type
- [ ] `intc()` - Integer type with same size as C int
- [ ] `intp()` - Integer type with same size as C intptr
- [ ] `uint()` - Unsigned integer type with same size as C int
- [ ] `uintc()` - Unsigned integer type with same size as C int
- [ ] `uintp()` - Unsigned integer type with same size as C intptr
- [ ] `long()` - Long integer type
- [ ] `longlong()` - Long long integer type
- [ ] `short()` - Short integer type
- [ ] `ushort()` - Unsigned short integer type
- [ ] `ulong()` - Unsigned long integer type
- [ ] `ulonglong()` - Unsigned long long integer type
- [ ] `csingle()` - Complex single-precision type
- [ ] `cdouble()` - Complex double-precision type
- [ ] `clongdouble()` - Complex long double-precision type

---

## Priority 3: Random Module Improvements (~70% coverage)

### Missing Random Functions

- [ ] `MT19937(seed=None)` - Mersenne Twister pseudorandom number generator
- [ ] `PCG64DXSM(seed=None)` - Parallelized PCG64 with DXSM (Small Carry) algorithm
- [ ] `Philox(seed=None)` - Philox 4x64 random number generator
- [ ] `SFC64(seed=None)` - Small Fast Chaotic PRNG
- [ ] `SeedSequence(entropy=None, spawn_key=None, *, child_seed=None)` - Seed sequence bit generator
- [ ] `bytes(length)` - Return random bytes
- [ ] `choice(a, size=None, replace=True, p=None, axis=None, shuffle=True)` - Generates a random sample from a given 1-D array
- [ ] `get_state()` - Return tuple representing the internal state of the generator
- [ ] `set_state(state)` - Set the internal state of the generator from a tuple
- [ ] `permutation(x)` - Randomly permute a sequence
- [ ] `shuffle(x)` - Modify a sequence in-place by shuffling its contents
- [ ] `rand(d0, d1, ..., dn)` - Random values in a given half-open interval
- [ ] `randn(d0, d1, ..., dn)` - Return a sample from the "standard normal" distribution
- [ ] `random_integers(low, high=None, size=None, dtype=int, endpoint=False)` - Return random integers from low (inclusive) to high (exclusive)
- [ ] `random_sample(size)` - Return random floats in the half-open interval [0.0, 1.0)
- [ ] `ranf(size)` - Return random floats in the half-open interval [0.0, 1.0) (alias)
- [ ] `sample(population, k, replace=False, p=None, axis=None, shuffle=True)` - Random sample without replacement
- [ ] `laplace(loc, scale, size=None)` - Draw samples from the Laplace or double exponential distribution
- [ ] `multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)` - Draw random samples from a multivariate normal distribution
- [ ] `noncentral_chisquare(df, nonc, size=None)` - Draw samples from a noncentral chi-squared distribution
- [ ] `noncentral_f(dfnum, dfden, size=None)` - Draw samples from a noncentral F distribution
- [ ] `standard_t(df, size=None)` - Draw samples from a standard Student's t distribution

---

## Priority 4: Masked Array (ma) Module (~30% coverage)

### Missing ma Functions

- [ ] `anom(a, axis=None, dtype=None)` - Compute the anomalies of an array
- [ ] `anomalies(a, axis=None)` - Compute the anomalies of an array (alias)
- [ ] `clump_masked(a, m=None)` - Find consecutive unmasked data in a masked array
- [ ] `clump_unmasked(a)` - Find consecutive unmasked data in an array (alias)
- [ ] `common_fill_value(a)` - Return the most common fill value for the array
- [ ] `compress_cols(a, axis=None)` - Compress 2-D arrays along columns
- [ ] `compress_nd(a, axis=None)` - Compress n-D arrays along given axis
- [ ] `compress_rowcols(a, axis=None)` - Compress 2-D arrays along rows and columns
- [ ] `compress_rows(a, axis=None)` - Compress 2-D arrays along rows
- [ ] `default_fill_value(a)` - Return the default fill value for the array
- [ ] `filled(a, fill_value=None)` - Return input as an array with masked data replaced by fill value
- [ ] `fix_invalid(a, fill_value=None)` - Return input with invalid data replaced by fill value
- [ ] `flatnotmasked_contiguous(a)` - Return a 1-D view of a flattened masked array
- [ ] `flatnotmasked_edges(a)` - Return a 1-D view of a flattened masked array with missing values at edges
- [ ] `flatten_mask(a)` - Return a copy of a, collapsed to one dimension
- [ ] `flatten_structured_array(a)` - Flatten a structured array
- [ ] `harden_mask(a, limit=None, inplace=False)` - Force unshared mask
- [ ] `ids(a)` - Return the indices of unmasked elements along given axis
- [ ] `in1d(a)` - Return a 1-D view of the input
- [ ] `isMA(x)` - Test if input is a masked array
- [ ] `isMaskedArray(x)` - Test if input is a masked array (alias)
- [ ] `is_mask(x)` - Test if input is a mask
- [ ] `is_masked(a)` - Test if input is masked (alias)
- [ ] `isarray(x)` - Test if input is an array
- [ ] `make_mask(condition, m=None, shrink=True, dtype=None)` - Create a boolean mask array
- [ ] `make_mask_descr(mdt)` - Create a dtype from a mask array description
- [ ] `make_mask_none(shape)` - Create a mask array filled with False
- [ ] `mask_cols(a, axis=None)` - Mask columns of a 2-D array
- [ ] `mask_or(m1, m2, copy=True, shrink=True)` - Combine masks with logical_or
- [ ] `mask_rowcols(a, axis=None)` - Mask rows and columns of a 2-D array
- [ ] `mask_rows(a, axis=None)` - Mask rows of a 2-D array
- [ ] `masked_all(a, axis=None, out=None, keepdims=False)` - Return True if all data in the masked array are valid
- [ ] `masked_all_like(a, axis=None, out=None, keepdims=False)` - Return True if all data in the masked array are valid (like version)
- [ ] `masked_array(condition, a)` - Create a masked array where condition is True
- [ ] `masked_equal(a, b)` - Return masked where a equals b
- [ ] `masked_greater(a, b)` - Return masked where a is greater than b
- [ ] `masked_greater_equal(a, b)` - Return masked where a is greater than or equal to b
- [ ] `masked_inside(a, v1, v2)` - Return masked where a is inside [v1, v2]
- [ ] `masked_invalid(a)` - Create a masked array where all values are invalid
- [ ] `masked_less(a, b)` - Return masked where a is less than b
- [ ] `masked_less_equal(a, b)` - Return masked where a is less than or equal to b
- [ ] `masked_not_equal(a, b)` - Return masked where a is not equal to b
- [ ] `masked_object(a)` - Create a masked array with object dtype
- [ ] `masked_outside(a, v1, v2)` - Return masked where a is outside [v1, v2]
- [ ] `masked_values(a)` - Return the data values of a masked array
- [ ] `masked_where(condition, a)` - Create a masked array where condition is True (like version)
- [ ] `maximum_fill_value(a)` - Return the fill value used for masked maximum
- [ ] `minimum_fill_value(a)` - Return the fill value used for masked minimum
- [ ] `mvoid(a)` - Return the masked array as a void type
- [ ] `ndenumerate(a)` - Multidimensional index iterator for masked arrays
- [ ] `notmasked_contiguous(a)` - Return a 1-D view of a masked array
- [ ] `notmasked_edges(a)` - Return a 1-D view of a masked array with missing values at edges
- [ ] `outerproduct(a, b)` - Compute the outer product of two arrays
- [ ] `polyfit(x, y, deg=None, rcond=None, full=False, w=None, cov=False)` - Least squares polynomial fit
- [ ] `product(a, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True)` - Return the product of array elements
- [ ] `set_fill_value(a, fill_value=None)` - Set the fill value for the masked array
- [ ] `setdiff1d(a, to_end=None, to_begin=None)` - Return the differences between consecutive elements of a masked array
- [ ] `setxor1d(a, n=None)` - Return exclusive or of a masked array over a given axis
- [ ] `soften_mask(a, harden=True)` - Soften a mask by setting adjacent elements to True
- [ ] `sometrue(a, axis=None, out=None)` - Return True if all data in the masked array are valid (alias)
- [ ] `swapaxes(a, axis1, axis2)` - Interchange two axes of a masked array
- [ ] `union1d(ar1, ar2)` - Return the unique, sorted union of 1-D arrays
- [ ] `vander(x, n=None)` - Generate a Vandermonde matrix for masked arrays
- [ ] `zeros_like(a, dtype=None, order=None, subok=True, shape=None)` - Return an array of zeros with the same shape and type as a given masked array

---

## Priority 5: Testing Module Improvements (~50% coverage)

### Missing Testing Functions

- [ ] `IgnoreException` - Exception raised when a condition is ignored
- [ ] `KnownFailureException` - Exception raised when a known failure occurs
- [ ] `SkipTest` - Exception raised when a test is skipped
- [ ] `TestCase` - Class context for tests
- [ ] `assert_no_gc_cycles(*args, **kwargs)` - Assert that no reference cycles are created
- [ ] `assert_no_warnings(*args, **kwargs)` - Assert that no warnings were raised
- [ ] `break_cycles()` - Break reference cycles
- [ ] `build_err_msg(*args, **kwargs)` - Build an error message from arguments
- [ ] `check_support_sve(*args, **kwargs)` - Check for SVE support
- [ ] `clear_and_catch_warnings()` - Context manager to catch warnings
- [ ] `decorate_methods(cls, casting='no', order='K', dtype=None, copy=None, subok=True, keepdims=False)` - Decorate methods
- [ ] `jiffies()` - Return number of jiffies since an epoch
- [ ] `measure(*args, **kwargs)` - Measure execution time
- [ ] `memusage(*args, **kwargs)` - Measure memory usage
- [ ] `print_assert_equal(*args, **kwargs)` - Print assertion messages
- [ ] `run_threaded(func, *, thread_count=1)` - Run a function in a separate thread
- [ ] `rundocs(*args, **kwargs)` - Run doctests
- [ ] `runstring(*args, **kwargs)` - Run string examples
- [ ] `tempdir(prefix=None)` - Return a temporary directory
- [ ] `tempprefix()` - Return the prefix used for temporary files

---

## Priority 6: Character (char) Module Improvements (~75% coverage)

### Missing char Functions

- [ ] `array(obj, itemsize=2, unicode=None, copy=True, order=None)` - Create a char array from an object
- [ ] `asarray(obj, itemsize=None, unicode=None, copy=None, order=None)` - Convert input to a char array
- [ ] `compare_chararrays(a1, a2, strip_whitespace=True, case_sensitive=True)` - Compare two character arrays element-wise
- [ ] `ljust(a, width, fillchar=' ')` - Return an array with the elements of a left-justified in a string of length width
- [ ] `rjust(a, width, fillchar=' ')` - Return an array with the elements of a right-justified in a string of length width
- [ ] `swapcase(a)` - Swap uppercase and lowercase characters
- [ ] `title(a)` - Convert element to titlecased version
- [ ] `partition(a, sep)` - Partition each element in a around sep
- [ ] `rpartition(a, sep)` - Partition (from the right) each element in a around sep
- [ ] `splitlines(a, keepends=False)` - Split string at line boundaries

---

## Priority 7: ndarray Methods Improvements (~80% coverage)

### Missing ndarray Methods

- [ ] `argpartition(a, kth, axis=-1, kind='introselect', order=None)` - Perform an indirect partition along the given axis
- [ ] `partition(a, kth, axis=-1, kind='introselect', order=None)` - Partition of an array
- [ ] `byteswap(inplace=False)` - Swap bytes in an array
- [ ] `choose(choices, out=None, mode='raise')` - Construct an array from an index array and a list of arrays
- [ ] `dump(file)` - Dump a pickle of the array to the specified file
- [ ] `dumps()` - Return the pickle of the array as a string
- [ ] `getfield(dtype, offset=0)` - Fetch a field from a record array
- [ ] `item(*args)` - Copy an element of an array to a standard Python scalar and return it
- [ ] `setfield(val, dtype, offset=0)` - Assign a value to a field-position, in a record array
- [ ] `setflags(write=None, align=None, uic=None)` - Set array flags writeably, and set alignment
- [ ] `to_device(device, stream=None, blocking=True)` - Return array on specified device
- [ ] `mT` - Return the matrix transpose

### Missing ndarray Properties

- [ ] `ctypes` - Return ctypes object
- [ ] `device` - Device memory object
- [ ] `mT` - Matrix transpose (duplicate of above)

---

## Priority 8: Export Pattern Standardization

### Resolve Export Conflicts

- [ ] Fix duplicate `promote_types` exports (currently exported from both `utils` and `type_promotion`)
- [ ] Clarify char function naming (some exported with `char_*` prefix, others without)
- [ ] Verify `set_ops::exports::*` exports and document which specific functions are available
- [ ] Verify `comparison_ufuncs::exports::*` exports and document which specific functions are available
- [ ] Verify `modules::ma::exports::*` exports and document which specific functions are available
- [ ] Verify `modules::testing::exports::*` exports and document which specific functions are available

### Standardize Function Signatures

- [ ] Make optional parameters consistent with NumPy (use `Option<T>` consistently)
- [ ] Ensure error messages match NumPy exactly
- [ ] Standardize parameter names (e.g., `step` vs `step_opt`, `endpoint` vs `endpoint_opt`)

---

## Priority 9: Polynomial Module Exports Verification

### Verify Polynomial Class Exports

- [ ] Ensure `Chebyshev` class is exported from `polynomial` module
- [ ] Ensure `Hermite` class is exported from `polynomial` module
- [ ] Ensure `HermiteE` class is exported from `polynomial` module
- [ ] Ensure `Laguerre` class is exported from `polynomial` module
- [ ] Ensure `Legendre` class is exported from `polynomial` module
- [ ] Verify all polynomial classes are accessible via `use numpy::polynomial::*`

---

## Priority 10: FFT Module Completion (~90% coverage)

### Missing FFT Functions

- [ ] `hfft(a, n=None, axis=-1, norm=None)` - Compute the one-dimensional discrete Fourier Transform for Hermitian-symmetric input
- [ ] `ihfft(a, n=None, axis=-1, norm=None)` - Compute the one-dimensional inverse discrete Fourier Transform for Hermitian-symmetric input

---

## Priority 11: Set Operations Module

### Missing Set Functions

- [ ] `in1d(ar1, ar2)` - Test if arrays are 1-D and return unique elements
- [ ] `unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)` - Find the unique elements of an array
- [ ] `unique_counts(ar)` - Return the counts of unique elements
- [ ] `unique_inverse(ar)` - Return the indices of the unique elements that reconstruct the input array
- [ ] `unique_values(ar)` - Return the unique elements (alias)
- [ ] `unique_all(ar)` - Return unique elements and their indices, inverse, and counts
- [ ] `setdiff1d(ar1, ar2, assume_unique=False)` - Find the set difference of 1-D arrays
- [ ] `setxor1d(ar1, ar2, assume_unique=False)` - Find the exclusive or of 1-D arrays

---

## Priority 12: Bitwise Operations Module

### Missing Bitwise Functions

- [ ] `bitwise_count(x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Count the number of 1-bits in the binary representation
- [ ] `bitwise_invert(x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True)` - Invert bits of an array element-wise

---

## Priority 13: Documentation and Examples

### Create Usage Examples

- [ ] Add examples for all array creation functions
- [ ] Add examples for all manipulation functions
- [ ] Add examples for all reduction functions
- [ ] Add examples for all linear algebra functions
- [ ] Add examples for all FFT functions
- [ ] Add examples for all random functions
- [ ] Add examples for all polynomial functions
- [ ] Add examples for all masked array functions
- [ ] Add examples for all character operations

### Create Type System Documentation

- [ ] Document the dtype system and how to use type traits
- [ ] Document the difference between `Int32Bit` and `int32()` constructors
- [ ] Document how to work with generic arrays `Array<T>`
- [ ] Document type promotion rules and behavior

---

## Implementation Notes

### Code Organization

1. **Modular Implementation:** Each function should be implemented in the most appropriate module
2. **Type Safety:** Use Rust's type system to ensure type safety
3. **Error Handling:** Return `Result<T, NumPyError>` for all functions that can fail
4. **Testing:** Write comprehensive tests for each new function
5. **Documentation:** Add doc comments following Rust documentation standards

### Testing Strategy

1. **Unit Tests:** Test individual functions with various inputs
2. **Integration Tests:** Test functions working together
3. **Edge Cases:** Test boundary conditions, empty arrays, NaN handling
4. **Performance Tests:** Benchmark critical functions against NumPy reference
5. **Parity Tests:** Compare output with NumPy 2.4 for correctness

### Verification Checklist

Before marking any item as complete:

- [ ] Function is implemented
- [ ] Function is exported from lib.rs
- [ ] Function has comprehensive documentation
- [ ] Function has unit tests
- [ ] Function has been tested against NumPy 2.4
- [ ] Performance is acceptable (within 2x of NumPy)

---

## Progress Tracking

### Module Completion Status

| Module | Target | Current | Progress |
|---------|--------|---------|----------|
| numpy (core) | 462 | ~390 | ~85% |
| numpy.linalg | 32 | 32 | 100% ✅ |
| numpy.fft | 18 | 16 | ~90% |
| numpy.random | 60 | 42 | ~70% |
| numpy.polynomial | 7 | 7 | 100% ✅ |
| numpy.ma | 219 | 65 | ~30% |
| numpy.char | 53 | 40 | ~75% |
| numpy.testing | 36 | 18 | ~50% |
| ndarray methods | 53 | 42 | ~80% |
| ndarray properties | 17 | 15 | ~90% |

### Overall Progress

**Total Items:** ~561  
**Completed:** ~348  
**Remaining:** ~213  
**Overall Completion:** ~62%

---

## Notes

1. This plan is a living document and should be updated as functions are implemented
2. Priority levels can be adjusted based on project needs
3. Some functions may be simpler to implement than others
4. Consider dependencies between functions (e.g., `empty` needed for `empty_like`)
5. Performance optimization should be considered after correctness is established

---

**End of Implementation Plan**
