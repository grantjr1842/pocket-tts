# [Sub-Issue] Create numpy.char Submodule Structure (#570)

**Issue #583**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/583  
**Created:** 2026-01-29T03:04:04Z  
**Updated:** 2026-01-29T03:23:53Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:submodules, ralph/numpy:impl  

---

## Parent Issue
Part of #570 - Submodule structure doesn't match NumPy hierarchy

## Scope
Create `numpy.char` Submodule Structure

## Background
NumPy has `numpy.char` module for vectorized string operations. Current rust-numpy has `src/char.rs` but it should be a proper submodule `src/char/mod.rs`.

## Tasks

### 1. Module Structure
- [ ] Create `src/char/mod.rs`
- [ ] Create `src/char/core.rs` - Core string operations
- [ ] Create `src/char/exports.rs` - Public exports
- [ ] Update `src/lib.rs` to use `pub mod char;` instead of `pub mod char;` from file

### 2. Character Array Types (if not exists)
- [ ] `chararray` class equivalent
- [ ] Support for string, unicode, and byte arrays

### 3. String Operations to Expose
- [ ] `add(x1, x2)` - String concatenation
- [ ] `multiply(a, i)` - String repetition
- [ ] `capitalize(a)` - Capitalize first character
- [ ] `center(a, width, fillchar=' ')` - Center string
- [ ] `count(a, sub, start=0, end=None)` - Count substring occurrences
- [ ] `decode(a, encoding=None, errors=None)` - Decode bytes
- [ ] `encode(a, encoding=None, errors=None)` - Encode to bytes
- [ ] `endswith(a, suffix, start=0, end=None)` - Check suffix
- [ ] `expandtabs(a, tabsize=8)` - Replace tabs with spaces
- [ ] `find(a, sub, start=0, end=None)` - Find substring
- [ ] `index(a, sub, start=0, end=None)` - Find substring (raises if not found)
- [ ] `isalnum(a)` - Check alphanumeric
- [ ] `isalpha(a)` - Check alphabetic
- [ ] `isdecimal(a)` - Check decimal
- [ ] `isdigit(a)` - Check digit
- [ ] `islower(a)` - Check lowercase
- [ ] `isnumeric(a)` - Check numeric
- [ ] `isspace(a)` - Check whitespace
- [ ] `istitle(a)` - Check titlecase
- [ ] `isupper(a)` - Check uppercase
- [ ] `join(sep, seq)` - Join strings
- [ ] `ljust(a, width, fillchar=' ')` - Left justify
- [ ] `lower(a)` - Convert to lowercase
- [ ] `lstrip(a, chars=None)` - Left strip
- [ ] `mod(a, values)` - String formatting
- [ ] `partition(a, sep)` - Partition string
- [ ] `replace(a, old, new, count=None)` - Replace substring
- [ ] `rfind(a, sub, start=0, end=None)` - Find from right
- [ ] `rindex(a, sub, start=0, end=None)` - Find from right (raises)
- [ ] `rjust(a, width, fillchar=' ')` - Right justify
- [ ] `rpartition(a, sep)` - Partition from right
- [ ] `rsplit(a, sep=None, maxsplit=None)` - Split from right
- [ ] `rstrip(a, chars=None)` - Right strip
- [ ] `split(a, sep=None, maxsplit=None)` - Split string
- [ ] `splitlines(a, keepends=None)` - Split at line breaks
- [ ] `startswith(a, prefix, start=0, end=None)` - Check prefix
- [ ] `strip(a, chars=None)` - Strip whitespace
- [ ] `swapcase(a)` - Swap case
- [ ] `title(a)` - Title case
- [ ] `translate(a, table, deletechars=None)` - Translate characters
- [ ] `upper(a)` - Convert to uppercase
- [ ] `zfill(a, width)` - Pad with zeros
- [ ] `str_len(a)` - String length

## Implementation Notes
- Vectorized operations on arrays of strings
- Support Unicode and byte strings
- Handle encoding/decoding properly

## Files to Modify/Create
- NEW: `src/char/mod.rs`
- NEW: `src/char/core.rs`
- NEW: `src/char/exports.rs`
- MODIFY: `src/lib.rs` - Update char module declaration
- NEW/EXISTING: `src/char.rs` content moves to module

## Acceptance Criteria
- [ ] `char` submodule exists with proper structure
- [ ] All string operations available as `numpy::char::*`
- [ ] Vectorized operations work on string arrays
- [ ] Tests for all functions

## Conflict Avoidance
⚠️ **COORDINATION:**
- Coordinate with #550 (Complete String Operations) - this is STRUCTURAL work
- #550 is IMPLEMENTATION work
- This issue creates the module structure; #550 implements the functions

