# [CRITICAL] Remove duplicate exports in src/modules/ma.rs

**Issue #599**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/599  
**Created:** 2026-01-29T03:34:22Z  
**Updated:** 2026-01-29T03:34:22Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:submodules  

---

## Summary
The `src/modules/ma.rs` file contains **numerous duplicate exports** in its exports block. This appears to be a copy-paste error from multiple agents working on the same file.

## Location
File: `src/modules/ma.rs`
Around line 1210 in the `exports` module

## Duplicates Found

```rust
pub mod exports {
    pub use super::{
        all, all,                    // ← DUPLICATED
        allequal, allequal,          // ← DUPLICATED  
        any, any,                    // ← DUPLICATED
        append, append,              // ← DUPLICATED
        array, array,                // ← DUPLICATED
        average, average,            // ← DUPLICATED
        concatenate, concatenate,    // ← DUPLICATED
        copy, copy,                  // ← DUPLICATED
        count, count,                // ← DUPLICATED
        empty, empty,                // ← DUPLICATED
        filled, filled,              // ← DUPLICATED
        full, full,                  // ← DUPLICATED
        full_like, full_like,        // ← DUPLICATED
        getdata, getdata,            // ← DUPLICATED
        is_masked_array, is_masked_array,  // ← DUPLICATED
    };
}
```

## Impact
- **Compilation:** May cause warnings or errors with strict compiler settings
- **Documentation:** Duplicate entries in generated docs
- **Confusion:** Developers may think these are different items

## Fix
Remove duplicate entries from the exports block.

## Acceptance Criteria
- [ ] All duplicates removed from exports block
- [ ] `cargo build` compiles without warnings
- [ ] No duplicate entries in `cargo doc` output

