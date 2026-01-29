---
name: Export Changes
about: Changes to public API exports
title: "export: "
labels: ["ralph/numpy:lib-exports"]
---

## Export Changes Checklist

- [ ] I have checked for duplicate exports
- [ ] All exports are necessary for the public API
- [ ] Internal items are not exposed
- [ ] Documentation is updated (if needed)

## Files Modified

<!-- List files that were modified -->
- `rust-numpy/src/lib.rs` (exports)
- 

## New Exports

<!-- List new items being exported -->
```rust
pub use module::{
    // new items
};
```

## Verification

- [ ] `cargo build` passes
- [ ] `cargo doc` generates correctly
- [ ] No duplicate export warnings

## Related Issues

Resolves #
