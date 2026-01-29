# FULL AUTONOMOUS AGENT PROMPT

## MISSION

Implement 100% NumPy-compatible export structure for rust-numpy by resolving all 37 documented issues.

## ALL 37 ISSUES

| # | Title | Priority | Directory |
|---|-------|----------|-----------|
| 569 | Fix top-level lib.rs exports - array manipulation | P2 | parent-569/ |
| 570 | Create missing submodules (char, emath, exceptions) | P2 | parent-570/ |
| 571 | Fix dtype type system exports | P2 | parent-571/ |
| 572 | Complete typing module exports | P2 | parent-572/ |
| 573 | Fix random module exports | P2 | parent-573/ |
| 574 | Fix linalg module exports | P2 | parent-574/ |
| 575 | Fix fft and io module exports | P2 | parent-575/ |
| 576 | Export missing array manipulation functions | P2 | parent-569/ |
| 577 | Export basic mathematical ufuncs at root | P2 | parent-569/ |
| 578 | Export array creation shortcuts | P2 | parent-569/ |
| 579 | Export logical operations | P2 | parent-569/ |
| 580 | Export basic statistics functions | P2 | parent-569/ |
| 581 | Export shape manipulation functions | P2 | parent-569/ |
| 582 | Export array set operations | P2 | parent-569/ |
| 583 | Create char submodule | P2 | parent-570/ |
| 584 | Create emath and exceptions submodules | P2 | parent-570/ |
| 585 | Export dtype abstract traits | P2 | parent-571/ |
| 586 | Export concrete dtype aliases | P2 | parent-571/ |
| 587 | Export BitGenerators and SeedSequence | P2 | parent-573/ |
| 588 | Export Generator methods | P2 | parent-573/ |
| 589 | Export random convenience functions | P2 | parent-573/ |
| 590 | Export missing linalg functions | P2 | parent-574/ |
| 591 | Move linalg to submodule directory | P2 | parent-574/ |
| 592 | Export fft N-D functions | P2 | parent-575/ |
| 593 | Export NPZ file I/O | P2 | parent-575/ |
| 594 | Export text I/O helpers | P2 | parent-575/ |
| 595 | Export memory mapping utilities | P2 | parent-575/ |
| 596 | Export typing protocols | P2 | parent-572/ |
| 597 | Export NDArray type alias | P2 | parent-572/ |
| 599 | Remove duplicate exports in ma.rs | **P0** | critical-fixes/ |
| 600 | Merge duplicate statistics imports | **P0** | critical-fixes/ |
| 601 | Hide internal modules | P1 | high-priority/ |
| 602 | Convert wildcards to explicit | P1 | high-priority/ |
| 603 | Reorganize module structure | P1 | high-priority/ |
| 604 | Create export registry | P2 | systemic-improvements/ |
| 605 | Add automated validation | P2 | systemic-improvements/ |
| 606 | Implement file locking | P2 | systemic-improvements/ |

## IMPLEMENTATION RULES

### Never Use Wildcards
```rust
// ❌ WRONG
pub use crate::module::exports::*;

// ✅ CORRECT
pub use crate::module::{
    func1, func2, func3,
};
```

### Hide Internal Modules
```rust
// ❌ WRONG
pub mod kernel_ops;

// ✅ CORRECT
#[doc(hidden)]
pub mod _kernel_ops;
```

### Use Type Aliases, Not Functions
```rust
// ❌ WRONG (placeholder functions)
pub fn int8() -> i8 { 0 }
pub fn float32() -> f32 { 0.0 }

// ✅ CORRECT (type aliases)
pub type int8 = i8;
pub type float32 = f32;
```

## VERIFICATION SCRIPT

```bash
#!/bin/bash
set -e

cd rust-numpy

echo "=== Duplicate Check ==="
DUPS=$(grep "pub use" src/lib.rs | sed 's/.*pub use //' | sort | uniq -d)
if [ -n "$DUPS" ]; then
    echo "❌ DUPLICATES:"
    echo "$DUPS"
    exit 1
fi
echo "✓ No duplicates"

echo "=== Build ==="
cargo build --quiet
echo "✓ Build OK"

echo "=== Tests ==="
cargo test --quiet
echo "✓ Tests OK"

echo "=== Clippy ==="
cargo clippy --quiet 2>&1 | grep -v "warning:" | head -5
echo "✓ Clippy OK"

echo ""
echo "🎉 ALL CHECKS PASSED"
```

## ORDER OF IMPLEMENTATION

1. **Start with P0** (critical-fixes/) - Fixes P0 before anything else
2. **Then P1** (high-priority/) - Structure and hiding
3. **Then pick ONE P2 group** by module - Complete all issues in that group
4. **Finally P2 systemic** - Architecture improvements

## CONFLICT AVOIDANCE

- Each agent: Pick ONE parent-XXX/ directory
- Never work on multiple module groups simultaneously
- Coordinate before touching lib.rs or Cargo.toml
- Document every change clearly

## COMPLETION CRITERIA

✅ All 37 issues resolved
✅ `cargo build` passes
✅ `cargo test` passes
✅ No duplicate exports
✅ No wildcard exports (except legitimate patterns like `pub use num_traits::*;` for re-exports)
✅ Internal modules hidden
✅ Type aliases correct (not placeholder functions)

