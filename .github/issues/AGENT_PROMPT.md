# AGENT PROMPT: Fix Rust-NumPy Export Issues

## Your Mission

Fix export/import/duplicate issues in rust-numpy to match NumPy's public API.

## Repository Context

- **Location**: `rust-numpy/` directory (sibling to `.github/`)
- **Main Files**: 
  - `rust-numpy/src/lib.rs` - Main exports (coordinate changes)
  - `rust-numpy/src/modules/ma.rs` - Has duplicate exports (#599)
  - `rust-numpy/Cargo.toml` - Dependencies

## Where to Start

1. Read this prompt
2. Pick a directory based on priority:
   - `critical-fixes/` - P0 (start here)
   - `high-priority/` - P1
   - `parent-569/` through `parent-575/` - P2 (pick one)
   - `systemic-improvements/` - P2 architecture
3. Read the issue files in that directory
4. Implement fixes
5. Run verification

## Issue Overview (37 Total)

| Priority | Directory | Issues | Description |
|----------|-----------|--------|-------------|
| **P0** | `critical-fixes/` | 599-600 | Duplicate exports - FIX FIRST |
| **P1** | `high-priority/` | 601-603 | Structure fixes |
| **P2** | `parent-569/` | 569, 576-582 | lib-exports (8 issues) |
| **P2** | `parent-570/` | 570, 583-584 | submodules (3 issues) |
| **P2** | `parent-571/` | 571, 585-586 | dtype (3 issues) |
| **P2** | `parent-572/` | 572, 596-597 | typing (3 issues) |
| **P2** | `parent-573/` | 573, 587-589 | random (4 issues) |
| **P2** | `parent-574/` | 574, 590-591 | linalg (3 issues) |
| **P2** | `parent-575/` | 575, 592-595 | fft-io (5 issues) |
| **P2** | `systemic-improvements/` | 604-606 | Architecture |

## Critical Rules

### Code Patterns

**Exports - NEVER use wildcards:**
```rust
// ❌ WRONG
pub use crate::module::exports::*;

// ✅ CORRECT
pub use crate::module::{
    func1, func2, func3,
};
```

**Types - use aliases, NOT placeholder functions:**
```rust
// ❌ WRONG
pub fn int8() -> i8 { 0 }

// ✅ CORRECT
pub type int8 = i8;
```

**Internal modules - ALWAYS hide:**
```rust
// ❌ WRONG
pub mod kernel_ops;

// ✅ CORRECT
#[doc(hidden)]
pub mod _kernel_ops;
```

### Workflow Rules

1. **One directory at a time** - Complete all issues in a directory before moving on
2. **Verify after EVERY change** - Run the verification script
3. **Coordinate on shared files** - Check before modifying lib.rs or Cargo.toml
4. **No duplicates** - Always check for existing exports before adding

## Verification Script

Run this after every change:

```bash
#!/bin/bash
set -e
cd rust-numpy

echo "=== Duplicate Check ==="
DUPS=$(grep "pub use" src/lib.rs 2>/dev/null | sed 's/.*pub use //' | sort | uniq -d)
if [ -n "$DUPS" ]; then
    echo "❌ DUPLICATES: $DUPS"
    exit 1
fi
echo "✓ No duplicates"

echo ""
echo "=== Build ==="
cargo build --quiet
echo "✓ Build OK"

echo ""
echo "=== Tests ==="
cargo test --quiet
echo "✓ Tests OK"

echo ""
echo "✅ ALL CHECKS PASSED"
```

Or use the provided script: `bash verify.sh`

## Implementation Workflow

### Step 1: Pick Your Work

```bash
# List available directories
ls -la .github/issues/

# For first-time agents - start with P0
cat .github/issues/critical-fixes/issue-599.md
```

### Step 2: Read the Issue

Each issue file contains:
- Problem description
- Expected behavior
- Implementation notes
- Verification steps

### Step 3: Implement

Make the minimal changes required to fix the issue.

### Step 4: Verify

```bash
cd rust-numpy
cargo build
cargo test

# Check for duplicates
grep "pub use" src/lib.rs | sed 's/.*pub use //' | sort | uniq -d

# Should output nothing
```

### Step 5: Repeat

Move to the next issue in your chosen directory.

## Example: Fixing #599

```bash
# Read issue
cat .github/issues/critical-fixes/issue-599.md

# Problem: src/modules/ma.rs has duplicate exports
# Open file and remove duplicates:
# - all, all → keep one
# - any, any → keep one
# - append, append → keep one
# etc.

vim rust-numpy/src/modules/ma.rs

# Verify
cd rust-numpy
cargo build && cargo test

# Check specifically
grep "pub use" src/modules/ma.rs | \
  sed 's/.*{//; s/}.*//' | \
  tr ',' '\n' | \
  sort | \
  uniq -d
# Output should be empty
```

## Completion Criteria

✅ Issue requirements met (per issue file)
✅ `cargo build` passes
✅ `cargo test` passes
✅ No duplicate exports
✅ No wildcard exports (where applicable)
✅ Internal modules hidden

## Multi-Agent Coordination

If working with other agents:

1. **Claim a directory** - Each agent works in ONE `parent-XXX/` directory
2. **Don't touch shared files** - lib.rs and Cargo.toml require coordination
3. **Complete before switching** - Finish all issues in directory before claiming another

## Module Labels

| Directory | Label |
|-----------|-------|
| parent-569 | ralph/numpy:lib-exports |
| parent-570 | ralph/numpy:submodules |
| parent-571 | ralph/numpy:dtype |
| parent-572 | ralph/numpy:typing |
| parent-573 | ralph/numpy:random |
| parent-574 | ralph/numpy:linalg |
| parent-575 | ralph/numpy:fft-io |

## Quick Reference

| Command | Purpose |
|---------|---------|
| `cat AGENTS.md` | Full agent guide |
| `cat FULL_PROMPT.md` | Complete reference |
| `ls issue-*.md` | List issues (flat) |
| `cat critical-fixes/issue-599.md` | Read specific issue |
| `bash verify.sh` | Run verification |

## Success Criteria

The mission is complete when:
- All 37 issues are resolved
- `cargo build` passes with no warnings
- `cargo test` passes
- No duplicate exports exist
- No wildcard exports in lib.rs
- All internal modules are hidden
- NumPy API compatibility achieved

---

**Remember**: Match NumPy exactly. Verify after every change. Prefer explicit over implicit.
