# AGENTS.md - Autonomous Agent Guide for Rust-NumPy Export Issues

## Quick Start for Agents

```bash
# 1. Read this file
cat AGENTS.md

# 2. See available issue directories
ls -la

# 3. Pick an issue by priority
cat critical-fixes/issue-599.md  # Start with P0

# 4. Implement the fix

# 5. Run verification
bash verify.sh
```

## Issue Inventory (37 Total)

### P0 - Critical (Fix First)
**Directory: `critical-fixes/`**

| Issue | Title |
|-------|-------|
| [599](critical-fixes/issue-599.md) | Remove duplicate exports in ma.rs |
| [600](critical-fixes/issue-600.md) | Merge duplicate statistics imports |

### P1 - High Priority  
**Directory: `high-priority/`**

| Issue | Title |
|-------|-------|
| [601](high-priority/issue-601.md) | Hide internal modules with #[doc(hidden)] |
| [602](high-priority/issue-602.md) | Convert wildcard exports to explicit |
| [603](high-priority/issue-603.md) | Reorganize module structure |

### P2 - By Module (Pick ONE Directory)

**`parent-569/` - lib-exports (8 issues)**
- [569](parent-569/issue-569.md) - Parent: Fix top-level lib.rs exports
- [576](parent-569/issue-576.md) - Export missing array manipulation
- [577](parent-569/issue-577.md) - Export basic mathematical ufuncs
- [578](parent-569/issue-578.md) - Export array creation shortcuts
- [579](parent-569/issue-579.md) - Export logical operations
- [580](parent-569/issue-580.md) - Export basic statistics
- [581](parent-569/issue-581.md) - Export shape manipulation
- [582](parent-569/issue-582.md) - Export array set operations

**`parent-570/` - submodules (3 issues)**
- [570](parent-570/issue-570.md) - Parent: Create missing submodules
- [583](parent-570/issue-583.md) - Create char submodule
- [584](parent-570/issue-584.md) - Create emath and exceptions

**`parent-571/` - dtype (3 issues)**
- [571](parent-571/issue-571.md) - Parent: Fix dtype type system
- [585](parent-571/issue-585.md) - Export dtype abstract traits
- [586](parent-571/issue-586.md) - Export concrete dtype aliases

**`parent-572/` - typing (3 issues)**
- [572](parent-572/issue-572.md) - Parent: Complete typing module
- [596](parent-572/issue-596.md) - Export typing protocols
- [597](parent-572/issue-597.md) - Export NDArray type alias

**`parent-573/` - random (4 issues)**
- [573](parent-573/issue-573.md) - Parent: Fix random module
- [587](parent-573/issue-587.md) - Export BitGenerators and SeedSequence
- [588](parent-573/issue-588.md) - Export Generator methods
- [589](parent-573/issue-589.md) - Export random convenience functions

**`parent-574/` - linalg (3 issues)**
- [574](parent-574/issue-574.md) - Parent: Fix linalg exports
- [590](parent-574/issue-590.md) - Export missing linalg functions
- [591](parent-574/issue-591.md) - Move linalg to submodule directory

**`parent-575/` - fft-io (5 issues)**
- [575](parent-575/issue-575.md) - Parent: Fix fft and io modules
- [592](parent-575/issue-592.md) - Export fft N-D functions
- [593](parent-575/issue-593.md) - Export NPZ file I/O
- [594](parent-575/issue-594.md) - Export text I/O helpers
- [595](parent-575/issue-595.md) - Export memory mapping utilities

### P2 - Systemic Improvements
**Directory: `systemic-improvements/`**

| Issue | Title |
|-------|-------|
| [604](systemic-improvements/issue-604.md) | Create centralized export registry |
| [605](systemic-improvements/issue-605.md) | Add automated export validation |
| [606](systemic-improvements/issue-606.md) | Implement CODEOWNERS and file locking |

## Implementation Order

```
Phase 1: Critical (P0)
  └── critical-fixes/
      ├── issue-599.md  (duplicate exports in ma.rs)
      └── issue-600.md  (duplicate statistics imports)

Phase 2: High Priority (P1)
  └── high-priority/
      ├── issue-601.md  (hide internal modules)
      ├── issue-602.md  (convert wildcards)
      └── issue-603.md  (reorganize structure)

Phase 3: By Module (P2) - Pick ONE directory
  ├── parent-569/  (lib-exports: 8 issues)
  ├── parent-570/  (submodules: 3 issues)
  ├── parent-571/  (dtype: 3 issues)
  ├── parent-572/  (typing: 3 issues)
  ├── parent-573/  (random: 4 issues)
  ├── parent-574/  (linalg: 3 issues)
  └── parent-575/  (fft-io: 5 issues)

Phase 4: Systemic (P2)
  └── systemic-improvements/
      ├── issue-604.md  (export registry)
      ├── issue-605.md  (automated validation)
      └── issue-606.md  (file locking)
```

## Critical Rules

### NEVER Do These
1. ❌ Use wildcard exports: `pub use module::*;`
2. ❌ Export placeholder functions for types (use `pub type`)
3. ❌ Leave internal modules exposed
4. ❌ Create duplicate exports
5. ❌ Modify `rust-numpy/src/lib.rs` or `Cargo.toml` without checking first

### ALWAYS Do These
1. ✅ Use explicit exports: `pub use module::{a, b, c};`
2. ✅ Use type aliases: `pub type int8 = i8;`
3. ✅ Hide internals: `#[doc(hidden)] pub mod _internal;`
4. ✅ Run verification after EVERY change
5. ✅ Work on ONE issue at a time

## Verification

The `verify.sh` script checks:
- No duplicate exports
- No wildcard exports
- Build passes (`cargo build`)
- Tests pass (`cargo test`)

Run it after every change:
```bash
bash verify.sh
```

## Shared Files Protocol

These files require coordination - check before modifying:
- `rust-numpy/src/lib.rs`
- `rust-numpy/Cargo.toml`

If multiple agents:
1. Each agent claims ONE `parent-XXX/` directory
2. Never modify lib.rs/Cargo.toml simultaneously
3. Make minimal, documented changes

## File Locations

```
pocket-tts/
├── rust-numpy/
│   ├── src/
│   │   ├── lib.rs              # Main exports (coordinate!)
│   │   ├── modules/
│   │   │   ├── ma.rs           # Has duplicates (#599)
│   │   │   └── testing.rs
│   │   ├── linalg/
│   │   ├── random/
│   │   └── ...
│   └── Cargo.toml              # Shared (coordinate!)
│
└── .github/issues/             # This directory
    ├── AGENTS.md               # This file
    ├── README.md               # Issue index
    ├── FULL_PROMPT.md          # Complete reference
    ├── verify.sh               # Verification script
    ├── critical-fixes/         # P0 issues
    ├── high-priority/          # P1 issues
    ├── systemic-improvements/  # P2 architecture
    └── parent-*/               # P2 by module
```

## Example: Fixing Issue #599

```bash
# 1. Read the issue
cat critical-fixes/issue-599.md

# 2. Edit the file with duplicates
vim ../rust-numpy/src/modules/ma.rs
# Remove duplicate exports (all, any, append, array, copy, count, etc.)

# 3. Verify
cd ../rust-numpy
cargo build
cargo test

# 4. Check specifically for duplicates
grep "pub use" src/modules/ma.rs | sed 's/.*{//; s/}.*//' | tr ',' '\n' | sort | uniq -d
# Should return nothing

# 5. Run full verification
bash verify.sh
```

## Completion Criteria

An issue is DONE when:
- [ ] Issue requirements implemented (per issue file)
- [ ] `cargo build` passes with no errors
- [ ] `cargo test` passes
- [ ] No duplicates (run verify.sh)
- [ ] No wildcards (if applicable)
- [ ] Changes documented

## Module Labels (for Multiple Agents)

| Directory | Module Label |
|-----------|--------------|
| parent-569 | ralph/numpy:lib-exports |
| parent-570 | ralph/numpy:submodules |
| parent-571 | ralph/numpy:dtype |
| parent-572 | ralph/numpy:typing |
| parent-573 | ralph/numpy:random |
| parent-574 | ralph/numpy:linalg |
| parent-575 | ralph/numpy:fft-io |

## Need Help?

- Issue index: `cat README.md`
- Full details: `cat FULL_PROMPT.md`
- Verify: `bash verify.sh`

---

**Goal**: 100% NumPy API compatibility. Match NumPy exactly - no more, no less.
