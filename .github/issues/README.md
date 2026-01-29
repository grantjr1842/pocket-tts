# Rust-NumPy Export Issues

All 37 export/import/duplicate issues for the rust-numpy crate.

## Quick Start

```bash
# For autonomous agents - start here
cat AGENTS.md

# Full detailed prompt  
cat FULL_PROMPT.md

# Run verification
bash verify.sh
```

## Directory Structure

```
.github/issues/
├── AGENTS.md              # Start here for agent guidance
├── FULL_PROMPT.md         # Complete reference
├── README.md              # This file
├── verify.sh              # Verification script
├── critical-fixes/        # P0 - Critical issues
├── high-priority/         # P1 - High priority issues
├── systemic-improvements/ # P2 - Architecture & automation
└── parent-*/              # P2 - Issues grouped by parent
```

## Issue Inventory (37 Total)

### P0 - Critical (Fix First)
**Directory: `critical-fixes/`**

| Issue | Title |
|-------|-------|
| 599 | Remove duplicate exports in ma.rs |
| 600 | Merge duplicate statistics imports |

### P1 - High Priority  
**Directory: `high-priority/`**

| Issue | Title |
|-------|-------|
| 601 | Hide internal modules with #[doc(hidden)] |
| 602 | Convert wildcard exports to explicit |
| 603 | Reorganize module structure |

### P2 - By Module (Grouped by Parent)

**`parent-569/` - lib-exports (8 issues)**
- 569: Fix top-level lib.rs exports - array manipulation
- 576: Export missing array manipulation functions
- 577: Export basic mathematical ufuncs at root
- 578: Export array creation shortcuts
- 579: Export logical operations
- 580: Export basic statistics functions
- 581: Export shape manipulation functions
- 582: Export array set operations

**`parent-570/` - submodules (3 issues)**
- 570: Create missing submodules (char, emath, exceptions)
- 583: Create char submodule
- 584: Create emath and exceptions submodules

**`parent-571/` - dtype (3 issues)**
- 571: Fix dtype type system exports
- 585: Export dtype abstract traits
- 586: Export concrete dtype aliases

**`parent-572/` - typing (3 issues)**
- 572: Complete typing module exports
- 596: Export typing protocols
- 597: Export NDArray type alias

**`parent-573/` - random (4 issues)**
- 573: Fix random module exports
- 587: Export BitGenerators and SeedSequence
- 588: Export Generator methods
- 589: Export random convenience functions

**`parent-574/` - linalg (3 issues)**
- 574: Fix linalg module exports
- 590: Export missing linalg functions
- 591: Move linalg to submodule directory

**`parent-575/` - fft-io (5 issues)**
- 575: Fix fft and io module exports
- 592: Export fft N-D functions
- 593: Export NPZ file I/O
- 594: Export text I/O helpers
- 595: Export memory mapping utilities

### P2 - Systemic Improvements
**Directory: `systemic-improvements/`**

| Issue | Title |
|-------|-------|
| 604 | Create centralized export registry |
| 605 | Add automated export validation |
| 606 | Implement CODEOWNERS and file locking |

## Implementation Order

1. **P0 First**: Start with `critical-fixes/` - fixes blocking other work
2. **P1 Second**: Then `high-priority/` - structure improvements
3. **P2 By Module**: Pick ONE `parent-XXX/` directory and complete all issues there
4. **P2 Systemic**: Finally `systemic-improvements/` - automation

## Module Labels (for Agent Coordination)

When multiple agents work in parallel:

| Directory | Module Label | Description |
|-----------|--------------|-------------|
| parent-569 | ralph/numpy:lib-exports | Top-level lib.rs |
| parent-570 | ralph/numpy:submodules | New submodules |
| parent-571 | ralph/numpy:dtype | Type system |
| parent-572 | ralph/numpy:typing | Typing module |
| parent-573 | ralph/numpy:random | Random module |
| parent-574 | ralph/numpy:linalg | Linear algebra |
| parent-575 | ralph/numpy:fft-io | FFT and I/O |

Each agent should claim ONE directory and complete all issues within it.
