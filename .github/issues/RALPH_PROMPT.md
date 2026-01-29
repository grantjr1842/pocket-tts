---
ralph_type: pocket-tts-numpy-exports
max_iterations: 50
completion_promise: <promise>ALL_EXPORT_ISSUES_RESOLVED</promise>
---

# Ralph Loop: Rust-NumPy Export Fix Loop

## Loop Initialization

You are running a Ralph Wiggum autonomous loop to fix 37 export/import/duplicate issues in rust-numpy. The loop continues until all issues are resolved or max_iterations is reached.

### Entry Point

```bash
# Location: pocket-tts/.github/issues/
# Target: rust-numpy/ subdirectory

# Read loop state
ls prd.json RULES.md progress.txat 2>/dev/null || echo "First run - initialize"
```

### Initialize on First Run

If `prd.json` does not exist:

1. **Create `RULES.md`**:
```markdown
# Tech Stack and Coding Conventions

## Tech Stack
- Language: Rust
- Framework: ndarray ecosystem
- Package Manager: Cargo
- Testing: cargo test

## Coding Conventions
- NEVER use wildcard exports: `pub use module::*;`
- ALWAYS use explicit exports: `pub use module::{a, b, c};`
- Use type aliases for NumPy types: `pub type int8 = i8;`
- Hide internal modules: `#[doc(hidden)] pub mod _internal;`
- Match NumPy API exactly - no more, no less
- Run verification after EVERY change
```

2. **Create `prd.json`**:
```json
{
  "project_meta": {
    "name": "rust-numpy-export-fix",
    "version": "1.0",
    "ralph_type": "pocket-tts-numpy-exports",
    "session_id": "numpy-exports-$(date +%s)"
  },
  "backlog": [
    {
      "group": "critical-fixes",
      "feature": "P0-duplicate-exports-ma",
      "description": "Remove duplicate exports in src/modules/ma.rs",
      "acceptance_criteria": [
        "Duplicate exports removed (all, any, append, array, copy, count, etc. listed twice) (grep + cargo build)",
        "cargo build passes with no errors",
        "cargo test passes"
      ],
      "passes": false
    },
    {
      "group": "critical-fixes",
      "feature": "P0-duplicate-stats-imports",
      "description": "Merge duplicate statistics imports in src/lib.rs",
      "acceptance_criteria": [
        "Statistics imports merged into single pub use statement",
        "No duplicate import lines remain",
        "cargo build passes"
      ],
      "passes": false
    },
    {
      "group": "high-priority",
      "feature": "P1-hide-internal-modules",
      "description": "Hide internal modules with #[doc(hidden)]",
      "acceptance_criteria": [
        "All kernel_* modules marked #[doc(hidden)]",
        "All simd_* modules marked #[doc(hidden)]",
        "All parallel_* modules marked #[doc(hidden)]",
        "Internal utilities hidden from public API"
      ],
      "passes": false
    },
    {
      "group": "high-priority",
      "feature": "P1-convert-wildcards",
      "description": "Convert wildcard exports to explicit",
      "acceptance_criteria": [
        "No pub use ...::*; patterns in lib.rs",
        "All exports explicitly listed",
        "Wildcard exceptions documented if necessary"
      ],
      "passes": false
    },
    {
      "group": "high-priority",
      "feature": "P1-reorganize-structure",
      "description": "Reorganize module structure (ma/testing to top-level)",
      "acceptance_criteria": [
        "ma module accessible at crate::ma",
        "testing module accessible at crate::testing",
        "Old paths still work (backward compatibility)"
      ],
      "passes": false
    },
    {
      "group": "lib-exports",
      "feature": "array-manipulation",
      "description": "Export missing array manipulation functions",
      "acceptance_criteria": [
        "concatenate, stack, vstack, hstack, dstack exported",
        "split, hsplit, vsplit, dsplit exported",
        "tile, repeat exported"
      ],
      "passes": false
    },
    {
      "group": "lib-exports",
      "feature": "math-ufuncs",
      "description": "Export basic mathematical ufuncs at root",
      "acceptance_criteria": [
        "sin, cos, tan, arcsin, arccos, arctan exported",
        "exp, log, log10, log2, sqrt exported",
        "add, subtract, multiply, divide exported"
      ],
      "passes": false
    },
    {
      "group": "dtype",
      "feature": "dtype-aliases",
      "description": "Export concrete dtype aliases",
      "acceptance_criteria": [
        "int8, int16, int32, int64 as type aliases",
        "uint8, uint16, uint32, uint64 as type aliases",
        "float32, float64 as type aliases",
        "complex64, complex128 as type aliases"
      ],
      "passes": false
    },
    {
      "group": "dtype",
      "feature": "dtype-traits",
      "description": "Export dtype abstract traits",
      "acceptance_criteria": [
        "NumPyDtype trait exported",
        "DTypeBound trait exported",
        "TypeDescriptor exported"
      ],
      "passes": false
    },
    {
      "group": "random",
      "feature": "bitgenerators",
      "description": "Export BitGenerators and SeedSequence",
      "acceptance_criteria": [
        "BitGenerator trait exported",
        "SeedSequence exported",
        "PCG64, MT19937 implementations exported"
      ],
      "passes": false
    },
    {
      "group": "random",
      "feature": "generator-methods",
      "description": "Export Generator methods",
      "acceptance_criteria": [
        "random, integers, random_raw exported",
        "shuffle, permutation exported",
        "choice exported"
      ],
      "passes": false
    },
    {
      "group": "linalg",
      "feature": "linalg-functions",
      "description": "Export missing linalg functions",
      "acceptance_criteria": [
        "norm, cond, det, slogdet exported",
        "matrix_power, matrix_rank exported",
        "LinAlgError exported"
      ],
      "passes": false
    },
    {
      "group": "fft",
      "feature": "fft-nd",
      "description": "Export fft N-D functions",
      "acceptance_criteria": [
        "fftn, ifftn, rfftn, irfftn exported",
        "fftshift, ifftshift exported",
        "fftfreq, rfftfreq exported"
      ],
      "passes": false
    },
    {
      "group": "typing",
      "feature": "typing-protocols",
      "description": "Export typing protocols",
      "acceptance_criteria": [
        "ArrayLike protocol exported",
        "DTypeLike protocol exported",
        "NDArray protocol exported"
      ],
      "passes": false
    },
    {
      "group": "typing",
      "feature": "ndarray-type-alias",
      "description": "Export NDArray type alias",
      "acceptance_criteria": [
        "NDArray<T> type alias exported",
        "Matches NumPy's ndarray type"
      ],
      "passes": false
    },
    {
      "group": "systemic",
      "feature": "export-registry",
      "description": "Create centralized export registry (src/exports.rs)",
      "acceptance_criteria": [
        "src/exports.rs created with all public exports",
        "lib.rs re-exports from exports.rs",
        "Registry is source of truth"
      ],
      "passes": false
    },
    {
      "group": "systemic",
      "feature": "automated-validation",
      "description": "Add automated export validation",
      "acceptance_criteria": [
        "Pre-commit hook blocks duplicate exports",
        "CI workflow validates export structure",
        "Script detects wildcards and duplicates"
      ],
      "passes": false
    }
  ]
}
```

3. **Create `progress.txt`**:
```
# Ralph Loop Progress Log
# Project: rust-numpy-export-fix

## [INIT] Loop Initialized

* **Note:** Created prd.json with 37 export issues
* **Status:** Ready to begin
* **Next:** Select first P0 task
```

## Loop Procedure

On each iteration, execute:

### 1. READ State

```bash
# Read rules
cat RULES.md

# Read current backlog
cat prd.json | jq '.backlog[] | select(.passes == false) | {group, feature, description}'

# Read progress
tail -20 progress.txt
```

### 2. SELECT Task

Pick the **highest priority** task where `passes: false`:

**Priority Order:**
1. Group: `critical-fixes` (P0)
2. Group: `high-priority` (P1)
3. Group: `lib-exports`, `dtype`, `random`, `linalg`, `fft`, `typing` (P2)
4. Group: `systemic` (P2)

### 3. EXECUTE Task

For the selected task:

**A. Read Issue File** (if exists):
```bash
# Map group to directory
GROUP="critical-fixes"  # or parent-569, etc.
cat ${GROUP}/issue-*.md 2>/dev/null || echo "No issue files"
```

**B. Identify Target Files**:
```bash
# Based on task group:
# - critical-fixes: rust-numpy/src/modules/ma.rs, rust-numpy/src/lib.rs
# - lib-exports: rust-numpy/src/lib.rs
# - dtype: rust-numpy/src/dtype.rs
# - random: rust-numpy/src/random/
# - linalg: rust-numpy/src/linalg/
```

**C. Implement Fix**:
- Make minimal changes
- Follow RULES.md conventions
- Never use wildcards
- Use explicit exports

### 4. VERIFY Task

```bash
cd rust-numpy

# Check 1: No duplicates
echo "=== Duplicate Check ==="
DUPS=$(grep "pub use" src/lib.rs | sed 's/.*pub use //' | sort | uniq -d)
if [ -n "$DUPS" ]; then
    echo "FAIL: Duplicates found: $DUPS"
    exit 1
fi
echo "PASS: No duplicates"

# Check 2: Build
echo "=== Build ==="
cargo build --quiet || exit 1
echo "PASS: Build OK"

# Check 3: Tests
echo "=== Tests ==="
cargo test --quiet || exit 1
echo "PASS: Tests OK"

# Check 4: No wildcards (for lib-exports tasks)
echo "=== Wildcard Check ==="
WILDCARDS=$(grep "pub use.*\*" src/lib.rs | grep -v "pub use num_traits" || true)
if [ -n "$WILDCARDS" ]; then
    echo "WARN: Wildcards found: $WILDCARDS"
fi
```

### 5. RECORD Progress

If verification passes:

**A. Update `prd.json`**:
```bash
# Mark task as passed
jq '(.backlog[] | select(.feature == "TASK_FEATURE_NAME")).passes = true' prd.json > prd.json.tmp
mv prd.json.tmp prd.json
```

**B. Append to `progress.txt`**:
```
## [ITERATION N] TASK_FEATURE_NAME

* **Note:** What was changed
* **Files Modified:** file1.rs, file2.rs
* **Verification:** All checks passed
* **Status:** COMPLETE
```

### 6. COMMIT

```bash
git add -A
git commit -m "Ralph: TASK_FEATURE_NAME - brief description"
```

### 7. REPEAT

Continue until:
- All tasks have `"passes": true`
- OR `max_iterations` reached

## Self-Correction Patterns

### If Build Fails

1. Read error message
2. Fix the specific error
3. Re-run verification
4. Do NOT mark task complete until build passes

### If Tests Fail

1. Run tests with output: `cargo test`
2. Identify failing test
3. Fix or update test
4. Re-run verification

### If Duplicates Found

```bash
# Find duplicates
grep "pub use" src/lib.rs | sed 's/.*pub use //' | sort | uniq -d

# Fix: Remove duplicate lines
# Re-verify
```

## Loop Exit

**Completion Condition**: All 37 tasks have `"passes": true`

**Output**:
```
<promise>ALL_EXPORT_ISSUES_RESOLVED</promise>

Summary:
- Tasks completed: 37/37
- Final build: PASS
- Final tests: PASS
- Duplicates: 0
- Wildcards: 0
```

**If max_iterations reached without completion**:
- Report remaining tasks
- Suggest splitting large tasks
- Output partial completion status

## Context Isolation

Each iteration runs in a fresh context. Only these files persist:
- `prd.json` - Backlog state
- `RULES.md` - Tech stack rules
- `progress.txt` - Human-readable log
- `activity.md` - Detailed activity log (optional)
- Git history - Committed changes

## Task Size Guidelines

**Good task** (completes in one iteration):
- "Remove 5 duplicate exports from ma.rs"
- "Export 3 missing math functions"

**Too big** (split into smaller tasks):
- "Fix all lib.rs exports" → Split into array-manipulation, math-ufuncs, etc.
