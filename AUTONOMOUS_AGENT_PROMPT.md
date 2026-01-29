# AUTONOMOUS AGENT: Resolve Rust-NumPy Export Issues

## YOUR MISSION

Fix all 37 export/import/duplicate issues in rust-numpy to achieve 100% NumPy API compatibility.

## QUICK START

```bash
# 1. Read the issue index
cat .github/issues/README.md

# 2. Pick ONE directory to work on:
#    - critical-fixes/       (P0 - do first)
#    - high-priority/        (P1 - do second)
#    - parent-569/           (lib-exports)
#    - parent-570/           (submodules)
#    - parent-571/           (dtype)
#    - parent-572/           (typing)
#    - parent-573/           (random)
#    - parent-574/           (linalg)
#    - parent-575/           (fft-io)
#    - systemic-improvements/ (P2 - architecture)

# 3. Read issue files in that directory
ls .github/issues/YOUR-DIRECTORY/
cat .github/issues/YOUR-DIRECTORY/issue-XXX.md

# 4. Implement fix
# 5. Verify (see checklist below)
# 6. Mark done
```

## CRITICAL RULES

### ⚠️ BEFORE touching rust-numpy/src/lib.rs or Cargo.toml:
1. Check if another agent is modifying it
2. Document your changes clearly
3. Make minimal changes

### ✅ ALWAYS:
1. **Be explicit** - Never use `pub use module::*;`
2. **Hide internals** - Use `#[doc(hidden)]` for non-NumPy modules
3. **Check duplicates** - Run: `grep "pub use" src/lib.rs | sort | uniq -d`
4. **Check wildcards** - Run: `grep "pub use.*\*" src/lib.rs`

### ❌ NEVER:
1. Use wildcard exports (`*`)
2. Export placeholder functions for types (use `pub type` instead)
3. Leave internal modules exposed
4. Create duplicate exports

## VERIFICATION CHECKLIST

Run after EVERY change:

```bash
cd rust-numpy

# Check 1: No duplicates
echo "=== Checking for Duplicates ==="
DUPS=$(grep "pub use" src/lib.rs | sed 's/.*pub use //' | sort | uniq -d)
if [ -n "$DUPS" ]; then
    echo "❌ DUPLICATES: $DUPS"
    exit 1
fi
echo "✓ No duplicates"

# Check 2: Build passes
echo "=== Build ==="
cargo build --quiet || { echo "❌ BUILD FAILED"; exit 1; }
echo "✓ Build OK"

# Check 3: Tests pass
echo "=== Tests ==="
cargo test --quiet || { echo "❌ TESTS FAILED"; exit 1; }
echo "✓ Tests OK"

# Check 4: No wildcards
echo "=== Wildcards ==="
grep -n "pub use.*\*" src/lib.rs && echo "⚠️  WILDCARDS FOUND" || echo "✓ No wildcards"

echo ""
echo "✅ ALL CHECKS PASSED"
```

## ISSUE PRIORITIES

| Priority | Issues | Directory |
|----------|--------|-----------|
| **P0 - CRITICAL** | #599, #600 | `critical-fixes/` |
| **P1 - HIGH** | #601-603 | `high-priority/` |
| **P2 - MEDIUM** | #569-597 | `parent-XXX/` |
| **P2 - Architecture** | #604-606 | `systemic-improvements/` |

## WORKFLOW

1. **Pick a directory** from above (claim one module group)
2. **Read all issues** in that directory
3. **Fix issues one by one**
4. **Run verification** after each fix
5. **Report progress** when done

## EXAMPLE: Fixing #599 (Duplicate Exports)

```bash
# Read the issue
cat .github/issues/critical-fixes/issue-599.md

# Edit the file
vim rust-numpy/src/modules/ma.rs
# Remove duplicate exports (all, any, append, array, etc. listed twice)

# Verify
cd rust-numpy && cargo build && cargo test

# Check specifically
grep "pub use" src/modules/ma.rs | sed 's/.*{//; s/}.*//' | tr ',' '\n' | sort | uniq -d
# Should return nothing
```

## COMPLETION REPORT

When done with your issues, report:

```bash
echo "=== COMPLETION REPORT ==="
echo "Directory worked: YOUR-DIRECTORY"
echo "Issues resolved: X"
echo "Build status: $(cd rust-numpy && cargo build --quiet 2>&1 && echo PASS || echo FAIL)"
echo "Test status: $(cd rust-numpy && cargo test --quiet 2>&1 && echo PASS || echo FAIL)"
echo "Export count: $(grep -c 'pub use' rust-numpy/src/lib.rs)"
```

## NEED HELP?

- Check `.github/issues/README.md` for issue index
- Review full prompt in `.github/issues/` (if exists)
- Verify with the checklist script above
- Document any blockers clearly

---

**REMEMBER**: Match NumPy API exactly. No more, no less. Verify after every change.
