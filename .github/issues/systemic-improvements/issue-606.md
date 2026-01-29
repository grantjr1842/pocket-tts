# [P2] Implement CODEOWNERS and file locking protocol

**Issue #606**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/606  
**Created:** 2026-01-29T04:20:28Z  
**Updated:** 2026-01-29T04:20:28Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports  

---

## Summary
Implement file ownership and coordination protocols to prevent multi-agent conflicts on shared files like `lib.rs` and `Cargo.toml`.

## Problem
Multiple agents modifying the same files without coordination leads to:
- Merge conflicts
- Duplicate exports
- Inconsistent structure
- Wasted effort

## Solution: CODEOWNERS + Locking Protocol

### 1. CODEOWNERS File

Create `.github/CODEOWNERS`:

```
# Rust NumPy - File Ownership
# Lines starting with # are comments
# Format: file-pattern @owner-label

# Default ownership
* @ralph/owner:orchestrator

# Core exports - orchestrator must review
rust-numpy/src/lib.rs @ralph/owner:orchestrator
rust-numpy/Cargo.toml @ralph/owner:orchestrator

# Module ownership
rust-numpy/src/linalg/** @ralph/owner:codex
rust-numpy/src/random/** @ralph/owner:gemini
rust-numpy/src/polynomial/** @ralph/owner:zai
rust-numpy/src/fft/** @ralph/owner:orchestrator
rust-numpy/src/io/** @ralph/owner:orchestrator

# Group labels for coordination
rust-numpy/src/modules/** @ralph/numpy:submodules
rust-numpy/src/array*.rs @ralph/numpy:lib-exports
rust-numpy/src/math_ufuncs.rs @ralph/numpy:lib-exports
rust-numpy/src/reductions.rs @ralph/numpy:lib-exports
rust-numpy/src/statistics.rs @ralph/numpy:lib-exports
```

### 2. Shared File Locking Protocol

For files with `@ralph/owner:orchestrator` ownership:

**Before modifying:**
```bash
# 1. Create a lock issue
gh issue create --title "[LOCK] lib.rs - Adding exports for #<parent-issue>" \
  --body "Locking rust-numpy/src/lib.rs for modifications.

Planned changes:
- Add exports for X module
- Duration: ~2 hours

Parent issue: #<parent-issue>" \
  --label "ralph/status:blocked"

# 2. Wait for orchestrator approval (or proceed if urgent)
# 3. Make changes
# 4. Close lock issue when PR merged
gh issue close <lock-issue-number>
```

### 3. Export Change Template

PR template for export changes:

```markdown
## Export Changes

### Files Modified
- [ ] lib.rs
- [ ] Module file(s)
- [ ] Tests

### Checklist
- [ ] I ran `./scripts/audit-exports.py` - PASSED
- [ ] No duplicate exports
- [ ] No new wildcards (or justified below)
- [ ] Internal modules marked `#[doc(hidden)]`
- [ ] Lock issue created (if modifying lib.rs)

### Coordination
- Parent issue: #<issue>
- Lock issue: #<lock-issue> (if applicable)
- Related PRs: #<pr>
```

### 4. Merge Rules

Configure branch protection:

```yaml
# .github/settings.yml or repo settings
branches:
  - name: main
    protection:
      required_pull_request_reviews:
        required_approving_review_count: 1
        require_code_owner_reviews: true  # Enforce CODEOWNERS
      required_status_checks:
        contexts:
          - "Export Structure Checks"  # From CI
```

## Implementation

- [ ] Create `.github/CODEOWNERS`
- [ ] Add PR template for export changes
- [ ] Configure branch protection
- [ ] Document protocol in `rust-numpy/CONTRIBUTING.md`
- [ ] Test with orchestrator approval

## Related
- #604 (centralized exports - reduces need for coordination)
- #605 (automation - catches issues automatically)

