# [P2] Add automated export validation (CI + pre-commit)

**Issue #605**  
**State:** OPEN  
**URL:** https://github.com/grantjr1842/pocket-tts/issues/605  
**Created:** 2026-01-29T04:20:03Z  
**Updated:** 2026-01-29T04:20:03Z  

**Labels:** ralph/workset, ralph/status:queued, ralph/numpy:lib-exports  

---

## Summary
Implement automated checks to prevent duplicate/wildcard export problems from recurring. Catch issues in CI before merge.

## Proposed Automation

### 1. Pre-commit Hook (Local Development)

Create `.pre-commit/check-exports.sh`:

```bash
#!/bin/bash
set -e

echo "🔍 Checking rust-numpy exports..."

cd rust-numpy

# Check for duplicate exports
DUPS=$(grep "pub use" src/lib.rs | sed 's/.*pub use //' | sort | uniq -d)
if [ -n "$DUPS" ]; then
    echo "❌ Duplicate imports found:"
    echo "$DUPS"
    exit 1
fi

# Check for wildcards (warn, don't fail)
WILDCARDS=$(grep -c "pub use.*\*" src/lib.rs || true)
if [ "$WILDCARDS" -gt 0 ]; then
    echo "⚠️  Warning: $WILDCARDS wildcard exports found"
fi

# Check exports block duplicates in other files
for file in src/modules/ma.rs src/char.rs; do
    if [ -f "$file" ]; then
        # Extract export blocks and check for duplicates
        BLOCKS=$(grep -Pzo '(?s)pub use\s*\{[^}]+\}' "$file" 2>/dev/null || true)
        if [ -n "$BLOCKS" ]; then
            echo "$BLOCKS" | tr ',' '\n' | sed 's/ //g' | sort | uniq -d | while read dup; do
                if [ -n "$dup" ]; then
                    echo "❌ Duplicate in $file: $dup"
                    exit 1
                fi
            done
        fi
    fi
done

echo "✅ Export checks passed"
```

### 2. CI Workflow (GitHub Actions)

Add to `.github/workflows/rust-numpy-checks.yml`:

```yaml
name: Export Structure Checks

on:
  push:
    paths:
      - 'rust-numpy/src/lib.rs'
      - 'rust-numpy/src/**/exports.rs'
  pull_request:
    paths:
      - 'rust-numpy/src/lib.rs'
      - 'rust-numpy/src/**/exports.rs'

jobs:
  export-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Check for duplicate exports
        run: |
          DUPS=$(grep "pub use" rust-numpy/src/lib.rs | sed 's/.*pub use //' | sort | uniq -d)
          if [ -n "$DUPS" ]; then
            echo "::error::Duplicate exports found:"
            echo "$DUPS"
            exit 1
          fi
      
      - name: Check for wildcard exports
        run: |
          WILDCARDS=$(grep -n "pub use.*\*" rust-numpy/src/lib.rs || true)
          if [ -n "$WILDCARDS" ]; then
            echo "::warning::Wildcard exports found:"
            echo "$WILDCARDS"
          fi
      
      - name: Check internal modules are hidden
        run: |
          INTERNAL_MODS="kernel simd parallel profiler"
          for mod in $INTERNAL_MODS; do
            if grep -q "^pub mod $mod" rust-numpy/src/lib.rs && \
               ! grep -q "#\[doc(hidden)\]$" rust-numpy/src/lib.rs | grep -A1 "pub mod $mod"; then
              echo "::warning::Internal module '$mod' not marked #[doc(hidden)]"
            fi
          done
      
      - name: Count exports
        run: |
          COUNT=$(grep -c "pub use" rust-numpy/src/lib.rs)
          echo "Found $COUNT export statements"
          if [ "$COUNT" -lt 50 ]; then
            echo "::warning::Low export count ($COUNT) - verify API completeness"
          fi
```

### 3. Audit Script (Comprehensive)

Create `scripts/audit-exports.py`:

```python
#!/usr/bin/env python3
"""Comprehensive export audit for rust-numpy"""

import re
import sys
from pathlib import Path
from collections import defaultdict

class ExportAuditor:
    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.errors = []
        self.warnings = []
        self.exports = defaultdict(list)
    
    def check_duplicates(self, file_path: Path):
        """Check for duplicate exports in a file"""
        content = file_path.read_text()
        
        # Check pub use { ... } blocks
        for block_match in re.finditer(r'pub use\s*\{([^}]+)\}', content):
            items = [i.strip().split(' as ')[0].strip() for i in block_match.group(1).split(',')]
            seen = {}
            for item in items:
                if item in seen:
                    self.errors.append(f"{file_path}: Duplicate '{item}' in exports block")
                seen[item] = True
        
        # Check multiple imports from same module
        module_imports = defaultdict(list)
        for match in re.finditer(r'pub use\s+(\w+)::', content):
            module = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            module_imports[module].append(line_num)
        
        for module, lines in module_imports.items():
            if len(lines) > 1:
                self.warnings.append(f"{file_path}: Multiple imports from '{module}' at lines {lines}")
    
    def check_wildcards(self, file_path: Path):
        """Check for wildcard exports"""
        content = file_path.read_text()
        for match in re.finditer(r'pub use.*\*', content):
            line_num = content[:match.start()].count('\n') + 1
            self.warnings.append(f"{file_path}: Wildcard export at line {line_num}")
    
    def check_internal_modules(self, file_path: Path):
        """Check internal modules are properly marked"""
        content = file_path.read_text()
        internal_patterns = ['kernel', 'simd', 'parallel', 'profiler']
        
        for pattern in internal_patterns:
            if re.search(rf'^pub mod {pattern}\b', content, re.M):
                # Check if preceded by #[doc(hidden)]
                if not re.search(rf'#\[doc\(hidden\)\]\s*\npub mod {pattern}\b', content):
                    self.warnings.append(f"{file_path}: Internal module '{pattern}' not marked #[doc(hidden)]")
    
    def run(self):
        lib_rs = self.src_dir / 'lib.rs'
        
        print("🔍 Auditing rust-numpy exports...\n")
        
        self.check_duplicates(lib_rs)
        self.check_wildcards(lib_rs)
        self.check_internal_modules(lib_rs)
        
        # Check other files with exports
        for exports_file in self.src_dir.rglob('**/exports.rs'):
            self.check_duplicates(exports_file)
        
        # Report
        if self.warnings:
            print("⚠️  WARNINGS:")
            for w in self.warnings:
                print(f"  - {w}")
            print()
        
        if self.errors:
            print("❌ ERRORS:")
            for e in self.errors:
                print(f"  - {e}")
            return 1
        
        print("✅ All export checks passed!")
        return 0

if __name__ == "__main__":
    src = Path("rust-numpy/src")
    auditor = ExportAuditor(src)
    sys.exit(auditor.run())
```

## Implementation Checklist

- [ ] Create `.pre-commit/check-exports.sh`
- [ ] Add pre-commit hook to `.pre-commit-config.yaml`
- [ ] Create `.github/workflows/rust-numpy-checks.yml`
- [ ] Create `scripts/audit-exports.py`
- [ ] Add to CI pipeline
- [ ] Document in CONTRIBUTING.md

## Benefits

1. **Prevents Problems**: Catch duplicates before commit
2. **CI Enforcement**: Block PRs with export issues
3. **Developer Feedback**: Immediate warning on wildcard use
4. **Audit Trail**: Regular automated audits

## Related
- #599, #600 (current duplicate issues)
- #602 (wildcard exports)

