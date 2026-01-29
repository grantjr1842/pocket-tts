#!/bin/bash
# Pre-commit hook to check exports
set -e

cd rust-numpy

echo "Checking exports..."

# Check for duplicate exports
echo "  - Checking for duplicate exports..."
if grep -E "^pub use" src/lib.rs | sort | uniq -d > /dev/null 2>&1; then
  echo "❌ Duplicate exports found in lib.rs"
  grep -E "^pub use" src/lib.rs | sort | uniq -d
  exit 1
fi

# Check for wildcard exports
echo "  - Checking for wildcard exports..."
WILDCARDS=$(grep -c "pub use.*::\*" src/lib.rs || echo "0")
if [ "$WILDCARDS" -gt 0 ]; then
  echo "⚠️  $WILDCARDS wildcard exports found:"
  grep -n "pub use.*::\*" src/lib.rs
fi

echo "✅ Export checks passed"
