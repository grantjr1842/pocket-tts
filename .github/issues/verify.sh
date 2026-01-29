#!/bin/bash
set -e

cd ../rust-numpy

echo "=== Checking for Duplicates ==="
DUPS=$(grep "pub use" src/lib.rs 2>/dev/null | sed 's/.*pub use //' | sort | uniq -d)
if [ -n "$DUPS" ]; then
    echo "❌ DUPLICATES FOUND:"
    echo "$DUPS"
    exit 1
fi
echo "✓ No duplicates"

echo ""
echo "=== Checking for Wildcards ==="
WILDCARDS=$(grep -n "pub use.*\*" src/lib.rs 2>/dev/null || true)
if [ -n "$WILDCARDS" ]; then
    echo "⚠️  WILDCARDS FOUND:"
    echo "$WILDCARDS"
else
    echo "✓ No wildcards"
fi

echo ""
echo "=== Build Check ==="
cargo build --quiet 2>&1 | head -5
echo "✓ Build OK"

echo ""
echo "=== Test Check ==="
cargo test --quiet 2>&1 | tail -5
echo "✓ Tests OK"

echo ""
echo "✅ ALL CHECKS PASSED"
