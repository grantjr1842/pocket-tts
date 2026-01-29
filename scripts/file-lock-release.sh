#!/bin/bash
# Release a file lock
# Usage: ./scripts/file-lock-release.sh <lock-issue-number> [pr-number]

set -e

LOCK_ISSUE="$1"
PR_NUM="$2"

if [ -z "$LOCK_ISSUE" ]; then
    echo "Usage: $0 <lock-issue-number> [pr-number]"
    echo "Example: $0 456 123"
    exit 1
fi

COMMENT="File lock released."
if [ -n "$PR_NUM" ]; then
    COMMENT="File lock released. Changes completed in PR #$PR_NUM"
fi

echo "Releasing file lock #$LOCK_ISSUE..."
gh issue close "$LOCK_ISSUE" --comment "$COMMENT"

echo "✅ File lock #$LOCK_ISSUE released"
