#!/bin/bash
# File Locking Protocol for Multi-Agent Coordination
# Usage: ./scripts/file-lock.sh <file-path> <issue-number> [duration-hours]

set -e

FILE_PATH="$1"
ISSUE_NUM="$2"
DURATION="${3:-2}"

if [ -z "$FILE_PATH" ] || [ -z "$ISSUE_NUM" ]; then
    echo "Usage: $0 <file-path> <issue-number> [duration-hours]"
    echo "Example: $0 rust-numpy/src/lib.rs 123 2"
    exit 1
fi

LOCK_ISSUE_TITLE="[LOCK] $FILE_PATH - Issue #$ISSUE_NUM"
LOCK_ISSUE_BODY="## File Lock Request

**File:** \`$FILE_PATH\`
**Requested by:** Issue #$ISSUE_NUM
**Duration:** ~${DURATION} hours
**Requested at:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Planned Changes
<!-- Describe planned modifications -->
- 

## Checklist
- [ ] Lock acquired
- [ ] Changes implemented
- [ ] PR created
- [ ] Lock released (close this issue)

---
*This is an automated file lock for multi-agent coordination.*"

echo "Creating file lock issue..."
LOCK_ISSUE=$(gh issue create \
    --title "$LOCK_ISSUE_TITLE" \
    --body "$LOCK_ISSUE_BODY" \
    --label "ralph/status:blocked,ralph/blocked:needs-decision" \
    --json number \
    --jq '.number')

echo "✅ File lock issue created: #$LOCK_ISSUE"
echo ""
echo "To release the lock when done:"
echo "  gh issue close $LOCK_ISSUE --comment \"Changes completed in PR #<pr-number>\""
