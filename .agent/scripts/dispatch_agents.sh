#!/bin/bash
# Dispatch multiple agents to work on pending issues
# Usage: ./dispatch_agents.sh [num_agents] [--dry-run]

set -e

NUM_AGENTS=${1:-2}
DRY_RUN=false

if [ "$1" = "--dry-run" ] || [ "$2" = "--dry-run" ]; then
    DRY_RUN=true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Agent Dispatch ==="
echo "Agents to dispatch: $NUM_AGENTS"
echo ""

# Fetch pending issues
PENDING=$(gh issue list --assignee @me --label "agent:pending" --json number,title --limit "$NUM_AGENTS")

if [ "$PENDING" = "[]" ]; then
    echo "No pending issues available"
    exit 0
fi

# Parse issues and dispatch
AGENT_ID=2  # Start at 2 (agent-1 is coordinator/main)
echo "$PENDING" | jq -c '.[]' | while read -r issue; do
    ISSUE_NUM=$(echo "$issue" | jq -r '.number')
    ISSUE_TITLE=$(echo "$issue" | jq -r '.title')

    echo "Issue #$ISSUE_NUM: $ISSUE_TITLE"

    if [ "$DRY_RUN" = true ]; then
        echo "  [dry-run] Would dispatch to agent-$AGENT_ID"
    else
        echo "  Dispatching to agent-$AGENT_ID..."
        "$SCRIPT_DIR/create_worktree.sh" "$AGENT_ID" "$ISSUE_NUM"
    fi

    echo ""
    AGENT_ID=$((AGENT_ID + 1))
done

echo "=== Dispatch Complete ==="
if [ "$DRY_RUN" = true ]; then
    echo "(dry-run mode - no changes made)"
fi
