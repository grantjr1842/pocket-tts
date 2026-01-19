#!/bin/bash
# Remove a completed agent's worktree
# Usage: ./cleanup_worktree.sh <agent_id>

set -e

AGENT_ID=$1
WORKTREE_BASE="../pocket-tts-worktrees"

if [ -z "$AGENT_ID" ]; then
    echo "Usage: $0 <agent_id>"
    echo "Example: $0 2"
    exit 1
fi

WORKTREE_PATH="$WORKTREE_BASE/agent-${AGENT_ID}"

if [ ! -d "$WORKTREE_PATH" ]; then
    echo "Error: Worktree for agent-${AGENT_ID} does not exist"
    exit 1
fi

# Get issue number from task file
ISSUE_NUM=$(cat "$WORKTREE_PATH/.agent/current_task.json" 2>/dev/null | grep -o '"issue": [0-9]*' | grep -o '[0-9]*' || echo "")

# Remove worktree
git worktree remove "$WORKTREE_PATH" --force

# Prune stale worktree references
git worktree prune

# Unlock the issue if we have the issue number
if [ -n "$ISSUE_NUM" ]; then
    gh issue edit "$ISSUE_NUM" --remove-label "agent:locked-${AGENT_ID}" 2>/dev/null || true
    echo "✓ Issue #${ISSUE_NUM} unlocked"
fi

echo "✓ Worktree for agent-${AGENT_ID} removed"
echo "✓ Worktree references pruned"
