#!/bin/bash
# Create a git worktree for a new agent to work on an issue
# Usage: ./create_worktree.sh <agent_id> <issue_num>

set -e

AGENT_ID=$1
ISSUE_NUM=$2
WORKTREE_BASE="../pocket-tts-worktrees"

if [ -z "$AGENT_ID" ] || [ -z "$ISSUE_NUM" ]; then
    echo "Usage: $0 <agent_id> <issue_num>"
    echo "Example: $0 2 17"
    exit 1
fi

# Create worktree directory if needed
mkdir -p "$WORKTREE_BASE"

# Generate branch name
BRANCH="feat/${ISSUE_NUM}-agent-${AGENT_ID}"

# Check if worktree already exists
if [ -d "$WORKTREE_BASE/agent-${AGENT_ID}" ]; then
    echo "Error: Worktree for agent-${AGENT_ID} already exists"
    echo "Use cleanup_worktree.sh to remove it first"
    exit 1
fi

# Ensure we're up to date
git fetch origin main

# Create worktree with new branch from main
git worktree add "$WORKTREE_BASE/agent-${AGENT_ID}" -b "$BRANCH" origin/main

# Initialize agent state in worktree
mkdir -p "$WORKTREE_BASE/agent-${AGENT_ID}/.agent"
cat > "$WORKTREE_BASE/agent-${AGENT_ID}/.agent/current_task.json" << EOF
{
  "issue": ${ISSUE_NUM},
  "agent_id": "${AGENT_ID}",
  "branch": "${BRANCH}",
  "started": "$(date -Iseconds)",
  "status": "initialized"
}
EOF

# Create status file
cat > "$WORKTREE_BASE/agent-${AGENT_ID}/.agent/status.json" << EOF
{
  "agent_id": "${AGENT_ID}",
  "issue": ${ISSUE_NUM},
  "status": "ready",
  "progress": "0/0 tasks",
  "last_update": "$(date -Iseconds)",
  "errors": []
}
EOF

# Lock the issue
gh issue edit "$ISSUE_NUM" --remove-label "agent:pending" --add-label "agent:locked-${AGENT_ID}" 2>/dev/null || true

echo "✓ Worktree created: $WORKTREE_BASE/agent-${AGENT_ID}"
echo "✓ Branch: $BRANCH"
echo "✓ Issue #${ISSUE_NUM} locked for agent-${AGENT_ID}"
echo ""
echo "To start working:"
echo "  cd $WORKTREE_BASE/agent-${AGENT_ID}"
