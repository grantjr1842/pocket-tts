#!/bin/bash
# List all active agent worktrees and their status
# Usage: ./list_agents.sh

WORKTREE_BASE="../pocket-tts-worktrees"

echo "=== Active Agent Worktrees ==="
echo ""

# List git worktrees
git worktree list

echo ""
echo "=== Agent Status ==="
echo ""

# Check for agent status files
for dir in "$WORKTREE_BASE"/agent-*/; do
    if [ -d "$dir" ]; then
        AGENT_ID=$(basename "$dir")

        if [ -f "$dir/.agent/status.json" ]; then
            echo "[$AGENT_ID]"
            cat "$dir/.agent/status.json" | grep -E '"(status|issue|progress|last_update)"' | sed 's/^/  /'
            echo ""
        else
            echo "[$AGENT_ID] - No status file"
        fi
    fi
done

# Count active agents
AGENT_COUNT=$(ls -d "$WORKTREE_BASE"/agent-*/ 2>/dev/null | wc -l || echo 0)
echo "Total active agents: $AGENT_COUNT"
