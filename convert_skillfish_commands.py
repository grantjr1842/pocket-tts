#!/usr/bin/env python3
"""Convert skillfish add commands to add-skill format."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Optional

# List of URLs to process
URLS = [
    "https://mcpmarket.com/tools/skills/prompt-lookup",
    "https://mcpmarket.com/tools/skills/dify-frontend-testing",
    "https://mcpmarket.com/tools/skills/estimation-scope-analysis",
    "https://mcpmarket.com/tools/skills/context-gathering-protocol",
    "https://mcpmarket.com/tools/skills/carplay-framework-for-ios",
    "https://mcpmarket.com/tools/skills/carplay-ios-development",
    "https://mcpmarket.com/tools/skills/precision-implementation-planning",
    "https://mcpmarket.com/tools/skills/ai-task-delegation-principle",
    "https://mcpmarket.com/tools/skills/concise-agent-communication",
    "https://mcpmarket.com/tools/skills/execution-manager",
    "https://mcpmarket.com/tools/skills/git-worktree-manager-1769320301351",
    "https://mcpmarket.com/tools/skills/playwright-browser-automation-1769320286928",
    "https://mcpmarket.com/tools/skills/implementation-spec-writer",
    "https://mcpmarket.com/tools/skills/context-optimization-1",
    "https://mcpmarket.com/tools/skills/multi-agent-architecture-patterns-5",
    "https://mcpmarket.com/tools/skills/filesystem-context-engineering",
    "https://mcpmarket.com/tools/skills/rust-skill-creator",
    "https://mcpmarket.com/tools/skills/rust-web-domain-expert",
    "https://mcpmarket.com/tools/skills/skill-creation-template-8",
    "https://mcpmarket.com/tools/skills/rust-coding-guidelines",
    "https://mcpmarket.com/tools/skills/cloud-native-rust-development",
    "https://mcpmarket.com/tools/skills/rust-anti-pattern-refactor",
    "https://mcpmarket.com/tools/skills/domain-error-strategy-for-rust",
    "https://mcpmarket.com/tools/skills/rust-embedded-no-std-development",
    "https://mcpmarket.com/tools/skills/rust-mental-models",
    "https://mcpmarket.com/tools/skills/rust-performance-optimization",
    "https://mcpmarket.com/tools/skills/rust-unsafe-code-checker",
    "https://mcpmarket.com/tools/skills/rust-mutability-master",
    "https://mcpmarket.com/tools/skills/rust-ecosystem-dependency-manager",
    "https://mcpmarket.com/tools/skills/rust-concurrency-async-expert",
    "https://mcpmarket.com/tools/skills/rust-fintech-domain-expert",
    "https://mcpmarket.com/tools/skills/rust-ownership-lifetimes-expert",
    "https://mcpmarket.com/tools/skills/rust-learner",
    "https://mcpmarket.com/tools/skills/rust-zero-cost-abstractions",
    "https://mcpmarket.com/tools/skills/rust-resource-lifecycle-management",
    "https://mcpmarket.com/tools/skills/rust-resource-smart-pointer-guide",
    "https://mcpmarket.com/tools/skills/rust-question-router-meta-cognition-skill",
    "https://mcpmarket.com/tools/skills/rust-iot-domain-expert",
    "https://mcpmarket.com/tools/skills/rust-cli-development",
    "https://mcpmarket.com/tools/skills/rust-domain-modeling-ddd",
    "https://mcpmarket.com/tools/skills/rust-community-news-reports",
    "https://mcpmarket.com/tools/skills/functional-core-imperative-shell-fcis-architect",
    "https://mcpmarket.com/tools/skills/rust-machine-learning-ai",
    "https://mcpmarket.com/tools/skills/ruff-python-linting",
    "https://mcpmarket.com/tools/skills/github-labels-manager",
    "https://mcpmarket.com/tools/skills/agent-handoff-markers",
    "https://mcpmarket.com/tools/skills/shell-expert",
    "https://mcpmarket.com/tools/skills/slash-command-context-patterns",
]


def check_network_connectivity() -> bool:
    """Check if we have basic network connectivity."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--connect-timeout", "5", "https://www.google.com"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def fetch_page(url: str, max_retries: int = 3) -> Optional[str]:
    """Fetch page content using curl with retry logic.

    Returns None if all retries fail, otherwise returns the HTML content.
    """
    curl_cmd = [
        "curl",
        "-sL",  # Silent and follow redirects
        "--connect-timeout", "10",  # Connection timeout
        "--max-time", "30",  # Total operation timeout
        "--retry", str(max_retries),
        "--retry-delay", "2",
        "--retry-max-time", "60",
        "-H", "User-Agent: Mozilla/5.0 (compatible; SkillConverter/1.0)",
        url,
    ]

    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return result.stdout
            else:
                # Provide detailed error information
                stderr = result.stderr.strip() if result.stderr else "No error details"
                if attempt < max_retries:
                    print(
                        f"  ⚠ Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {stderr}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  ✗ All attempts failed for {url}\n"
                        f"    Return code: {result.returncode}\n"
                        f"    Error: {stderr}",
                        file=sys.stderr,
                    )
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                print(
                    f"  ⚠ Attempt {attempt + 1}/{max_retries + 1} timed out for {url}",
                    file=sys.stderr,
                )
            else:
                print(f"  ✗ All attempts timed out for {url}", file=sys.stderr)
        except OSError as e:
            # Network or system-level errors - don't retry
            print(
                f"  ✗ Network error for {url}: {e}\n"
                f"    This usually indicates no internet connectivity or DNS issues.",
                file=sys.stderr,
            )
            return None

    return None


def extract_skillfish_command(html: str) -> Optional[str]:
    """Extract the skillfish add command from HTML content."""
    # Look for the "Install with skill.fish" section and extract the command
    # Pattern: npx skillfish add <repo> <skill-name>
    pattern = r"npx skillfish add\s+(\S+)\s+(\S+)"
    match = re.search(pattern, html)
    if match:
        repo = match.group(1)
        skill_name = match.group(2)
        return f"npx add-skill --global {repo} --skill {skill_name}"
    return None


def main() -> None:
    """Process all URLs and output converted commands."""
    # Check network connectivity first
    print("Checking network connectivity...", file=sys.stderr)
    if not check_network_connectivity():
        print(
            "✗ Network connectivity check failed!\n"
            "  Please ensure you have internet access and DNS resolution is working.\n"
            "  Common fixes:\n"
            "    - Check your internet connection\n"
            "    - Try running from a different terminal/environment\n"
            "    - If using WSL, network may need to be configured\n"
            "    - Check firewall settings\n"
            "  Exiting early.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    print("✓ Network connectivity confirmed\n", file=sys.stderr)

    converted_commands: list[str] = []
    failed_urls: list[str] = []

    for url in URLS:
        print(f"Processing: {url}", file=sys.stderr)
        html = fetch_page(url)
        if html:
            command = extract_skillfish_command(html)
            if command:
                converted_commands.append(command)
                print(f"  ✓ {command}", file=sys.stderr)
            else:
                failed_urls.append(url)
                print(f"  ✗ Could not find skillfish command", file=sys.stderr)
        else:
            failed_urls.append(url)

    # Output all converted commands
    print("\n=== Converted Commands ===")
    for cmd in converted_commands:
        print(cmd)

    # Output summary
    print(f"\n=== Summary ===", file=sys.stderr)
    print(f"Successfully converted: {len(converted_commands)}/{len(URLS)}", file=sys.stderr)
    if failed_urls:
        print(f"\nFailed URLs: {len(failed_urls)}", file=sys.stderr)
        for url in failed_urls:
            print(f"  - {url}", file=sys.stderr)


if __name__ == "__main__":
    main()
