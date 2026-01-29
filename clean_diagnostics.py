#!/usr/bin/env python3
"""Clean, deduplicate, and format compiler diagnostics.

Supports diagnostics from multiple compilers (Rust, TypeScript, Python, etc.).

Usage:
    python clean_diagnostics.py < diagnostics.json          # Default: todo list format
    python clean_diagnostics.py diagnostics.json           # Default: todo list format
    python clean_diagnostics.py --text diagnostics.json    # Traditional text format
    python clean_diagnostics.py --github diagnostics.json  # GitHub issue templates
"""


import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_short_path(resource: str, base_path: str = '/home/grant/github/pocket-tts/') -> str:
    """Get a shorter relative path from the resource."""
    if base_path in resource:
        return resource.split(base_path, 1)[1]
    # Try to make it relative to current directory as fallback
    try:
        return str(Path(resource).relative_to(Path.cwd()))
    except (ValueError, OSError):
        return resource


def extract_error_code(diag: Dict) -> str:
    """Extract error code from diagnostic (works with multiple compilers)."""
    code = diag.get('code', {})
    if isinstance(code, dict):
        return code.get('value', 'unknown')
    return str(code) if code else 'diagnostic'


def get_severity_name(severity: Optional[int]) -> str:
    """Convert severity number to name (handles different compiler formats)."""
    if severity is None:
        return "diagnostic"
    # Rust: error=8, warning=4
    if severity == 8:
        return "error"
    if severity == 4:
        return "warning"
    # TypeScript/ESLint: error=0, warning=1
    if severity == 0:
        return "error"
    if severity == 1:
        return "warning"
    # Generic
    if severity > 4:
        return "error"
    return "warning"


def extract_summary(message: str) -> str:
    """Extract the first meaningful line of the error message."""
    lines = message.split('\n')

    # Get the first line with actual error text
    for line in lines:
        line = line.strip()
        if not line or line.startswith('-->') or line.startswith('|') or line.startswith('='):
            continue
        if line.startswith('error[') or line.startswith('warning['):
            # Extract message after "error[E0308]: "
            parts = line.split(']: ', 1)
            if len(parts) > 1:
                return parts[1].strip()
        return line
    return lines[0].strip() if lines else ""


def create_diagnostic_key(diag: Dict) -> Tuple[str, str, int]:
    """Create a unique key for deduplication."""
    resource = get_short_path(diag.get('resource', ''))
    code = extract_error_code(diag)
    line = diag.get('startLineNumber', 0)
    return (resource, code, line)


def group_and_dedupe(diagnostics: List[Dict]) -> Dict[str, List[Dict]]:
    """Group diagnostics by file and deduplicate."""
    grouped = defaultdict(list)
    seen_keys = set()

    for diag in diagnostics:
        key = create_diagnostic_key(diag)
        if key not in seen_keys:
            seen_keys.add(key)
            short_path = get_short_path(diag.get('resource', ''))
            grouped[short_path].append(diag)

    return grouped


def format_diagnostic(diag: Dict) -> str:
    """Format a single diagnostic."""
    code = extract_error_code(diag)
    severity = get_severity_name(diag.get('severity', 8))
    line = diag.get('startLineNumber', '?')
    message = diag.get('message', '')

    summary = extract_summary(message)

    return f"  [{line}] {severity}[{code}]: {summary}"


def format_diagnostic_as_todo(diag: Dict) -> str:
    """Format a diagnostic as a todo item for TodoWrite."""
    file_path = get_short_path(diag.get('resource', ''))
    code = extract_error_code(diag)
    severity = get_severity_name(diag.get('severity', 8))
    line = diag.get('startLineNumber', '?')
    summary = extract_summary(diag.get('message', ''))

    # Format: Fix error[E0308] at path:line - summary
    return f"Fix {severity}[{code}] at {file_path}:{line} - {summary}"


def format_diagnostic_as_github_issue(diag: Dict, issue_number: int) -> str:
    """Format a diagnostic as a GitHub issue template."""
    file_path = get_short_path(diag.get('resource', ''))
    code = extract_error_code(diag)
    severity = get_severity_name(diag.get('severity'))
    line = diag.get('startLineNumber', '?')
    summary = extract_summary(diag.get('message', ''))
    full_message = diag.get('message', '')

    # Create a title
    title = f"{severity.capitalize()}: {summary} ({file_path}:{line})"

    # Create the issue body
    body = f"""## Issue Description

**File:** `{file_path}:{line}`
**Severity:** {severity}
**Code:** {code}

### Summary
{summary}

### Full Diagnostic Message
```
{full_message}
```

### Steps to Reproduce
1. Build/run the compiler/linter
2. See error at `{file_path}:{line}`

### Expected Behavior
Code should compile/run without this {severity}.

### Actual Behavior
{severity} occurs at `{file_path}:{line}`.

### Additional Context
- This issue was auto-generated from compiler diagnostics
- Issue Number: {issue_number}

---

## Labels
`compiler-diagnostic`
`{severity}`
`auto-generated`

## Type
`bug`

"""

    return f"""---
### Issue #{issue_number}: {title}

{body}
"""


def output_as_github_issues(diagnostics: List[Dict]) -> None:
    """Output diagnostics as GitHub issue templates."""
    deduplicated = defaultdict(list)
    seen = set()

    # Group and dedupe
    for diag in diagnostics:
        key = create_diagnostic_key(diag)
        if key not in seen:
            seen.add(key)
            file_path = get_short_path(diag.get('resource', ''))
            deduplicated[file_path].append(diag)

    # Sort by file path
    sorted_files = sorted(deduplicated.keys())

    # Count errors and warnings
    error_count = 0
    warning_count = 0
    issue_number = 1

    print("# GitHub Issue Templates from Compiler Diagnostics")
    print("# Generated by clean_diagnostics.py --github\n")
    print("Usage: Copy each issue template below and create a GitHub issue using `gh issue create` or the web UI.\n")

    for file_path in sorted_files:
        file_diags = sorted(deduplicated[file_path], key=lambda d: d.get('startLineNumber', 0))

        for diag in file_diags:
            issue_template = format_diagnostic_as_github_issue(diag, issue_number)
            print(issue_template)
            issue_number += 1

            if diag.get('severity', 8) == 8:
                error_count += 1
            else:
                warning_count += 1

    print(f"\n# Summary: {error_count} errors, {warning_count} warnings across {len(sorted_files)} files")
    print(f"# Total issues generated: {issue_number - 1}")


def output_as_todo_list(diagnostics: List[Dict]) -> None:
    """Output diagnostics as a todo list in TodoWrite format."""
    deduplicated = defaultdict(list)
    seen = set()

    # Group and dedupe
    for diag in diagnostics:
        key = create_diagnostic_key(diag)
        if key not in seen:
            seen.add(key)
            file_path = get_short_path(diag.get('resource', ''))
            deduplicated[file_path].append(diag)

    # Sort by file path
    sorted_files = sorted(deduplicated.keys())

    # Count errors and warnings
    error_count = 0
    warning_count = 0

    print("# Todo List from Rust Diagnostics")
    print("# Generated by clean_diagnostics.py --todos\n")

    for file_path in sorted_files:
        file_diags = sorted(deduplicated[file_path], key=lambda d: d.get('startLineNumber', 0))

        print(f"## File: {file_path}")

        for diag in file_diags:
            todo_item = format_diagnostic_as_todo(diag)
            print(f"- [ ] {todo_item}")

            if diag.get('severity', 8) == 8:
                error_count += 1
            else:
                warning_count += 1

        print()  # Empty line between files

    print(f"# Summary: {error_count} errors, {warning_count} warnings across {len(sorted_files)} files")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Clean, deduplicate, and format compiler diagnostics',
        epilog='Supports diagnostics from Rust, TypeScript, Python, and other compilers.'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Input JSON file containing diagnostics (reads from stdin if not provided)'
    )
    parser.add_argument(
        '--text',
        action='store_true',
        help='Output in traditional text format (default: todo list format)'
    )
    parser.add_argument(
        '--github',
        action='store_true',
        help='Output as GitHub issue templates'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'summary'],
        default='text',
        help='Output format for text mode (default: text)'
    )

    args = parser.parse_args()

    # Read JSON input
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{args.input_file}' not found", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file '{args.input_file}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            data = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
            print("\nHint: Try saving your JSON to a file and running:", file=sys.stderr)
            print("  python clean_diagnostics.py diagnostics.json", file=sys.stderr)
            sys.exit(1)

    if not isinstance(data, list):
        print("Error: Input must be a JSON array of diagnostics", file=sys.stderr)
        sys.exit(1)

    # Output as GitHub issues if requested
    if args.github:
        output_as_github_issues(data)
        return

    # Output in traditional text format if requested
    if args.text:
        grouped = group_and_dedupe(data)
        sorted_files = sorted(grouped.keys())
        error_count = 0
        warning_count = 0
        output = []

        for file_path in sorted_files:
            diagnostics = grouped[file_path]
            if not diagnostics:
                continue

            output.append(f"\n{'='*80}")
            output.append(f"File: {file_path}")
            output.append(f"{'='*80}")

            sorted_diags = sorted(
                diagnostics,
                key=lambda d: d.get('startLineNumber', 0)
            )

            for diag in sorted_diags:
                output.append(format_diagnostic(diag))
                severity = diag.get('severity', 8)
                if severity == 8:
                    error_count += 1
                else:
                    warning_count += 1

        output.append(f"\n{'='*80}")
        output.append(f"Summary: {error_count} errors, {warning_count} warnings across {len(sorted_files)} files")
        output.append(f"{'='*80}\n")
        print('\n'.join(output))
        return

    # Default: output as todo list
    output_as_todo_list(data)


if __name__ == '__main__':
    main()
