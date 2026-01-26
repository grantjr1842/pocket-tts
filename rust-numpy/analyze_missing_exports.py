#!/usr/bin/env python3
"""
Script to analyze which functions with namespace_mismatch status
are NOT exported in lib.rs
"""

import csv

# Read the CSV file
missing_exports = []
with open('rust-numpy/PARITY_BASELINE_NUMPY_2_4.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['status'] == 'namespace_mismatch' and row['namespace'] == 'numpy':
            missing_exports.append({
                'function': row['symbol'],
                'namespace': row['namespace'],
                'evidence': row['evidence']
            })

# Check lib.rs to see which are already exported
lib_rs_exports = set()
with open('rust-numpy/src/lib.rs', 'r') as f:
    content = f.read()
    # Look for pub use statements
    import re
    pub_use_pattern = r'pub use\s+crate::\w+([a-zA-Z_]+)::\s*\{([^}]+)\}'
    matches = re.findall(pub_use_pattern, content)
    for match in matches:
        module = match[0]
        exports = match[1].strip()
        # Split by commas and extract individual function names
        for export in exports.split(','):
            export = export.strip()
            if export and not export.startswith('*'):
                lib_rs_exports.add(export)

print("=" * 80)
print("Functions with namespace_mismatch status:")
print("=" * 80)

# Categorize by module
by_module = {}
for func in missing_exports:
    module = func['namespace']
    if module not in by_module:
        by_module[module] = []
    by_module[module].append(func['function'])

print(f"\nTotal: {len(missing_exports)} functions")
print("\nBy module:")
for module, funcs in sorted(by_module.items()):
    print(f"\n  {module}: {len(funcs)} functions")
    for func in funcs:
        status = "✓ EXPORTED" if func in lib_rs_exports else "✗ NOT EXPORTED"
        print(f"    - {func:30s} {status}")

print("\n" + "=" * 80)
print("Functions NOT exported in lib.rs:")
print("=" * 80)

not_exported = []
for func in missing_exports:
    if func['function'] not in lib_rs_exports:
        not_exported.append(func)

print(f"\nTotal NOT exported: {len(not_exported)}")
print("\nDetailed list:")
for func in not_exported:
    print(f"  - {func['function']} (from {func['namespace']})")

# Save to file for reference
with open('rust-numpy/MISSING_EXPORTS.txt', 'w') as f:
    f.write("Functions that need to be exported from lib.rs:\n\n")
    for func in not_exported:
        f.write(f"- {func['function']} (from {func['namespace']})\n")
