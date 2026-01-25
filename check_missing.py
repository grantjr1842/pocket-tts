#!/usr/bin/env python3
"""
Script to analyze which missing numpy functions are already implemented
"""

import re
import os

# Read the CSV to get missing functions
missing_functions = []
with open("rust-numpy/PARITY_BASELINE_NUMPY_2_4.csv", "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if (
            len(parts) >= 5
            and parts[0] == "numpy"
            and parts[3] == "missing"
            and parts[4] == "not_found"
            and parts[5] == "critical"
        ):
            func_name = parts[2]  # symbol name
            missing_functions.append(func_name)

print(f"Found {len(missing_functions)} missing critical functions")
print("First 20:", missing_functions[:20])

# Check which are already implemented in source files
implemented_functions = set()
source_files = []
for root, dirs, files in os.walk("rust-numpy/src"):
    for file in files:
        if file.endswith(".rs"):
            source_files.append(os.path.join(root, file))

for file_path in source_files:
    with open(file_path, "r") as f:
        content = f.read()
        # Look for pub fn function_name
        for func in missing_functions:
            if re.search(rf"pub\s+fn\s+{func}\s*\(", content):
                implemented_functions.add(func)

print(f"\nFound {len(implemented_functions)} functions already implemented:")
for func in sorted(implemented_functions):
    print(f"  {func}")

truly_missing = set(missing_functions) - implemented_functions
print(f"\n{len(truly_missing)} functions still need implementation:")
for func in sorted(truly_missing):
    print(f"  {func}")
